"""Temporal, cluster-aware splitting, and a manifest that freezes the result.

The protocol is fixed in `docs/evaluation.md`, written before any model ran. This
implements it.

    train  published_at <= T - 14 days
    val    T - 14 days  <  published_at <= T - 7 days
    test   published_at >  T - 7 days

Three properties make this harder than slicing on a date, and each corresponds to
a way the resulting numbers would otherwise be wrong.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

Split = Literal["train", "val", "test"]

TRAIN_CUTOFF_DAYS = 14
VAL_CUTOFF_DAYS = 7

# A test set that is overwhelmingly one publisher measures that publisher, not
# the task — and with leakage rates ranging from 4% to 100% by source, it would
# also measure a wildly different amount of leakage than the corpus as a whole.
MAX_TEST_SOURCE_SHARE = 0.80

# Below this, results are labelled provisional per the protocol.
MIN_TEST_ROWS = 150


@dataclass(frozen=True, slots=True)
class Article:
    """The minimum a split needs to know about a row."""

    article_key: str
    cluster_id: str
    published_at: datetime
    source: str
    kanal: str


@dataclass
class SplitManifest:
    """A frozen record of one split, identified by content hash.

    An evaluation result is only valid if it names the manifest it ran against.
    Two candidates compared on different splits are not compared at all, and
    without a hash that mistake is invisible — the numbers still look
    comparable.
    """

    created_at: str
    anchor: str
    train_cutoff: str
    val_cutoff: str
    assignment: dict[str, Split]
    cluster_of: dict[str, str]
    counts: dict[str, int]
    source_mix: dict[str, dict[str, int]]
    kanal_mix: dict[str, dict[str, int]]
    clusters_spanning_boundary: int
    warnings: list[str] = field(default_factory=list)

    @property
    def hash(self) -> str:
        """SHA-256 over the assignment and the parameters that produced it.

        Deliberately excludes `created_at`: the same corpus split with the same
        anchor must produce the same hash, or the identifier is a timestamp
        wearing a hash's clothes.
        """
        payload = json.dumps(
            {
                "anchor": self.anchor,
                "train_cutoff": self.train_cutoff,
                "val_cutoff": self.val_cutoff,
                "assignment": dict(sorted(self.assignment.items())),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def is_provisional(self) -> bool:
        return self.counts.get("test", 0) < MIN_TEST_ROWS

    def keys(self, split: Split) -> list[str]:
        return sorted(k for k, s in self.assignment.items() if s == split)

    def to_dict(self) -> dict[str, object]:
        return {
            "hash": self.hash,
            "created_at": self.created_at,
            "anchor": self.anchor,
            "train_cutoff": self.train_cutoff,
            "val_cutoff": self.val_cutoff,
            "counts": self.counts,
            "source_mix": self.source_mix,
            "kanal_mix": self.kanal_mix,
            "clusters_spanning_boundary": self.clusters_spanning_boundary,
            "is_provisional": self.is_provisional,
            "warnings": self.warnings,
            "assignment": self.assignment,
            "cluster_of": self.cluster_of,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=False),
            encoding="utf-8",
        )
        return path

    def summary(self) -> str:
        lines = [
            f"split {self.hash[:12]}  anchor={self.anchor}",
            "  " + "  ".join(f"{s}={self.counts.get(s, 0)}" for s in ("train", "val", "test")),
        ]
        for split in ("train", "val", "test"):
            mix = self.source_mix.get(split, {})
            total = sum(mix.values()) or 1
            share = ", ".join(
                f"{src} {n / total:.0%}" for src, n in sorted(mix.items(), key=lambda x: -x[1])
            )
            lines.append(f"  {split:<6} {share}")
        if self.clusters_spanning_boundary:
            lines.append(
                f"  {self.clusters_spanning_boundary} cluster(s) spanned a boundary "
                f"and were assigned whole"
            )
        for w in self.warnings:
            lines.append(f"  ! {w}")
        if self.is_provisional:
            lines.append(
                f"  ! PROVISIONAL — test has {self.counts.get('test', 0)} rows, "
                f"below the {MIN_TEST_ROWS} the protocol requires"
            )
        return "\n".join(lines)


def temporal_split(
    articles: list[Article],
    *,
    anchor: datetime | None = None,
    train_cutoff_days: int = TRAIN_CUTOFF_DAYS,
    val_cutoff_days: int = VAL_CUTOFF_DAYS,
) -> SplitManifest:
    """Split by publication time, assigning whole clusters.

    **Clusters are assigned by their most recent article.** A cluster spanning a
    boundary goes entirely to the later split, and the direction matters:

    - assigning it to the *later* split puts an older copy into val or test,
      which is harmless — the test set simply contains an article that was
      written earlier;
    - assigning it to the *earlier* split would put a test-period article into
      training, which is a model learning from the period it is about to be
      evaluated on.

    Only one of those two mistakes inflates a score, so the choice is not
    arbitrary.
    """
    if not articles:
        raise ValueError("cannot split an empty corpus")

    now = anchor or max(a.published_at for a in articles)
    train_cutoff = now - timedelta(days=train_cutoff_days)
    val_cutoff = now - timedelta(days=val_cutoff_days)

    def split_for(when: datetime) -> Split:
        if when <= train_cutoff:
            return "train"
        if when <= val_cutoff:
            return "val"
        return "test"

    # Group by cluster, then decide once per cluster.
    by_cluster: dict[str, list[Article]] = {}
    for article in articles:
        by_cluster.setdefault(article.cluster_id, []).append(article)

    assignment: dict[str, Split] = {}
    spanning = 0

    for members in by_cluster.values():
        splits_present = {split_for(m.published_at) for m in members}
        if len(splits_present) > 1:
            spanning += 1

        decided = split_for(max(m.published_at for m in members))
        for member in members:
            assignment[member.article_key] = decided

    counts: dict[str, int] = dict(Counter[str](assignment.values()))
    by_key = {a.article_key: a for a in articles}

    source_mix: dict[str, dict[str, int]] = {}
    kanal_mix: dict[str, dict[str, int]] = {}
    for key, split in assignment.items():
        article = by_key[key]
        source_mix.setdefault(split, {})
        source_mix[split][article.source] = source_mix[split].get(article.source, 0) + 1
        kanal_mix.setdefault(split, {})
        kanal_mix[split][article.kanal] = kanal_mix[split].get(article.kanal, 0) + 1

    warnings: list[str] = []

    # The evergreen check from the protocol. ANTARA files months-old explainers
    # into its live feeds, so a naive split pushes it wholly into train and
    # leaves a test set that is effectively one publisher — with a completely
    # different leakage rate from the corpus average.
    test_mix = source_mix.get("test", {})
    test_total = sum(test_mix.values())
    if test_total:
        top_source, top_n = max(test_mix.items(), key=lambda x: x[1])
        share = top_n / test_total
        if share > MAX_TEST_SOURCE_SHARE:
            warnings.append(
                f"test set is {share:.0%} {top_source} — a test set that is "
                f"effectively one publisher measures that publisher, not the task"
            )

    for split in ("train", "val", "test"):
        present = len(kanal_mix.get(split, {}))
        if present < 8:
            warnings.append(f"{split} covers only {present} of 8 classes")

    # The pathology that caught this project out, and that the composition check
    # above does not see.
    #
    # A temporal split assumes the corpus spans meaningfully more time than the
    # split windows. Measured on the first real run: train=163, val=55,
    # test=1077 — the test set was 83% of the corpus, because the *collection*
    # period was two days while `published_at` ranged over a year. That range
    # came almost entirely from ANTARA's evergreen explainers, not from history.
    #
    # The split was technically correct and practically useless: 163 training
    # rows, and a train set 71% leakage-free against a test set only 8% clean.
    # Nothing about the assignment was wrong, so nothing else would have flagged
    # it.
    train_n = counts.get("train", 0)
    test_n = counts.get("test", 0)
    if test_n > train_n:
        warnings.append(
            f"test ({test_n}) is larger than train ({train_n}) — the corpus does "
            f"not yet span enough *collection* time for these windows. Old "
            f"published_at values from evergreen content are not history"
        )

    return SplitManifest(
        created_at=datetime.now(tz=now.tzinfo).isoformat(),
        anchor=now.isoformat(),
        train_cutoff=train_cutoff.isoformat(),
        val_cutoff=val_cutoff.isoformat(),
        assignment=assignment,
        cluster_of={a.article_key: a.cluster_id for a in articles},
        counts=dict(counts),
        source_mix=source_mix,
        kanal_mix=kanal_mix,
        clusters_spanning_boundary=spanning,
        warnings=warnings,
    )


def random_split(
    articles: list[Article],
    *,
    seed: int = 42,
    train_share: float = 0.7,
    val_share: float = 0.15,
) -> SplitManifest:
    """A random, cluster-aware split — for comparison only.

    Exists so the inflation from random splitting can be *quantified* rather than
    described. The protocol requires both numbers to be reported, because the gap
    between them is what most published figures are unknowingly quoting.

    Still cluster-aware: comparing a cluster-aware temporal split against a
    row-wise random split would conflate two effects, and the interesting one is
    the temporal ordering.
    """
    if not articles:
        raise ValueError("cannot split an empty corpus")

    clusters = sorted({a.cluster_id for a in articles})
    rng = random.Random(seed)
    rng.shuffle(clusters)

    n = len(clusters)
    train_end = int(n * train_share)
    val_end = train_end + int(n * val_share)

    cluster_split: dict[str, Split] = {}
    for i, cluster_id in enumerate(clusters):
        if i < train_end:
            cluster_split[cluster_id] = "train"
        elif i < val_end:
            cluster_split[cluster_id] = "val"
        else:
            cluster_split[cluster_id] = "test"

    assignment = {a.article_key: cluster_split[a.cluster_id] for a in articles}
    counts: dict[str, int] = dict(Counter[str](assignment.values()))

    source_mix: dict[str, dict[str, int]] = {}
    kanal_mix: dict[str, dict[str, int]] = {}
    for article in articles:
        split = assignment[article.article_key]
        source_mix.setdefault(split, {})
        source_mix[split][article.source] = source_mix[split].get(article.source, 0) + 1
        kanal_mix.setdefault(split, {})
        kanal_mix[split][article.kanal] = kanal_mix[split].get(article.kanal, 0) + 1

    latest = max(a.published_at for a in articles)
    return SplitManifest(
        created_at=datetime.now(tz=latest.tzinfo).isoformat(),
        anchor=f"random:seed={seed}",
        train_cutoff="n/a",
        val_cutoff="n/a",
        assignment=assignment,
        cluster_of={a.article_key: a.cluster_id for a in articles},
        counts=dict(counts),
        source_mix=source_mix,
        kanal_mix=kanal_mix,
        clusters_spanning_boundary=0,
        warnings=["random split — for measuring the inflation, never for reporting a result"],
    )


def assert_no_cluster_leak(manifest: SplitManifest) -> None:
    """Raise if any cluster appears in more than one split.

    The one invariant the whole evaluation rests on. Called by the tests, and
    cheap enough to call again before any training run.
    """
    seen: dict[str, Split] = {}
    for key, split in manifest.assignment.items():
        cluster_id = manifest.cluster_of[key]
        if cluster_id in seen and seen[cluster_id] != split:
            raise AssertionError(
                f"cluster {cluster_id!r} spans {seen[cluster_id]} and {split} — "
                f"a story in both train and test means memorisation is being "
                f"scored as generalisation"
            )
        seen[cluster_id] = split
