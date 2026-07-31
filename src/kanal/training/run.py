"""From the warehouse to a promoted champion, in one pass.

Deliberately one function rather than a notebook. Every step it takes — build a
split, fit the candidates, score them, run the gate, record the decision — is a
step that has to happen identically when this runs unattended in CI, and a
notebook is the standard way for those to drift apart.

Two choices here are worth stating because both could reasonably go the other
way.

**The incumbent is whatever the registry says, not the best candidate in this
run.** A run that promoted its own winner would compare challengers against each
other and never against what is actually serving — which is the comparison that
decides whether to change anything.

**A run that promotes nothing is a success, not a failure.** It exits zero, logs
the refusal, and leaves the champion alone. Treating "no promotion" as an error
teaches whoever watches CI to ignore red, and the whole point of the gate is that
most challengers should not pass it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kanal.data.dedup import cluster
from kanal.data.splits import (
    TRAIN_CUTOFF_DAYS,
    VAL_CUTOFF_DAYS,
    Article,
    SplitManifest,
    assert_no_cluster_leak,
    random_split,
    temporal_split,
)
from kanal.eval.harness import Dataset, build_dataset, compare, score_candidate
from kanal.features.text import Example
from kanal.models.base import Candidate
from kanal.models.majority import MajorityClass
from kanal.models.tfidf import TfidfLinearSVC
from kanal.registry.artifact import load, save
from kanal.registry.promote import Incumbent, Verdict, evaluate_gate
from kanal.registry.store import CHAMPION, AliasNotSet, Decision, Registry

log = logging.getLogger(__name__)

DEFAULT_REGISTRY = Path("data/registry")

# The protocol fixes 14/7 days, and those numbers assumed a corpus with weeks of
# collection history. This one has days.
#
# When the default produces an unusable split, the fallback windows are derived
# from the operational question rather than chosen from a table of results:
# **classify the next day's news given everything before it.** That gives a
# one-day test window, a one-day validation window, and training on the rest.
# Nothing about those numbers refers to a score.
#
# The distinction matters because narrowing the window does not only give the
# model more data — it makes the task easier. Predicting twelve hours ahead is
# not the same problem as predicting seven days ahead, so macro-F1 measured at
# one horizon cannot be compared with macro-F1 at another. Every run records the
# horizon it used for exactly this reason.
FALLBACK_TRAIN_CUTOFF_DAYS = 2.0
FALLBACK_VAL_CUTOFF_DAYS = 1.0


@dataclass
class TrainingRun:
    """What one training pass produced."""

    split_hash: str
    n_train: int
    n_val: int
    n_test: int
    is_provisional: bool
    split_warnings: list[str]
    # The prediction horizon, in days. Recorded because macro-F1 at one horizon
    # is not comparable with macro-F1 at another, and without this on the run a
    # later reader has no way to know.
    horizon_days: float = 7.0
    used_fallback_window: bool = False
    candidates: list[tuple[str, float, float, float]] = field(default_factory=list)
    decision: Decision | None = None
    verdict: str = "NONE"
    reasons: list[str] = field(default_factory=list)
    champion_before: str | None = None
    champion_after: str | None = None

    @property
    def promoted(self) -> bool:
        return self.verdict == str(Verdict.PROMOTE)

    def summary(self) -> str:
        lines = [
            f"split {self.split_hash[:12]}  "
            f"train={self.n_train} val={self.n_val} test={self.n_test}  "
            f"horizon={self.horizon_days:g}d",
        ]
        if self.used_fallback_window:
            lines.append(
                f"  window narrowed to {self.horizon_days:g}d — the protocol's 14/7 "
                f"assumed weeks of collection history. Scores at this horizon are "
                f"NOT comparable with scores at another"
            )
        if self.is_provisional:
            lines.append("  PROVISIONAL — the test set is below the size the protocol requires")
        for warning in self.split_warnings:
            lines.append(f"  ! {warning}")

        for name, macro_f1, ece, usd in self.candidates:
            lines.append(f"  {name:<18} macro-F1 {macro_f1:.4f}  ECE {ece:.4f}  ${usd:.6f}/1k")

        lines.append(f"  verdict: {self.verdict}")
        for reason in self.reasons:
            lines.append(f"    refused: {reason}")

        if self.promoted:
            lines.append(f"  champion: {self.champion_before or 'none'} -> {self.champion_after}")
        else:
            lines.append(f"  champion unchanged: {self.champion_before or 'none'}")
        return "\n".join(lines)


def _load_corpus() -> tuple[list[Article], dict[str, Example], dict[str, str]]:
    from kanal.warehouse.duck import reader
    from kanal.warehouse.loader import default_db_path

    with reader(default_db_path()) as connection:
        rows = connection.execute("""
            SELECT article_key, title, coalesce(summary, ''), published_at, source, kanal
            FROM fct_articles
            WHERE published_at IS NOT NULL AND title IS NOT NULL AND title <> ''
            ORDER BY published_at
        """).fetchall()

    if not rows:
        raise RuntimeError(
            "the warehouse holds no articles — run `kanal load` and `dbt build` first"
        )

    report = cluster([(str(r[0]), str(r[1])) for r in rows])
    articles = [
        Article(
            article_key=str(r[0]),
            cluster_id=report.cluster_of[str(r[0])],
            published_at=r[3].replace(tzinfo=UTC),
            source=str(r[4]),
            kanal=str(r[5]),
        )
        for r in rows
    ]
    examples = {str(r[0]): Example(title=str(r[1]), summary=str(r[2])) for r in rows}
    labels = {str(r[0]): str(r[5]) for r in rows}
    return articles, examples, labels


def _incumbent(registry: Registry, split_hash: str) -> Incumbent | None:
    """The champion as the registry reports it.

    Returns None when the champion was scored on a different split. That is not
    a silent fallback to "no incumbent" — the gate's G1 would reject a
    cross-split comparison anyway, and forcing this run to be treated as a first
    promotion would hide the problem rather than surface it. The caller logs it.
    """
    try:
        current = registry.resolve(CHAMPION)
    except AliasNotSet:
        return None

    try:
        artifact = load(registry.artifacts / current)
    except Exception as err:  # the champion may predate a feature-code change
        log.warning("cannot read the current champion (%s); treating as no incumbent", err)
        return None

    if artifact.meta.split_hash != split_hash:
        log.warning(
            "champion %s was scored on split %s, this run uses %s — its recorded "
            "metrics are not comparable, so it is re-scored below",
            current,
            artifact.meta.split_hash[:12],
            split_hash[:12],
        )
        return None

    return Incumbent(
        name=artifact.meta.name,
        split_hash=artifact.meta.split_hash,
        macro_f1=float(artifact.meta.metrics.get("macro_f1", 0.0)),
        per_class_f1={
            k.removeprefix("f1_"): v
            for k, v in artifact.meta.metrics.items()
            if k.startswith("f1_")
        },
        usd_per_1000=float(artifact.meta.metrics.get("usd_per_1000", 0.0)),
        promoted_at=None,
    )


def _fit(train: Dataset, val: Dataset) -> list[Candidate]:
    """The candidate set. Order matters only for reporting."""
    baseline = MajorityClass()
    baseline.fit(train.texts, train.labels)

    model = TfidfLinearSVC(min_df=1)
    model.fit(
        train.texts,
        train.labels,
        val_texts=val.texts or None,
        val_labels=val.labels or None,
    )
    return [baseline, model]


def train_and_promote(
    *,
    registry_root: Path = DEFAULT_REGISTRY,
    use_random_split: bool = False,
    resamples: int = 10_000,
    seed: int = 42,
) -> TrainingRun:
    """Build a split, fit the candidates, run the gate, and act on the verdict."""
    articles, examples, labels = _load_corpus()
    taxonomy = sorted(set(labels.values()))

    def build(
        train_days: float, val_days: float
    ) -> tuple[SplitManifest, Dataset, Dataset, Dataset]:
        m = temporal_split(articles, train_cutoff_days=train_days, val_cutoff_days=val_days)
        assert_no_cluster_leak(m)
        return (
            m,
            build_dataset(m, "train", examples=examples, labels=labels),
            build_dataset(m, "val", examples=examples, labels=labels),
            build_dataset(m, "test", examples=examples, labels=labels),
        )

    def unusable(tr: Dataset, te: Dataset) -> bool:
        return len(tr) < 20 or tr.n_classes < 2 or len(te) < 20

    horizon = float(VAL_CUTOFF_DAYS)
    fell_back = False

    if use_random_split:
        manifest = random_split(articles, seed=seed)
        assert_no_cluster_leak(manifest)
        train = build_dataset(manifest, "train", examples=examples, labels=labels)
        val = build_dataset(manifest, "val", examples=examples, labels=labels)
        test = build_dataset(manifest, "test", examples=examples, labels=labels)
    else:
        manifest, train, val, test = build(TRAIN_CUTOFF_DAYS, VAL_CUTOFF_DAYS)

        # The protocol's window assumes weeks of collection history. When it does
        # not hold, fall back to the operationally-derived one rather than
        # producing a split with more test rows than training rows.
        if unusable(train, test) or len(test) > len(train):
            log.warning(
                "the %g/%g window gives train=%d test=%d; falling back to %g/%g, "
                "derived from the operational question rather than from any score",
                TRAIN_CUTOFF_DAYS,
                VAL_CUTOFF_DAYS,
                len(train),
                len(test),
                FALLBACK_TRAIN_CUTOFF_DAYS,
                FALLBACK_VAL_CUTOFF_DAYS,
            )
            manifest, train, val, test = build(FALLBACK_TRAIN_CUTOFF_DAYS, FALLBACK_VAL_CUTOFF_DAYS)
            horizon = FALLBACK_VAL_CUTOFF_DAYS
            fell_back = True

    run = TrainingRun(
        split_hash=manifest.hash,
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        is_provisional=manifest.is_provisional,
        split_warnings=list(manifest.warnings),
        horizon_days=horizon,
        used_fallback_window=fell_back,
    )

    if unusable(train, test):
        run.verdict = "UNUSABLE"
        run.reasons = [
            f"the split is not usable for training: train={len(train)} rows across "
            f"{train.n_classes} class(es), test={len(test)} rows"
        ]
        return run

    candidates = _fit(train, val)
    scored: list[tuple[Candidate, object, list[str]]] = []
    for candidate in candidates:
        result, predictions = score_candidate(candidate, test, taxonomy=taxonomy)
        scored.append((candidate, result, predictions))
        run.candidates.append(
            (
                candidate.name,
                result.report.macro_f1,
                result.ece,
                result.usd_per_1000,
            )
        )

    # The challenger is the best of this run's candidates by macro-F1; the
    # incumbent is whatever is actually serving.
    challenger, challenger_result, challenger_pred = max(
        scored,
        key=lambda s: s[1].report.macro_f1,  # type: ignore[attr-defined]
    )
    registry = Registry(registry_root)
    try:
        run.champion_before = registry.resolve(CHAMPION)
    except AliasNotSet:
        run.champion_before = None

    incumbent = _incumbent(registry, manifest.hash)
    if incumbent is None and run.champion_before is not None:
        # There is a champion, but its score is not comparable. Re-score it here
        # rather than pretending this is a first promotion.
        try:
            serving = load(registry.artifacts / run.champion_before)
            serving_result, serving_pred = score_candidate(serving.model, test, taxonomy=taxonomy)
            incumbent = Incumbent(
                name=serving.meta.name,
                split_hash=manifest.hash,
                macro_f1=serving_result.report.macro_f1,
                per_class_f1={s.label: s.f1 for s in serving_result.report.per_class},
                usd_per_1000=serving_result.usd_per_1000,
            )
            baseline_pred = serving_pred
        except Exception as err:
            log.warning("could not re-score the champion (%s)", err)
            baseline_pred = None
    else:
        baseline_pred = None

    # Fall back to this run's baseline when there is nothing serving, so the
    # comparison is always against something rather than skipped.
    if incumbent is None:
        base_candidate, base_result, base_pred = scored[0]
        if base_candidate is not challenger:
            incumbent = Incumbent(
                name=base_candidate.name,
                split_hash=manifest.hash,
                macro_f1=base_result.report.macro_f1,  # type: ignore[attr-defined]
                per_class_f1={
                    s.label: s.f1
                    for s in base_result.report.per_class  # type: ignore[attr-defined]
                },
                usd_per_1000=base_result.usd_per_1000,  # type: ignore[attr-defined]
            )
            baseline_pred = base_pred

    comparison = compare(
        test,
        incumbent.name if incumbent else "none",
        baseline_pred or challenger_pred,
        challenger.name,
        challenger_pred,
        taxonomy=taxonomy,
        resamples=resamples,
        seed=seed,
    )

    gate = evaluate_gate(
        challenger_result,  # type: ignore[arg-type]
        comparison,
        incumbent,
        split_hash=manifest.hash,
        n_test=len(test),
        cluster_leak=False,  # asserted above; a leak would have raised
    )
    run.verdict = str(gate.verdict)
    run.reasons = list(gate.reasons)

    staging = registry_root / "_staging" / challenger.name
    meta = save(
        challenger,
        staging,
        split_hash=manifest.hash,
        metrics={
            "macro_f1": challenger_result.report.macro_f1,  # type: ignore[attr-defined]
            "accuracy": challenger_result.report.accuracy,  # type: ignore[attr-defined]
            "ece": challenger_result.ece,  # type: ignore[attr-defined]
            "p95_ms": challenger_result.p95_ms,  # type: ignore[attr-defined]
            "usd_per_1000": challenger_result.usd_per_1000,  # type: ignore[attr-defined]
            **{
                f"f1_{s.label}": s.f1
                for s in challenger_result.report.per_class  # type: ignore[attr-defined]
            },
        },
        notes=f"trained {datetime.now(tz=UTC).isoformat()}",
    )

    run.decision = registry.apply(
        gate,
        artifact_dir=staging,
        artifact_id=meta.id,
        artifact_name=challenger.name,
        split_hash=manifest.hash,
        macro_f1=challenger_result.report.macro_f1,  # type: ignore[attr-defined]
    )

    try:
        run.champion_after = registry.resolve(CHAMPION)
    except AliasNotSet:
        run.champion_after = None
    return run
