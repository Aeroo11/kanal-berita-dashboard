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

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
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
from kanal.eval.harness import Dataset, RunResult, build_dataset, compare, score_candidate
from kanal.features.text import Example
from kanal.models.base import Candidate
from kanal.models.majority import MajorityClass
from kanal.models.tfidf import TfidfLinearSVC
from kanal.registry.artifact import load, save
from kanal.registry.promote import GateResult, Incumbent, Verdict, evaluate_gate
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
    horizon_days: float | None = 7.0
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
        horizon = "random split" if self.horizon_days is None else f"horizon={self.horizon_days:g}d"
        lines = [
            f"split {self.split_hash[:12]}  "
            f"train={self.n_train} val={self.n_val} test={self.n_test}  "
            f"{horizon}",
        ]
        if self.used_fallback_window and self.horizon_days is not None:
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


def _load_corpus(
    dataset: Path | None = None,
) -> tuple[list[Article], dict[str, Example], dict[str, str]]:
    import duckdb

    from kanal.warehouse.duck import reader
    from kanal.warehouse.loader import default_db_path

    query = """
        SELECT article_key, title, coalesce(summary, ''), published_at, source, kanal
        FROM fct_articles
        WHERE published_at IS NOT NULL AND title IS NOT NULL AND title <> ''
          AND source IS NOT NULL AND kanal IS NOT NULL
        ORDER BY published_at, article_key
    """
    if dataset is None:
        with reader(default_db_path()) as connection:
            rows = connection.execute(query).fetchall()
    else:
        with duckdb.connect() as connection:
            connection.read_parquet(str(dataset)).create_view("fct_articles")
            rows = connection.execute(query).fetchall()

    if not rows:
        if dataset is not None:
            raise RuntimeError(f"the dataset has no usable articles: {dataset}")
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


def _fit(train: Dataset, val: Dataset, seed: int) -> tuple[Candidate, Candidate]:
    """The candidate set. Order matters only for reporting."""
    baseline = MajorityClass()
    baseline.fit(train.texts, train.labels)

    model = TfidfLinearSVC(min_df=1, seed=seed)
    model.fit(
        train.texts,
        train.labels,
        val_texts=val.texts or None,
        val_labels=val.labels or None,
    )
    return baseline, model


def train_and_promote(
    *,
    registry_root: Path = DEFAULT_REGISTRY,
    dataset: Path | None = None,
    report_dir: Path | None = None,
    git_commit: str | None = None,
    use_random_split: bool = False,
    resamples: int = 10_000,
    seed: int = 42,
) -> TrainingRun:
    """Build a split, fit the candidates, run the gate, and act on the verdict."""
    dataset_sha256 = None
    if report_dir is not None and dataset is not None:
        with dataset.open("rb") as source:
            dataset_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    articles, examples, labels = _load_corpus(dataset)
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

    horizon = None if use_random_split else float(VAL_CUTOFF_DAYS)
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

    evaluation = RunResult(
        split_hash=manifest.hash,
        split_anchor=manifest.anchor,
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        is_provisional=manifest.is_provisional,
        warnings=manifest.warnings,
    )

    def finish() -> TrainingRun:
        if report_dir is not None:
            manifest.write(report_dir / "split.json")
            payload = evaluation.to_dict() | {
                "dataset_sha256": dataset_sha256,
                "git_commit": git_commit,
                "seed": seed,
                "resamples": resamples,
                "horizon_days": run.horizon_days,
                "used_fallback_window": run.used_fallback_window,
                "verdict": run.verdict,
                "reasons": run.reasons,
                "decision": asdict(run.decision) if run.decision else None,
                "champion_after": run.champion_after,
            }
            (report_dir / "evaluation.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return run

    if unusable(train, test):
        run.verdict = "UNUSABLE"
        run.reasons = [
            f"the split is not usable for training: train={len(train)} rows across "
            f"{train.n_classes} class(es), test={len(test)} rows"
        ]
        return finish()

    baseline, challenger = _fit(train, val, seed)
    baseline_result, baseline_pred = score_candidate(baseline, test, taxonomy=taxonomy)
    challenger_result, challenger_pred = score_candidate(challenger, test, taxonomy=taxonomy)
    evaluation.candidates.extend((baseline_result, challenger_result))
    run.candidates.extend(
        (result.name, result.report.macro_f1, result.ece, result.usd_per_1000)
        for result in evaluation.candidates
    )
    comparison = compare(
        test,
        baseline.name,
        baseline_pred,
        challenger.name,
        challenger_pred,
        taxonomy=taxonomy,
        resamples=resamples,
        seed=seed,
    )
    evaluation.comparisons.append(comparison)

    registry = Registry(registry_root)
    try:
        run.champion_before = registry.resolve(CHAMPION)
    except AliasNotSet:
        run.champion_before = None
    run.champion_after = run.champion_before

    if use_random_split:
        run.verdict = str(Verdict.HOLD)
        run.reasons = ["Random splits are for diagnostics. Promotion requires a temporal split."]
        return finish()

    reference_result = baseline_result
    promoted_at = None
    gate = None
    if run.champion_before is not None:
        try:
            serving = load(registry.artifacts / run.champion_before)
            reference_result, serving_pred = score_candidate(serving.model, test, taxonomy=taxonomy)
            set_at = registry.alias_set_at(CHAMPION)
            promoted_at = datetime.fromisoformat(set_at) if set_at else None
        except Exception as err:
            gate = GateResult(
                verdict=Verdict.HOLD,
                reasons=[
                    f"G1 provenance: could not evaluate champion {run.champion_before}: {err}"
                ],
            )
        else:
            comparison = compare(
                test,
                reference_result.name,
                serving_pred,
                challenger.name,
                challenger_pred,
                taxonomy=taxonomy,
                resamples=resamples,
                seed=seed,
            )
            evaluation.comparisons.append(comparison)

    if gate is None:
        incumbent = Incumbent(
            name=reference_result.name,
            split_hash=manifest.hash,
            macro_f1=reference_result.report.macro_f1,
            per_class_f1={s.label: s.f1 for s in reference_result.report.per_class},
            usd_per_1000=reference_result.usd_per_1000,
            promoted_at=promoted_at,
        )
        gate = evaluate_gate(
            challenger_result, comparison, incumbent, split_hash=manifest.hash, n_test=len(test)
        )
    run.verdict = str(gate.verdict)
    run.reasons = list(gate.reasons)

    staging = registry_root / "_staging" / challenger.name
    meta = save(
        challenger,
        staging,
        split_hash=manifest.hash,
        metrics={
            "macro_f1": challenger_result.report.macro_f1,
            "accuracy": challenger_result.report.accuracy,
            "ece": challenger_result.ece,
            "p95_ms": challenger_result.p95_ms,
            "usd_per_1000": challenger_result.usd_per_1000,
            **{f"f1_{s.label}": s.f1 for s in challenger_result.report.per_class},
        },
        git_commit=git_commit,
    )
    run.decision = registry.apply(
        gate,
        artifact_dir=staging,
        artifact_id=meta.id,
        artifact_name=challenger.name,
        split_hash=manifest.hash,
        macro_f1=challenger_result.report.macro_f1,
    )
    if run.promoted:
        run.champion_after = meta.id
    return finish()
