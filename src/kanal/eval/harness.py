"""Run candidates over a frozen split and record everything a result needs.

A number without its provenance is not a result. Every run records the split
manifest hash, the candidate configuration, the measured latency and cost, the
sample sizes, and whether the protocol's provisional rule applies — so two
results can be compared, or shown to be incomparable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kanal.data.splits import SplitManifest
from kanal.eval.metrics import Report, evaluate, expected_calibration_error
from kanal.eval.significance import BootstrapResult, McNemarResult, mcnemar, paired_bootstrap
from kanal.features.text import Example, to_text
from kanal.models.base import Candidate, measure_latency

# From docs/evaluation.md, declared before any model ran.
MDE = 0.01
MAX_ECE = 0.08
ALPHA = 0.05


@dataclass
class Dataset:
    """A split's worth of rows, already reduced to what a model may see."""

    keys: list[str]
    texts: list[str]
    labels: list[str]
    clusters: list[str]

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def n_classes(self) -> int:
        return len(set(self.labels))


@dataclass
class CandidateResult:
    """One candidate's performance, with everything needed to reproduce it."""

    name: str
    report: Report
    ece: float
    overconfident_by: float
    p50_ms: float
    p95_ms: float
    usd_per_1000: float
    config: dict[str, object]

    def summary(self) -> str:
        return (
            f"{self.name:<18} macro-F1 {self.report.macro_f1:.4f}  "
            f"acc {self.report.accuracy:.4f}  ECE {self.ece:.4f}  "
            f"p95 {self.p95_ms:.1f}ms  ${self.usd_per_1000:.6f}/1k"
        )


@dataclass
class Comparison:
    """A challenger against an incumbent, with both required tests."""

    champion: str
    challenger: str
    bootstrap: BootstrapResult
    mcnemar: McNemarResult

    @property
    def passes_quality_gate(self) -> bool:
        """G3 from the promotion gate: both tests, not either."""
        return self.bootstrap.clears(MDE) and self.mcnemar.significant(ALPHA)

    def reasons(self) -> list[str]:
        """Why it failed, in the words the promotion log will use."""
        out: list[str] = []
        if not self.bootstrap.clears(MDE):
            out.append(
                f"CI lower bound {self.bootstrap.lower:+.4f} does not clear the "
                f"MDE of {MDE} declared in advance"
            )
        if not self.mcnemar.significant(ALPHA):
            out.append(f"McNemar p={self.mcnemar.p_value:.4g}, not below {ALPHA}")
        return out

    def summary(self) -> str:
        verdict = "PASS" if self.passes_quality_gate else "HOLD"
        line = f"{self.challenger} vs {self.champion}: {verdict}  {self.bootstrap.summary()}"
        for reason in self.reasons():
            line += f"\n    rejected: {reason}"
        return line


@dataclass
class RunResult:
    """Everything one evaluation run produced."""

    split_hash: str
    split_anchor: str
    n_train: int
    n_val: int
    n_test: int
    is_provisional: bool
    warnings: list[str]
    candidates: list[CandidateResult] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "split_hash": self.split_hash,
            "split_anchor": self.split_anchor,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "is_provisional": self.is_provisional,
            "warnings": self.warnings,
            "candidates": [
                {
                    "name": c.name,
                    "macro_f1": c.report.macro_f1,
                    "micro_f1": c.report.micro_f1,
                    "accuracy": c.report.accuracy,
                    "classes_evaluated": c.report.classes_evaluated,
                    "per_class": {
                        s.label: {
                            "f1": s.f1,
                            "precision": s.precision,
                            "recall": s.recall,
                            "support": s.support,
                        }
                        for s in c.report.per_class
                    },
                    "ece": c.ece,
                    "overconfident_by": c.overconfident_by,
                    "p50_ms": c.p50_ms,
                    "p95_ms": c.p95_ms,
                    "usd_per_1000": c.usd_per_1000,
                    "config": c.config,
                }
                for c in self.candidates
            ],
            "comparisons": [
                {
                    "champion": c.champion,
                    "challenger": c.challenger,
                    "delta": c.bootstrap.observed_delta,
                    "ci_lower": c.bootstrap.lower,
                    "ci_upper": c.bootstrap.upper,
                    "resampled_unit": c.bootstrap.resampled_unit,
                    "mcnemar_p": c.mcnemar.p_value,
                    "passes_quality_gate": c.passes_quality_gate,
                    "reasons": c.reasons(),
                }
                for c in self.comparisons
            ],
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def summary(self) -> str:
        lines = [
            f"split {self.split_hash[:12]}  "
            f"train={self.n_train} val={self.n_val} test={self.n_test}"
        ]
        if self.is_provisional:
            lines.append("  PROVISIONAL — test set below the size the protocol requires")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        for candidate in self.candidates:
            lines.append(f"  {candidate.summary()}")
            if candidate.ece > MAX_ECE:
                lines.append(
                    f"    ECE {candidate.ece:.4f} exceeds the {MAX_ECE} gate — "
                    f"a cascade gated on this confidence would escalate on noise"
                )
        for comparison in self.comparisons:
            lines.append(f"  {comparison.summary()}")
        return "\n".join(lines)


def build_dataset(
    manifest: SplitManifest,
    split: str,
    *,
    examples: dict[str, Example],
    labels: dict[str, str],
) -> Dataset:
    """Reduce a split to text and labels, going through `to_text` and nothing else.

    Every route from a row to a model input passes here, so the serving path
    cannot diverge from the training path without the skew test catching it.
    """
    keys = [k for k in manifest.keys(split) if k in examples]  # type: ignore[arg-type]
    return Dataset(
        keys=keys,
        texts=[to_text(examples[k]) for k in keys],
        labels=[labels[k] for k in keys],
        clusters=[manifest.cluster_of[k] for k in keys],
    )


def score_candidate(
    candidate: Candidate,
    test: Dataset,
    *,
    taxonomy: Sequence[str],
    latency_sample: int = 200,
) -> tuple[CandidateResult, list[str]]:
    """Score a fitted candidate, measuring latency and cost rather than assuming."""
    predictions = candidate.predict(test.texts)
    predicted = [p.label for p in predictions]

    report = evaluate(test.labels, predicted, labels=taxonomy)
    calibration = expected_calibration_error(
        [p.confidence for p in predictions],
        [p.label == truth for p, truth in zip(predictions, test.labels, strict=True)],
    )

    sample = [Example(title=t) for t in test.texts[:latency_sample]]
    timing = measure_latency(candidate, sample)
    seconds = timing.mean_ms / 1000.0

    try:
        cost = candidate.unit_cost_usd(1000, measured_seconds_per_prediction=seconds)  # type: ignore[call-arg]
    except TypeError:
        cost = candidate.unit_cost_usd(1000)

    return (
        CandidateResult(
            name=candidate.name,
            report=report,
            ece=calibration.ece,
            overconfident_by=calibration.overconfident_by,
            p50_ms=timing.p50_ms,
            p95_ms=timing.p95_ms,
            usd_per_1000=cost,
            config=candidate.describe(),
        ),
        predicted,
    )


def compare(
    test: Dataset,
    champion_name: str,
    champion_pred: Sequence[str],
    challenger_name: str,
    challenger_pred: Sequence[str],
    *,
    taxonomy: Sequence[str],
    resamples: int = 10_000,
    seed: int = 42,
) -> Comparison:
    """Both required tests, on cluster-resampled data."""
    return Comparison(
        champion=champion_name,
        challenger=challenger_name,
        bootstrap=paired_bootstrap(
            test.labels,
            champion_pred,
            challenger_pred,
            clusters=test.clusters,
            labels=taxonomy,
            resamples=resamples,
            seed=seed,
        ),
        mcnemar=mcnemar(test.labels, champion_pred, challenger_pred),
    )
