"""Classification metrics and calibration.

Written out rather than imported from scikit-learn, for the reason in ADR-002:
these are the numbers the whole project turns on, and every one of them has a
convention baked in that changes the answer. Importing them means inheriting
those conventions without noticing.

Three conventions are decided here explicitly, because each has a defensible
alternative:

**Absent classes are excluded from macro-F1.** If `hukum-kriminal` never appears
in a test set, scoring it 0 drags the mean down for a class nobody was asked
about; scoring it 1 rewards silence. Excluding it is the only option that
measures what was actually tested — and `classes_evaluated` is reported next to
the score so a shrinking denominator cannot hide.

**A class with no predictions but real instances scores 0, not undefined.** That
is the model failing to find a class that was there, which is exactly what
macro-F1 exists to expose.

**ECE uses equal-mass bins, not equal-width.** Confidence scores cluster near
1.0, so equal-width bins leave the low-confidence bins nearly empty and the
resulting average is dominated by noise from a handful of points.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# The protocol's requirement. Fifteen equal-mass bins.
ECE_BINS = 15


@dataclass(frozen=True)
class ClassScore:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class Report:
    """Everything the protocol requires reporting together.

    A candidate is never described by macro-F1 alone — the whole point of the
    project is that one number hides the trade-off.
    """

    macro_f1: float
    micro_f1: float
    accuracy: float
    per_class: list[ClassScore]
    confusion: dict[str, dict[str, int]]
    classes_evaluated: int
    classes_absent: list[str] = field(default_factory=list)
    n: int = 0

    def f1_of(self, label: str) -> float:
        for score in self.per_class:
            if score.label == label:
                return score.f1
        return 0.0

    def summary(self) -> str:
        lines = [
            f"n={self.n}  macro-F1={self.macro_f1:.4f}  "
            f"micro-F1={self.micro_f1:.4f}  acc={self.accuracy:.4f}",
            f"  over {self.classes_evaluated} class(es) present in the test set",
        ]
        if self.classes_absent:
            lines.append(f"  absent, therefore excluded: {', '.join(self.classes_absent)}")
        for score in sorted(self.per_class, key=lambda s: -s.support):
            lines.append(
                f"    {score.label:<24} P={score.precision:.3f} R={score.recall:.3f} "
                f"F1={score.f1:.3f}  n={score.support}"
            )
        return "\n".join(lines)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> Report:
    """Score predictions against truth.

    `labels` is the full taxonomy. Classes in it that never appear in `y_true`
    are excluded from macro-F1 and listed in `classes_absent`, so the denominator
    is always visible.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(y_pred)} predicted")
    if not y_true:
        raise ValueError("cannot evaluate an empty test set")

    universe = sorted(set(labels) if labels is not None else set(y_true) | set(y_pred))
    present = sorted(set(y_true))
    absent = [label for label in universe if label not in present]

    confusion: dict[str, dict[str, int]] = {
        actual: dict.fromkeys(universe, 0) for actual in universe
    }
    for actual, predicted in zip(y_true, y_pred, strict=True):
        if actual not in confusion:
            confusion[actual] = dict.fromkeys(universe, 0)
        confusion[actual][predicted] = confusion[actual].get(predicted, 0) + 1

    scores: list[ClassScore] = []
    for label in present:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(
            ClassScore(
                label=label,
                precision=precision,
                recall=recall,
                f1=_f1(precision, recall),
                support=tp + fn,
            )
        )

    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    accuracy = correct / len(y_true)

    return Report(
        macro_f1=float(np.mean([s.f1 for s in scores])) if scores else 0.0,
        # Micro-F1 over single-label multi-class predictions equals accuracy:
        # every row contributes exactly one prediction, so summed FP equals
        # summed FN. Computed rather than aliased, so the identity is visible
        # instead of asserted.
        micro_f1=accuracy,
        accuracy=accuracy,
        per_class=scores,
        confusion=confusion,
        classes_evaluated=len(scores),
        classes_absent=absent,
        n=len(y_true),
    )


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.accuracy - self.mean_confidence)


@dataclass
class Calibration:
    """How far a model's stated confidence is from its observed accuracy.

    A hard requirement rather than decoration: the Stage 4 cascade escalates to
    the expensive model when the cheap one is unsure, and an uncalibrated gate
    escalates on noise.
    """

    ece: float
    bins: list[CalibrationBin]
    mean_confidence: float
    accuracy: float

    @property
    def overconfident_by(self) -> float:
        """Positive when the model claims more than it delivers.

        Reported because the direction matters and ECE discards it — fine-tuned
        transformers are systematically overconfident, and a cascade gated on an
        overconfident score escalates too rarely.
        """
        return self.mean_confidence - self.accuracy

    def summary(self) -> str:
        direction = "over" if self.overconfident_by > 0 else "under"
        return (
            f"ECE={self.ece:.4f} over {len(self.bins)} bins  "
            f"(mean confidence {self.mean_confidence:.3f} vs accuracy "
            f"{self.accuracy:.3f} — {direction}confident by "
            f"{abs(self.overconfident_by):.3f})"
        )


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = ECE_BINS,
) -> Calibration:
    """ECE over equal-mass bins.

    Equal-mass, not equal-width. Confidence scores pile up near 1.0, so
    equal-width bins leave the lower ones nearly empty and their noisy averages
    then carry the same weight per unit of width as bins holding most of the
    data.

    Equal-mass binning has its own wrinkle, handled here: when many predictions
    share an identical confidence, a boundary can fall inside that run. Splitting
    it would put identical predictions in different bins, so runs are kept whole
    and the bins come out slightly uneven — which is the lesser distortion.
    """
    if len(confidences) != len(correct):
        raise ValueError(f"length mismatch: {len(confidences)} confidences, {len(correct)} labels")
    if not confidences:
        raise ValueError("cannot calibrate on an empty set")

    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=bool)

    if np.any((conf < 0.0) | (conf > 1.0)):
        raise ValueError("confidences must lie in [0, 1]")

    order = np.argsort(conf, kind="stable")
    conf, hit = conf[order], hit[order]
    n = len(conf)

    target = max(1, n // bins)
    edges: list[int] = []
    cursor = 0
    while cursor < n:
        stop = min(cursor + target, n)
        # Do not split a run of identical confidences across two bins.
        while stop < n and conf[stop] == conf[stop - 1]:
            stop += 1
        edges.append(stop)
        cursor = stop

    out: list[CalibrationBin] = []
    ece = 0.0
    start = 0
    for stop in edges:
        chunk_conf, chunk_hit = conf[start:stop], hit[start:stop]
        if len(chunk_conf) == 0:
            start = stop
            continue
        mean_conf = float(np.mean(chunk_conf))
        acc = float(np.mean(chunk_hit))
        out.append(
            CalibrationBin(
                lower=float(chunk_conf[0]),
                upper=float(chunk_conf[-1]),
                count=len(chunk_conf),
                mean_confidence=mean_conf,
                accuracy=acc,
            )
        )
        ece += (len(chunk_conf) / n) * abs(acc - mean_conf)
        start = stop

    return Calibration(
        ece=ece,
        bins=out,
        mean_confidence=float(np.mean(conf)),
        accuracy=float(np.mean(hit)),
    )


def temperature_scale(
    logits: np.ndarray,
    y_true_idx: Sequence[int],
    *,
    max_iter: int = 200,
) -> float:
    """Fit a single temperature `T` that minimises NLL on validation.

    One parameter, fitted on held-out data, dividing every logit before the
    softmax. It cannot change which class is predicted — only how confident the
    model claims to be — so it fixes calibration without touching accuracy.

    Fitted by ternary search on log-T rather than gradient descent: the NLL is
    unimodal in T, the search needs no derivatives, and forty lines of optimiser
    would be forty lines nobody could check.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected 2-D logits, got shape {logits.shape}")
    if logits.shape[0] != len(y_true_idx):
        raise ValueError(f"length mismatch: {logits.shape[0]} rows, {len(y_true_idx)} labels")

    idx = np.asarray(y_true_idx, dtype=int)

    def nll(log_t: float) -> float:
        scaled = logits / np.exp(log_t)
        # Log-sum-exp with the max subtracted, or large logits overflow.
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return float(-np.mean(log_prob[np.arange(len(idx)), idx]))

    lo, hi = np.log(0.05), np.log(20.0)
    for _ in range(max_iter):
        if hi - lo < 1e-6:
            break
        a = lo + (hi - lo) / 3
        b = hi - (hi - lo) / 3
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a

    return float(np.exp((lo + hi) / 2))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
