"""The baseline that makes every other number mean something.

Always predicts the most frequent training class. It is in the candidate set for
one reason: without it, "0.84 macro-F1" is unanchored. With it, the reader can
see what the task's floor actually is — and on a corpus this imbalanced the floor
is informative, because a majority baseline reaches respectable *accuracy* while
its macro-F1 collapses to roughly 1/k.

That gap is the argument for macro-F1 as the primary metric, demonstrated rather
than asserted.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from kanal.models.base import USD_PER_VCPU_SECOND, Prediction


class MajorityClass:
    """Predicts the training set's most common label, every time."""

    name = "majority"

    def __init__(self) -> None:
        self._label: str | None = None
        self._classes: list[str] = []
        self._prior: dict[str, float] = {}

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        if len(texts) != len(labels):
            raise ValueError(f"length mismatch: {len(texts)} texts, {len(labels)} labels")
        if not labels:
            raise ValueError("cannot fit on an empty training set")

        counts = Counter(labels)
        # Ties broken by label order, so a rerun on shuffled data gives the same
        # baseline. A baseline that moved between runs would make every delta
        # measured against it untrustworthy.
        self._label = min(
            (lab for lab, n in counts.items() if n == max(counts.values())),
        )
        self._classes = sorted(counts)
        total = sum(counts.values())
        self._prior = {lab: counts[lab] / total for lab in self._classes}

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        if self._label is None:
            raise RuntimeError("fit() must be called before predict()")

        # The confidence is the training prior, not 1.0. Claiming certainty would
        # make the baseline look perfectly calibrated at the top of the range
        # while being wrong most of the time — and ECE would report it as such,
        # which is misleading rather than merely poor.
        prediction = Prediction(
            label=self._label,
            confidence=self._prior[self._label],
            probabilities=dict(self._prior),
        )
        return [prediction] * len(texts)

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "constant",
            "predicted_label": self._label,
            "classes": self._classes,
            "training_prior": self._prior,
        }

    def unit_cost_usd(self, n: int = 1000) -> float:
        # A dictionary lookup, but not zero: it still occupies a process. Priced
        # at roughly a microsecond of vCPU per prediction so the comparison stays
        # on one scale rather than dividing by zero at the cheap end.
        return n * 1e-6 * USD_PER_VCPU_SECOND
