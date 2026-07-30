"""TF-IDF + LinearSVC — the baseline that might just win.

The honest challenger to a transformer. If it lands within a few points of
IndoBERT at a thousandth of the cost, that *is* the finding, and it is a more
useful finding than the transformer winning would be.

Two decisions worth stating.

**Word and character n-grams together.** Indonesian is agglutinative — *menaikkan*,
*kenaikan* and *naik* share a stem that word tokenisation splits apart — so
character n-grams recover the morphological overlap that word features miss.
Headlines are also short, 10–25 tokens, which leaves word features sparse.

**Calibrated, because LinearSVC has no probabilities.** It emits signed distances
from the hyperplane. The cascade in Stage 4 gates on confidence, and a decision
function is not a confidence, so the raw scores are converted through a softmax
whose temperature is fitted on validation. Doing this at fit time rather than at
serving time keeps it inside the artifact, where train/serve skew cannot reach
it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from kanal.eval.metrics import softmax, temperature_scale
from kanal.models.base import USD_PER_VCPU_SECOND, Prediction

# Measured on the corpus at the time of writing. Re-measured by
# `measure_latency`, never assumed — this is only the fallback for pricing when
# no measurement is supplied.
ASSUMED_VCPU_SECONDS_PER_PREDICTION = 0.0015


class TfidfLinearSVC:
    """Word + character TF-IDF into a linear SVM, with fitted calibration."""

    name = "tfidf-linearsvc"

    def __init__(
        self,
        *,
        c: float = 1.0,
        min_df: int = 2,
        max_features: int | None = 50_000,
        seed: int = 42,
    ) -> None:
        self.c = c
        self.min_df = min_df
        self.max_features = max_features
        self.seed = seed
        self._pipeline: Pipeline | None = None
        self._classes: list[str] = []
        self._temperature = 1.0

    def _build(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "word",
                                TfidfVectorizer(
                                    analyzer="word",
                                    ngram_range=(1, 2),
                                    min_df=self.min_df,
                                    max_features=self.max_features,
                                    sublinear_tf=True,
                                    lowercase=True,
                                ),
                            ),
                            (
                                # Recovers the shared stem that word tokens split
                                # apart. `char_wb` keeps n-grams inside word
                                # boundaries, so it learns morphology rather than
                                # accidental cross-word sequences.
                                "char",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    min_df=self.min_df,
                                    max_features=self.max_features,
                                    sublinear_tf=True,
                                    lowercase=True,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "clf",
                    LinearSVC(
                        C=self.c,
                        random_state=self.seed,
                        # The corpus is imbalanced by a factor of six between the
                        # largest and smallest class, and macro-F1 weights them
                        # equally. Without this the small classes are simply
                        # abandoned.
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        val_texts: Sequence[str] | None = None,
        val_labels: Sequence[str] | None = None,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(f"length mismatch: {len(texts)} texts, {len(labels)} labels")
        if not texts:
            raise ValueError("cannot fit on an empty training set")
        if len(set(labels)) < 2:
            raise ValueError("cannot fit a classifier on a single class")

        self._pipeline = self._build()
        self._pipeline.fit(list(texts), list(labels))
        self._classes = list(self._pipeline.named_steps["clf"].classes_)

        # Calibrate on validation when it is offered, and on training otherwise —
        # saying so in `describe`, because calibrating on the fitting data
        # understates the temperature and the reader must be able to see that it
        # happened.
        if val_texts is not None and val_labels is not None and len(val_texts) > 0:
            self._temperature = self._fit_temperature(val_texts, val_labels)
            self._calibrated_on = "validation"
        else:
            self._temperature = self._fit_temperature(texts, labels)
            self._calibrated_on = "train (no validation supplied)"

    def _fit_temperature(self, texts: Sequence[str], labels: Sequence[str]) -> float:
        assert self._pipeline is not None
        scores = self._decision(list(texts))
        index = {label: i for i, label in enumerate(self._classes)}
        known = [(row, index[lab]) for row, lab in zip(scores, labels, strict=True) if lab in index]
        if len(known) < 2:
            return 1.0
        matrix = np.vstack([row for row, _ in known])
        targets = [i for _, i in known]
        return temperature_scale(matrix, targets)

    def _decision(self, texts: list[str]) -> np.ndarray:
        assert self._pipeline is not None
        raw: Any = self._pipeline.decision_function(texts)
        scores = np.asarray(raw, dtype=float)
        if scores.ndim == 1:
            # Binary LinearSVC returns one column. Mirror it so the downstream
            # softmax sees the same shape it does for the multi-class case.
            scores = np.column_stack([-scores, scores])
        return scores

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        if self._pipeline is None:
            raise RuntimeError("fit() must be called before predict()")
        if not texts:
            return []

        probabilities = softmax(self._decision(list(texts)), self._temperature)
        out: list[Prediction] = []
        for row in probabilities:
            best = int(np.argmax(row))
            out.append(
                Prediction(
                    label=self._classes[best],
                    confidence=float(row[best]),
                    probabilities={
                        label: float(p) for label, p in zip(self._classes, row, strict=True)
                    },
                )
            )
        return out

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "linear",
            "C": self.c,
            "min_df": self.min_df,
            "max_features": self.max_features,
            "seed": self.seed,
            "class_weight": "balanced",
            "features": "word 1-2gram + char_wb 3-5gram, sublinear tf",
            "classes": self._classes,
            "temperature": self._temperature,
            "calibrated_on": getattr(self, "_calibrated_on", "not fitted"),
        }

    def unit_cost_usd(
        self, n: int = 1000, *, measured_seconds_per_prediction: float | None = None
    ) -> float:
        """Amortised vCPU cost. Never zero.

        Pass the measured per-prediction time when one exists; the fallback is a
        stated assumption rather than a silent default.
        """
        per = (
            measured_seconds_per_prediction
            if measured_seconds_per_prediction is not None
            else ASSUMED_VCPU_SECONDS_PER_PREDICTION
        )
        return n * per * USD_PER_VCPU_SECOND
