"""Metrics and calibration, checked against hand-computed values.

Every expected number in this file was worked out on paper from the definition,
not read off another library. Testing one implementation against another only
proves they share conventions — and the conventions are exactly what changes the
answer here.
"""

from __future__ import annotations

import numpy as np
import pytest

from kanal.eval.metrics import (
    evaluate,
    expected_calibration_error,
    softmax,
    temperature_scale,
)

TAXONOMY = [
    "ekonomi",
    "gaya-hidup-kesehatan",
    "hiburan",
    "hukum-kriminal",
    "internasional",
    "olahraga",
    "politik",
    "teknologi",
]


class TestBasicScores:
    def test_perfect_predictions(self) -> None:
        y = ["politik", "ekonomi", "olahraga"]
        r = evaluate(y, y)
        assert r.macro_f1 == 1.0
        assert r.accuracy == 1.0
        assert all(s.f1 == 1.0 for s in r.per_class)

    def test_everything_wrong(self) -> None:
        r = evaluate(["politik", "ekonomi"], ["ekonomi", "politik"])
        assert r.macro_f1 == 0.0
        assert r.accuracy == 0.0

    def test_hand_computed_two_class_case(self) -> None:
        # politik: TP=2 FP=1 FN=1  ->  P=2/3, R=2/3, F1=2/3
        # ekonomi: TP=1 FP=1 FN=1  ->  P=1/2, R=1/2, F1=1/2
        # macro = (2/3 + 1/2) / 2 = 7/12 = 0.58333...
        y_true = ["politik", "politik", "politik", "ekonomi", "ekonomi"]
        y_pred = ["politik", "politik", "ekonomi", "ekonomi", "politik"]

        r = evaluate(y_true, y_pred)
        assert r.f1_of("politik") == pytest.approx(2 / 3)
        assert r.f1_of("ekonomi") == pytest.approx(1 / 2)
        assert r.macro_f1 == pytest.approx(7 / 12)
        assert r.accuracy == pytest.approx(3 / 5)

    def test_micro_f1_equals_accuracy_for_single_label(self) -> None:
        # An identity worth pinning: with exactly one prediction per row, summed
        # false positives equal summed false negatives, so micro-F1 collapses to
        # accuracy. Anyone reporting both as if they were independent evidence
        # is reporting one number twice.
        y_true = ["a", "b", "c", "a", "b"]
        y_pred = ["a", "c", "c", "b", "b"]
        r = evaluate(y_true, y_pred)
        assert r.micro_f1 == pytest.approx(r.accuracy)

    def test_support_counts_true_instances(self) -> None:
        r = evaluate(["a", "a", "a", "b"], ["a", "b", "b", "b"])
        support = {s.label: s.support for s in r.per_class}
        assert support == {"a": 3, "b": 1}


class TestConventions:
    """The three decisions that change the answer, pinned so they stay decisions."""

    def test_absent_classes_are_excluded_not_scored_zero(self) -> None:
        # Only two of eight classes appear. Scoring the other six as 0 would give
        # macro-F1 = 2/8 = 0.25 for a model that got everything right.
        y = ["politik", "ekonomi"]
        r = evaluate(y, y, labels=TAXONOMY)

        assert r.macro_f1 == 1.0
        assert r.classes_evaluated == 2
        assert len(r.classes_absent) == 6
        assert "olahraga" in r.classes_absent

    def test_the_denominator_is_always_reported(self) -> None:
        # So a shrinking test set cannot quietly inflate macro-F1.
        r = evaluate(["politik"], ["politik"], labels=TAXONOMY)
        assert "over 1 class" in r.summary()
        assert "absent, therefore excluded" in r.summary()

    def test_a_class_never_predicted_scores_zero(self) -> None:
        # The model failed to find a class that was there. That is precisely what
        # macro-F1 exists to expose, so it must not be excused as undefined.
        y_true = ["politik", "politik", "hukum-kriminal"]
        y_pred = ["politik", "politik", "politik"]
        r = evaluate(y_true, y_pred)
        assert r.f1_of("hukum-kriminal") == 0.0

    def test_macro_f1_falls_when_a_small_class_collapses(self) -> None:
        # The reason macro-F1 is the primary metric. Accuracy barely moves;
        # macro-F1 halves.
        y_true = ["politik"] * 97 + ["hukum-kriminal"] * 3
        good = ["politik"] * 97 + ["hukum-kriminal"] * 3
        collapsed = ["politik"] * 100

        assert evaluate(y_true, good).macro_f1 == 1.0
        assert evaluate(y_true, collapsed).accuracy == pytest.approx(0.97)
        assert evaluate(y_true, collapsed).macro_f1 < 0.5


class TestValidation:
    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            evaluate(["a", "b"], ["a"])

    def test_empty_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty test set"):
            evaluate([], [])

    def test_confusion_matrix_rows_sum_to_support(self) -> None:
        y_true = ["a", "a", "b", "b", "b"]
        y_pred = ["a", "b", "b", "b", "a"]
        r = evaluate(y_true, y_pred)
        assert sum(r.confusion["a"].values()) == 2
        assert sum(r.confusion["b"].values()) == 3
        assert r.confusion["a"]["b"] == 1


class TestCalibration:
    def test_a_perfectly_calibrated_model_scores_near_zero(self) -> None:
        # Claims 70% confidence, is right 70% of the time.
        conf = [0.7] * 100
        correct = [True] * 70 + [False] * 30
        cal = expected_calibration_error(conf, correct, bins=1)
        assert cal.ece == pytest.approx(0.0, abs=1e-9)

    def test_maximum_overconfidence_scores_exactly_the_gap(self) -> None:
        # Claims certainty, is right half the time. |0.5 - 1.0| = 0.5.
        conf = [1.0] * 100
        correct = [True] * 50 + [False] * 50
        cal = expected_calibration_error(conf, correct, bins=10)
        assert cal.ece == pytest.approx(0.5)

    def test_reports_the_direction_not_only_the_magnitude(self) -> None:
        # ECE discards the sign, and the sign is what decides whether a cascade
        # escalates too often or too rarely.
        over = expected_calibration_error([0.95] * 100, [True] * 50 + [False] * 50)
        under = expected_calibration_error([0.55] * 100, [True] * 90 + [False] * 10)

        assert over.overconfident_by > 0
        assert under.overconfident_by < 0
        assert "overconfident" in over.summary()
        assert "underconfident" in under.summary()

    def test_equal_mass_bins_hold_roughly_equal_counts(self) -> None:
        rng = np.random.default_rng(0)
        conf = list(rng.uniform(0.5, 1.0, 300))
        correct = list(rng.random(300) < 0.8)
        cal = expected_calibration_error(conf, correct, bins=15)

        counts = [b.count for b in cal.bins]
        assert len(cal.bins) <= 16
        assert max(counts) - min(counts) <= 5

    def test_identical_confidences_are_never_split_across_bins(self) -> None:
        # Otherwise two predictions the model treated identically get scored in
        # different bins, which is an artefact of sorting rather than a property
        # of the model.
        conf = [0.9] * 50 + [0.95] * 50
        correct = [True] * 45 + [False] * 5 + [True] * 48 + [False] * 2
        cal = expected_calibration_error(conf, correct, bins=10)

        for b in cal.bins:
            assert b.lower == b.upper

    def test_rejects_confidences_outside_the_unit_interval(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            expected_calibration_error([1.5], [True])

    def test_rejects_empty_and_mismatched_input(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            expected_calibration_error([], [])
        with pytest.raises(ValueError, match="length mismatch"):
            expected_calibration_error([0.5, 0.5], [True])


class TestTemperatureScaling:
    def test_cannot_change_which_class_is_predicted(self) -> None:
        # The property that makes it safe: one scalar divides every logit, so the
        # argmax is invariant. It fixes calibration without touching accuracy.
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(50, 8)) * 3

        before = softmax(logits).argmax(axis=1)
        for t in (0.5, 1.0, 2.0, 5.0):
            assert np.array_equal(softmax(logits, t).argmax(axis=1), before)

    def test_finds_a_temperature_above_one_for_an_overconfident_model(self) -> None:
        # Sharp logits that are often wrong. The fix is to soften them, T > 1.
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(400, 4)) * 8
        truth = rng.integers(0, 4, size=400)  # labels unrelated to the logits

        assert temperature_scale(logits, list(truth)) > 1.0

    def test_finds_a_temperature_below_one_for_an_underconfident_model(self) -> None:
        rng = np.random.default_rng(3)
        logits = rng.normal(size=(400, 4)) * 0.2
        truth = logits.argmax(axis=1)  # always right, but barely committed

        assert temperature_scale(logits, list(truth)) < 1.0

    def test_scaling_actually_reduces_ece(self) -> None:
        # The end-to-end claim, rather than a property of the optimiser.
        rng = np.random.default_rng(4)
        n = 600
        truth = rng.integers(0, 4, size=n)
        logits = rng.normal(size=(n, 4))
        # Push the true class up a little, then exaggerate — confidently right
        # more often than not, but far too confident about it.
        logits[np.arange(n), truth] += 1.0
        logits *= 6

        probs = softmax(logits)
        before = expected_calibration_error(
            list(probs.max(axis=1)), list(probs.argmax(axis=1) == truth)
        )

        t = temperature_scale(logits, list(truth))
        scaled = softmax(logits, t)
        after = expected_calibration_error(
            list(scaled.max(axis=1)), list(scaled.argmax(axis=1) == truth)
        )

        assert after.ece < before.ece
        assert after.ece < 0.08  # the protocol's gate

    def test_rejects_malformed_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            temperature_scale(np.zeros(5), [0])
        with pytest.raises(ValueError, match="length mismatch"):
            temperature_scale(np.zeros((5, 3)), [0, 1])
