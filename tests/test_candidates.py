"""The candidates, and the protocol that keeps the comparison honest.

Four candidates spanning four orders of magnitude in cost only compare fairly if
the harness treats them identically — so the tests here are mostly about the
*protocol* holding, not about any candidate being good.
"""

from __future__ import annotations

import pytest

from kanal.features.text import Example
from kanal.models.base import Candidate, Prediction, measure_latency
from kanal.models.majority import MajorityClass
from kanal.models.tfidf import TfidfLinearSVC

TRAIN_TEXTS = [
    "Presiden resmikan bendungan baru di Jawa Barat",
    "Pemerintah umumkan kebijakan subsidi energi",
    "DPR sahkan undang-undang baru soal pemilu",
    "Menteri keuangan bicara soal anggaran negara",
    "Harga emas menguat di tengah ketidakpastian pasar",
    "Bank sentral pertahankan suku bunga acuan bulan ini",
    "Rupiah melemah terhadap dolar Amerika Serikat",
    "Inflasi tahunan tercatat naik tipis bulan lalu",
    "Timnas futsal menang atas Thailand di partai final",
    "Persib segel tiket lolos ke babak semifinal",
    "Pemain muda cetak gol kemenangan di menit akhir",
    "Kompetisi liga utama dimulai akhir pekan ini",
]
TRAIN_LABELS = ["politik"] * 4 + ["ekonomi"] * 4 + ["olahraga"] * 4


def fitted_tfidf() -> TfidfLinearSVC:
    model = TfidfLinearSVC(min_df=1)
    model.fit(TRAIN_TEXTS, TRAIN_LABELS)
    return model


def fitted_majority() -> MajorityClass:
    model = MajorityClass()
    model.fit(TRAIN_TEXTS, TRAIN_LABELS)
    return model


class TestProtocolConformance:
    @pytest.mark.parametrize("build", [fitted_majority, fitted_tfidf])
    def test_every_candidate_satisfies_the_protocol(self, build: object) -> None:
        model = build()  # type: ignore[operator]
        assert isinstance(model, Candidate)

    @pytest.mark.parametrize("build", [fitted_majority, fitted_tfidf])
    def test_predictions_are_well_formed(self, build: object) -> None:
        model = build()  # type: ignore[operator]
        preds = model.predict(["Harga emas naik tajam hari ini"])

        assert len(preds) == 1
        p = preds[0]
        assert isinstance(p, Prediction)
        assert 0.0 <= p.confidence <= 1.0
        assert p.label in p.probabilities
        assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
        # The stated confidence must *be* the probability of the stated label.
        assert p.confidence == pytest.approx(p.probabilities[p.label])

    @pytest.mark.parametrize("build", [fitted_majority, fitted_tfidf])
    def test_describe_is_enough_to_reproduce(self, build: object) -> None:
        described = build().describe()  # type: ignore[operator]
        assert "name" in described
        assert "classes" in described

    @pytest.mark.parametrize("build", [fitted_majority, fitted_tfidf])
    def test_no_candidate_prices_itself_at_zero(self, build: object) -> None:
        """The most common falsehood in cost comparisons.

        "Self-hosted is free" is wrong: inference occupies vCPU-seconds, and
        pricing them at zero makes a four-orders-of-magnitude comparison
        meaningless at the cheap end.
        """
        assert build().unit_cost_usd(1000) > 0.0  # type: ignore[operator]

    @pytest.mark.parametrize("build", [fitted_majority, fitted_tfidf])
    def test_cost_scales_with_volume(self, build: object) -> None:
        model = build()  # type: ignore[operator]
        assert model.unit_cost_usd(2000) == pytest.approx(2 * model.unit_cost_usd(1000))


class TestMajorityBaseline:
    def test_predicts_the_most_frequent_class(self) -> None:
        model = MajorityClass()
        model.fit(["a", "b", "c", "d"], ["x", "x", "x", "y"])
        assert model.predict(["anything"])[0].label == "x"

    def test_ties_break_deterministically(self) -> None:
        # A baseline that moved between runs would make every delta measured
        # against it untrustworthy.
        first = MajorityClass()
        first.fit(["a", "b"], ["x", "y"])
        second = MajorityClass()
        second.fit(["b", "a"], ["y", "x"])
        assert first.predict(["q"])[0].label == second.predict(["q"])[0].label

    def test_confidence_is_the_prior_not_certainty(self) -> None:
        """Claiming 1.0 would be worse than useless.

        A constant predictor that claims certainty and is right 40% of the time
        is not merely poorly calibrated — it would report a maximally confident
        wrong answer to a cascade gate. The training prior is the honest number.
        """
        model = MajorityClass()
        model.fit(["a"] * 4 + ["b"] * 6, ["x"] * 4 + ["y"] * 6)
        p = model.predict(["q"])[0]
        assert p.label == "y"
        assert p.confidence == pytest.approx(0.6)

    def test_macro_f1_collapses_while_accuracy_looks_respectable(self) -> None:
        """The demonstration that justifies macro-F1 as the primary metric."""
        from kanal.eval.metrics import evaluate

        truth = ["politik"] * 70 + ["ekonomi"] * 20 + ["olahraga"] * 10
        model = MajorityClass()
        model.fit(["t"] * 100, truth)
        predicted = [p.label for p in model.predict(["x"] * 100)]

        report = evaluate(truth, predicted)
        assert report.accuracy == pytest.approx(0.70)
        assert report.macro_f1 < 0.30

    def test_predict_before_fit_is_an_error(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            MajorityClass().predict(["x"])

    def test_rejects_malformed_training_data(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            MajorityClass().fit(["a"], ["x", "y"])
        with pytest.raises(ValueError, match="empty"):
            MajorityClass().fit([], [])


class TestTfidfLinearSVC:
    def test_learns_something_beyond_the_majority(self) -> None:
        model = fitted_tfidf()
        assert model.predict(["Rupiah melemah terhadap dolar"])[0].label == "ekonomi"
        assert model.predict(["Timnas menang di final"])[0].label == "olahraga"

    def test_character_ngrams_recover_indonesian_morphology(self) -> None:
        """Why char_wb is in the feature union.

        `kenaikan` never appears in training; `naik` does. Word features alone
        cannot connect them, and Indonesian is full of exactly this.
        """
        model = fitted_tfidf()
        p = model.predict(["Kenaikan harga emas berlanjut"])[0]
        assert p.label == "ekonomi"

    def test_probabilities_are_calibrated_not_raw_distances(self) -> None:
        # LinearSVC emits signed distances from the hyperplane. A decision
        # function is not a confidence, and the Stage 4 cascade gates on
        # confidence.
        model = fitted_tfidf()
        p = model.predict(["Harga emas naik"])[0]
        assert all(0.0 <= v <= 1.0 for v in p.probabilities.values())
        assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-6)

    def test_calibration_source_is_recorded(self) -> None:
        """Calibrating on the fitting data understates the temperature.

        That is sometimes unavoidable, but it must be visible in the artifact
        rather than inferred.
        """
        without = fitted_tfidf()
        assert "no validation" in str(without.describe()["calibrated_on"])

        with_val = TfidfLinearSVC(min_df=1)
        with_val.fit(
            TRAIN_TEXTS,
            TRAIN_LABELS,
            val_texts=TRAIN_TEXTS[:6],
            val_labels=TRAIN_LABELS[:6],
        )
        assert with_val.describe()["calibrated_on"] == "validation"

    def test_is_reproducible_from_the_seed(self) -> None:
        a, b = TfidfLinearSVC(min_df=1, seed=7), TfidfLinearSVC(min_df=1, seed=7)
        a.fit(TRAIN_TEXTS, TRAIN_LABELS)
        b.fit(TRAIN_TEXTS, TRAIN_LABELS)
        assert a.predict(["Harga emas naik"])[0].label == (b.predict(["Harga emas naik"])[0].label)

    def test_empty_input_returns_empty(self) -> None:
        assert fitted_tfidf().predict([]) == []

    def test_predict_before_fit_is_an_error(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            TfidfLinearSVC().predict(["x"])

    def test_rejects_a_single_class(self) -> None:
        with pytest.raises(ValueError, match="single class"):
            TfidfLinearSVC(min_df=1).fit(["a", "b"], ["x", "x"])

    def test_measured_cost_overrides_the_assumption(self) -> None:
        model = fitted_tfidf()
        assumed = model.unit_cost_usd(1000)
        measured = model.unit_cost_usd(1000, measured_seconds_per_prediction=0.010)
        assert measured != assumed
        assert measured > 0


class TestLatencyMeasurement:
    def test_records_one_sample_per_example(self) -> None:
        model = fitted_tfidf()
        examples = [Example(title=t) for t in TRAIN_TEXTS[:8]]
        timing = measure_latency(model, examples, warmup=2)

        assert len(timing.samples_ms) == 8
        assert timing.p95_ms >= timing.p50_ms
        assert "batch=1, warm" in timing.summary()

    def test_warmup_predictions_are_excluded(self) -> None:
        """The first call through a fitted pipeline pays one-off costs.

        A serving process pays them once at boot and never again, so including
        them would measure the wrong thing — and would penalise the heavier
        candidates most.
        """
        model = fitted_tfidf()
        examples = [Example(title=t) for t in TRAIN_TEXTS[:5]]
        assert len(measure_latency(model, examples, warmup=3).samples_ms) == 5

    def test_empty_input_gives_an_empty_timing(self) -> None:
        assert measure_latency(fitted_tfidf(), []).samples_ms == []
