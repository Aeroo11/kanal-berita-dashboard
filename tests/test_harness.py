"""The harness that turns candidates plus a split into a recorded result.

Mostly about provenance: a number without its split hash is not a result, and
the machinery that makes two results comparable is the machinery that can show
they are not.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kanal.data.splits import Article, temporal_split
from kanal.eval.harness import (
    MDE,
    build_dataset,
    compare,
    score_candidate,
)
from kanal.features.text import Example
from kanal.models.majority import MajorityClass
from kanal.models.tfidf import TfidfLinearSVC

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
TAXONOMY = ["ekonomi", "olahraga", "politik"]

SEEDS = {
    "politik": [
        "Presiden resmikan bendungan baru di provinsi itu",
        "DPR sahkan undang-undang baru soal pemilu",
        "Menteri umumkan kebijakan pemerintah terbaru",
        "Partai politik gelar kongres nasional tahunan",
    ],
    "ekonomi": [
        "Harga emas menguat di tengah ketidakpastian pasar",
        "Bank sentral pertahankan suku bunga acuan",
        "Rupiah melemah terhadap dolar Amerika Serikat",
        "Inflasi tahunan tercatat naik tipis bulan lalu",
    ],
    "olahraga": [
        "Timnas futsal menang atas Thailand di final",
        "Persib segel tiket lolos ke semifinal turnamen",
        "Pemain muda cetak gol kemenangan di menit akhir",
        "Kompetisi liga utama dimulai akhir pekan ini",
    ],
}


def corpus() -> tuple[list[Article], dict[str, Example], dict[str, str]]:
    articles: list[Article] = []
    examples: dict[str, Example] = {}
    labels: dict[str, str] = {}

    i = 0
    # Old rows land in train, recent rows in test.
    for age, count in ((30, 12), (10, 4), (2, 8)):
        for _ in range(count):
            for kanal, titles in SEEDS.items():
                key = f"k{i}"
                title = f"{titles[i % len(titles)]} nomor {i}"
                articles.append(
                    Article(
                        article_key=key,
                        cluster_id=key,
                        published_at=NOW - timedelta(days=age),
                        source="antara",
                        kanal=kanal,
                    )
                )
                examples[key] = Example(title=title)
                labels[key] = kanal
                i += 1
    return articles, examples, labels


class TestDatasetConstruction:
    def test_every_route_to_a_model_goes_through_to_text(self) -> None:
        # The structural defence against train/serve skew: there is one path, and
        # build_dataset is on it.
        articles, examples, labels = corpus()
        manifest = temporal_split(articles, anchor=NOW)
        train = build_dataset(manifest, "train", examples=examples, labels=labels)

        from kanal.features.text import to_text

        assert train.texts == [to_text(examples[k]) for k in train.keys]

    def test_splits_do_not_overlap(self) -> None:
        articles, examples, labels = corpus()
        manifest = temporal_split(articles, anchor=NOW)
        train = build_dataset(manifest, "train", examples=examples, labels=labels)
        test = build_dataset(manifest, "test", examples=examples, labels=labels)
        assert set(train.keys).isdisjoint(test.keys)

    def test_carries_cluster_ids_for_the_bootstrap(self) -> None:
        # Without these the bootstrap silently falls back to resampling rows,
        # which understates the variance.
        articles, examples, labels = corpus()
        manifest = temporal_split(articles, anchor=NOW)
        test = build_dataset(manifest, "test", examples=examples, labels=labels)
        assert len(test.clusters) == len(test)


class TestScoring:
    def test_records_measured_latency_and_non_zero_cost(self) -> None:
        articles, examples, labels = corpus()
        manifest = temporal_split(articles, anchor=NOW)
        train = build_dataset(manifest, "train", examples=examples, labels=labels)
        test = build_dataset(manifest, "test", examples=examples, labels=labels)

        model = TfidfLinearSVC(min_df=1)
        model.fit(train.texts, train.labels)
        result, predicted = score_candidate(model, test, taxonomy=TAXONOMY)

        assert len(predicted) == len(test)
        assert result.p95_ms >= result.p50_ms > 0
        assert result.usd_per_1000 > 0, "self-hosted is not free"
        assert "temperature" in result.config

    def test_the_baseline_is_scored_by_the_same_code(self) -> None:
        # The harness must not be able to treat candidates differently, because
        # it cannot tell them apart.
        articles, examples, labels = corpus()
        manifest = temporal_split(articles, anchor=NOW)
        train = build_dataset(manifest, "train", examples=examples, labels=labels)
        test = build_dataset(manifest, "test", examples=examples, labels=labels)

        base = MajorityClass()
        base.fit(train.texts, train.labels)
        result, _ = score_candidate(base, test, taxonomy=TAXONOMY)

        assert result.name == "majority"
        assert result.usd_per_1000 > 0


class TestComparison:
    def _fixture(self) -> tuple[object, list[str], list[str]]:
        articles, examples, labels = corpus()
        manifest = temporal_split(articles, anchor=NOW)
        train = build_dataset(manifest, "train", examples=examples, labels=labels)
        test = build_dataset(manifest, "test", examples=examples, labels=labels)

        base = MajorityClass()
        base.fit(train.texts, train.labels)
        model = TfidfLinearSVC(min_df=1)
        model.fit(train.texts, train.labels)

        _, base_pred = score_candidate(base, test, taxonomy=TAXONOMY)
        _, model_pred = score_candidate(model, test, taxonomy=TAXONOMY)
        return test, base_pred, model_pred

    def test_a_real_improvement_passes_both_tests(self) -> None:
        test, base_pred, model_pred = self._fixture()
        c = compare(
            test,
            "majority",
            base_pred,
            "tfidf",
            model_pred,
            taxonomy=TAXONOMY,
            resamples=500,
        )
        assert c.passes_quality_gate
        assert c.reasons() == []

    def test_a_candidate_against_itself_is_refused_with_reasons(self) -> None:
        """The gate must say no, and say why, in the words the log will use."""
        test, _, model_pred = self._fixture()
        c = compare(
            test,
            "tfidf",
            model_pred,
            "tfidf-copy",
            model_pred,
            taxonomy=TAXONOMY,
            resamples=500,
        )
        assert not c.passes_quality_gate
        reasons = " ".join(c.reasons())
        assert "does not clear the MDE" in reasons
        assert str(MDE) in reasons
        assert "McNemar" in reasons

    def test_both_tests_are_required_not_either(self) -> None:
        test, _, model_pred = self._fixture()
        c = compare(test, "a", model_pred, "b", model_pred, taxonomy=TAXONOMY, resamples=200)
        # Identical predictions: the bootstrap gives a zero interval and McNemar
        # gives p=1. Both must be reported, so a future reader can see which
        # gate a real challenger failed on.
        assert not c.bootstrap.clears(MDE)
        assert not c.mcnemar.significant()
        assert len(c.reasons()) == 2


class TestRunResultProvenance:
    def test_a_result_names_the_split_it_ran_against(self, tmp_path: Path) -> None:
        """Two candidates compared on different splits are not compared at all.

        Without the hash on the result, that mistake is invisible — the numbers
        still look comparable.
        """
        from kanal.eval.harness import RunResult

        articles, _, _ = corpus()
        manifest = temporal_split(articles, anchor=NOW)
        run = RunResult(
            split_hash=manifest.hash,
            split_anchor=manifest.anchor,
            n_train=manifest.counts.get("train", 0),
            n_val=manifest.counts.get("val", 0),
            n_test=manifest.counts.get("test", 0),
            is_provisional=manifest.is_provisional,
            warnings=manifest.warnings,
        )
        loaded = json.loads(run.write(tmp_path / "run.json").read_text(encoding="utf-8"))

        assert loaded["split_hash"] == manifest.hash
        assert loaded["n_train"] == manifest.counts.get("train", 0)
        assert "is_provisional" in loaded

    def test_provisional_is_surfaced_in_the_summary(self) -> None:
        from kanal.eval.harness import RunResult

        run = RunResult(
            split_hash="abc123",
            split_anchor="x",
            n_train=10,
            n_val=1,
            n_test=5,
            is_provisional=True,
            warnings=["test is larger than train"],
        )
        assert "PROVISIONAL" in run.summary()
        assert "larger than train" in run.summary()
