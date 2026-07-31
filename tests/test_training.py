"""The training job, and the two judgement calls inside it.

Both could reasonably go the other way, so both are pinned:

- the incumbent is whatever the registry says, not this run's best candidate
- a run that promotes nothing is a success
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from kanal.data.splits import Article
from kanal.features.text import Example
from kanal.registry.store import CHAMPION, AliasNotSet, Registry
from kanal.training import run as training

NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

SEEDS = {
    "politik": [
        "Presiden resmikan bendungan baru di provinsi itu",
        "DPR sahkan undang-undang baru soal pemilu",
        "Menteri umumkan kebijakan pemerintah terbaru",
    ],
    "ekonomi": [
        "Harga emas menguat di tengah ketidakpastian pasar",
        "Bank sentral pertahankan suku bunga acuan",
        "Rupiah melemah terhadap dolar Amerika Serikat",
    ],
    "olahraga": [
        "Timnas futsal menang atas Thailand di final",
        "Persib segel tiket lolos ke semifinal turnamen",
        "Pemain muda cetak gol kemenangan di menit akhir",
    ],
}


def corpus(
    *, train_rows: int = 120, test_rows: int = 60
) -> tuple[list[Article], dict[str, Example], dict[str, str]]:
    """A corpus with genuine history, so the temporal split is usable.

    The sizes are chosen to clear the gate's own thresholds rather than to be
    small. A first version used 12 test rows per class and every promotion test
    failed on `n_test >= 150` — the gate working correctly against a fixture that
    could never satisfy it. Multiplied by three classes, these give 360 train and
    180 test.
    """
    articles: list[Article] = []
    examples: dict[str, Example] = {}
    labels: dict[str, str] = {}

    i = 0
    for age, count in ((40, train_rows), (10, 4), (2, test_rows)):
        for _ in range(count):
            for kanal, titles in SEEDS.items():
                key = f"k{i}"
                articles.append(
                    Article(
                        article_key=key,
                        cluster_id=key,
                        published_at=NOW - timedelta(days=age),
                        source="antara",
                        kanal=kanal,
                    )
                )
                examples[key] = Example(title=f"{titles[i % len(titles)]} nomor {i}")
                labels[key] = kanal
                i += 1
    return articles, examples, labels


@pytest.fixture
def warehouse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the warehouse, so the job runs without DuckDB."""
    monkeypatch.setattr(training, "_load_corpus", lambda: corpus())


class TestAUsableSplit:
    def test_it_promotes_and_the_registry_can_serve_the_result(
        self, warehouse: None, tmp_path: Path
    ) -> None:
        result = training.train_and_promote(registry_root=tmp_path / "reg", resamples=300)

        assert result.verdict == "PROMOTE"
        assert result.champion_after is not None

        registry = Registry(tmp_path / "reg")
        assert registry.resolve(CHAMPION) == result.champion_after
        artifact = registry.load_alias(CHAMPION)
        assert artifact.predict(["Harga emas naik tajam"])

    def test_the_artifact_carries_the_split_it_was_judged_on(
        self, warehouse: None, tmp_path: Path
    ) -> None:
        # Without it the next run cannot tell whether the champion's recorded
        # metrics are comparable to its own.
        result = training.train_and_promote(registry_root=tmp_path / "reg", resamples=200)
        artifact = Registry(tmp_path / "reg").load_alias(CHAMPION)
        assert artifact.meta.split_hash == result.split_hash

    def test_per_class_scores_are_recorded_for_the_next_regression_check(
        self, warehouse: None, tmp_path: Path
    ) -> None:
        # G4 compares against these. If they are not stored, the no-regression
        # gate silently has nothing to check and passes everything.
        training.train_and_promote(registry_root=tmp_path / "reg", resamples=200)
        metrics = Registry(tmp_path / "reg").load_alias(CHAMPION).meta.metrics
        assert any(k.startswith("f1_") for k in metrics)
        assert "usd_per_1000" in metrics

    def test_every_candidate_is_reported_not_only_the_winner(
        self, warehouse: None, tmp_path: Path
    ) -> None:
        result = training.train_and_promote(registry_root=tmp_path / "reg", resamples=200)
        names = {name for name, *_ in result.candidates}
        assert "majority" in names, "the baseline anchors every other number"
        assert "tfidf-linearsvc" in names


class TestRefusingIsSuccess:
    def test_a_second_identical_run_is_refused_and_changes_nothing(
        self, warehouse: None, tmp_path: Path
    ) -> None:
        """The important half of the gate, exercised end to end.

        The same data produces the same model, so the second run's challenger
        cannot beat the incumbent. It must be refused, the champion must not
        move, and the refusal must be in the log.
        """
        root = tmp_path / "reg"
        first = training.train_and_promote(registry_root=root, resamples=300)
        assert first.verdict == "PROMOTE"

        second = training.train_and_promote(registry_root=root, resamples=300)
        assert second.verdict == "HOLD"
        assert second.reasons, "a refusal must say why"
        assert second.champion_before == second.champion_after == first.champion_after

        registry = Registry(root)
        refusals = registry.refusals()
        assert len(refusals) == 1
        assert "G3 quality" in " ".join(refusals[0].reasons)

    def test_a_refused_challenger_is_not_registered(self, warehouse: None, tmp_path: Path) -> None:
        root = tmp_path / "reg"
        training.train_and_promote(registry_root=root, resamples=300)
        second = training.train_and_promote(registry_root=root, resamples=300)

        registry = Registry(root)
        stored = {p.name for p in registry.artifacts.iterdir()}
        assert second.decision is not None
        assert second.decision.challenger_id not in stored or second.verdict == "PROMOTE"


class TestAnUnusableSplit:
    def test_too_little_data_is_reported_rather_than_crashed_on(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The state this project is actually in most days.

        A corpus without enough collection history produces a split that cannot
        train anything. That must be a clear message, not a stack trace from
        somewhere inside sklearn.
        """
        monkeypatch.setattr(training, "_load_corpus", lambda: corpus(train_rows=1, test_rows=1))
        result = training.train_and_promote(registry_root=tmp_path / "reg", resamples=100)

        assert result.verdict == "UNUSABLE"
        assert "not usable for training" in " ".join(result.reasons)
        with pytest.raises(AliasNotSet):
            Registry(tmp_path / "reg").resolve(CHAMPION)

    def test_an_empty_warehouse_says_what_to_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def empty() -> Any:
            raise RuntimeError(
                "the warehouse holds no articles — run `kanal load` and `dbt build` first"
            )

        monkeypatch.setattr(training, "_load_corpus", empty)
        with pytest.raises(RuntimeError, match="kanal load"):
            training.train_and_promote(registry_root=tmp_path / "reg")


class TestProvenanceIsCarried:
    def test_the_split_warnings_reach_the_run_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The "test is larger than train" finding must not be lost between the
        # split and whoever reads the training log.
        monkeypatch.setattr(training, "_load_corpus", lambda: corpus(train_rows=1, test_rows=30))
        result = training.train_and_promote(registry_root=tmp_path / "reg", resamples=100)
        assert any("larger than train" in w for w in result.split_warnings)
        assert "larger than train" in result.summary()

    def test_a_provisional_result_says_so(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(training, "_load_corpus", lambda: corpus(train_rows=30, test_rows=2))
        result = training.train_and_promote(registry_root=tmp_path / "reg", resamples=100)
        assert result.is_provisional
        assert "PROVISIONAL" in result.summary()
