"""Export, dataset card, and the upload guard rails.

The card is generated rather than written because its figures must match the
Parquet beside it. A card claiming 1,295 articles next to a file holding 824 is
worse than a card with no numbers — and hand-maintained figures drift the moment
ingestion runs again. These tests pin that coupling.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kanal.publish.card import build_card
from kanal.publish.export import PUBLISHED_COLUMNS, export
from kanal.publish.hub import MissingTokenError, upload
from kanal.warehouse.duck import connect

FETCHED = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


@pytest.fixture
def warehouse(tmp_path: Path) -> Path:
    """A minimal `fct_articles` with the properties the card reports on."""
    db = tmp_path / "kanal.duckdb"
    conn = connect(db)
    try:
        conn.execute(
            """
            CREATE TABLE fct_articles (
                article_key VARCHAR, title VARCHAR, summary VARCHAR, kanal VARCHAR,
                source VARCHAR, channel VARCHAR, canonical_url VARCHAR,
                published_at TIMESTAMP, fetched_at TIMESTAMP,
                is_evergreen BOOLEAN, url_leaks_label BOOLEAN,
                url_leaks_channel BOOLEAN, url_leaks_canonical BOOLEAN,
                cluster_id VARCHAR, cluster_size BIGINT,
                is_cross_source_duplicate BOOLEAN, has_label_disagreement BOOLEAN,
                label_is_judgement_call BOOLEAN, title_words BIGINT
            )
            """
        )
        rows = [
            # A clean source, and a totally leaky one.
            (
                "k1",
                "Judul bersih",
                "Ringkasan",
                "ekonomi",
                "antara",
                "ekonomi",
                "https://antaranews.com/berita/1/x",
                False,
                False,
                False,
                False,
                "c1",
                1,
                False,
                False,
                False,
                2,
            ),
            (
                "k2",
                "Judul lain",
                "Ringkasan",
                "politik",
                "antara",
                "politik",
                "https://antaranews.com/berita/2/x",
                True,
                False,
                False,
                False,
                "c2",
                1,
                False,
                False,
                True,
                2,
            ),
            (
                "k3",
                "Judul bocor",
                "Ringkasan",
                "ekonomi",
                "cnn",
                "ekonomi",
                "https://cnnindonesia.com/ekonomi/3/x",
                False,
                True,
                True,
                True,
                "c3",
                1,
                False,
                False,
                False,
                2,
            ),
            (
                "k4",
                "Judul bocor lagi",
                "Ringkasan",
                "olahraga",
                "cnn",
                "olahraga",
                "https://cnnindonesia.com/olahraga/4/x",
                False,
                True,
                True,
                True,
                "c4",
                1,
                False,
                False,
                False,
                3,
            ),
        ]
        conn.executemany(
            f"INSERT INTO fct_articles VALUES ({', '.join(['?'] * 19)})",
            [(*r[:7], FETCHED, FETCHED, *r[7:]) for r in rows],
        )
    finally:
        conn.close()
    return db


class TestExport:
    def test_writes_parquet_and_stats(self, warehouse: Path, tmp_path: Path) -> None:
        out = tmp_path / "export"
        report = export(out_dir=out, db_path=warehouse)

        assert report.parquet_path.exists()
        assert report.stats_path.exists()
        assert report.articles == 4
        assert report.sources == 2

    def test_publishes_an_explicit_column_list(self) -> None:
        # `SELECT *` would mean adding an internal column to the fact table
        # silently starts publishing it.
        assert "*" not in PUBLISHED_COLUMNS
        for expected in ("title", "summary", "kanal", "url_leaks_label", "cluster_id"):
            assert expected in PUBLISHED_COLUMNS

    def test_does_not_publish_internal_columns(self, warehouse: Path, tmp_path: Path) -> None:
        out = tmp_path / "export"
        report = export(out_dir=out, db_path=warehouse)

        conn = connect(":memory:")
        try:
            cols = {
                r[0]
                for r in conn.execute(
                    f"DESCRIBE SELECT * FROM '{report.parquet_path.as_posix()}'"
                ).fetchall()
            }
        finally:
            conn.close()

        # These exist in the fact table but are implementation detail.
        assert "url_leaks_channel" not in cols
        assert "url_leaks_canonical" not in cols
        # Provenance *is* published, on purpose: a user cannot reproduce the
        # leakage measurement without it.
        assert {"source", "channel", "canonical_url"} <= cols

    def test_stats_record_the_clean_subset(self, warehouse: Path, tmp_path: Path) -> None:
        report = export(out_dir=tmp_path / "export", db_path=warehouse)
        stats = json.loads(report.stats_path.read_text(encoding="utf-8"))

        # Two of the four rows are clean. This is the number a user needs, and it
        # is a per-row property that the per-source averages hide.
        assert stats["rows_without_url_leak"] == 2
        assert stats["url_leak_rate_by_source"]["cnn"] == 1.0
        assert stats["url_leak_rate_by_source"]["antara"] == 0.0

    def test_fails_loudly_on_an_empty_warehouse(self, tmp_path: Path) -> None:
        db = tmp_path / "empty.duckdb"
        conn = connect(db)
        try:
            conn.execute("CREATE TABLE fct_articles (article_key VARCHAR)")
        finally:
            conn.close()
        # Exporting nothing quietly would publish an empty dataset over a good one.
        with pytest.raises(Exception):  # noqa: B017 - DuckDB or our own guard
            export(out_dir=tmp_path / "export", db_path=db)


class TestCard:
    def test_figures_come_from_the_stats_file(self, warehouse: Path, tmp_path: Path) -> None:
        report = export(out_dir=tmp_path / "export", db_path=warehouse)
        card = build_card(report.stats_path)

        assert "4 articles" in card or "**4 articles" in card
        # The leakage table must carry the measured rates, not prose.
        assert "100.0%" in card
        assert "`cnn`" in card

    def test_warns_about_leakage_before_anything_else(
        self, warehouse: Path, tmp_path: Path
    ) -> None:
        report = export(out_dir=tmp_path / "export", db_path=warehouse)
        card = build_card(report.stats_path)

        leakage_at = card.index("gives the label away")
        distribution_at = card.index("## Distribution")
        # Someone who reads only the first screen must still learn that the URL
        # is poison. Ordering is the only mechanism for that.
        assert leakage_at < distribution_at

    def test_states_the_clean_subset_size(self, warehouse: Path, tmp_path: Path) -> None:
        report = export(out_dir=tmp_path / "export", db_path=warehouse)
        card = build_card(report.stats_path)
        assert "url_leaks_label" in card
        assert "2 of 4 rows" in card

    def test_declares_licence_and_attribution(self, warehouse: Path, tmp_path: Path) -> None:
        report = export(out_dir=tmp_path / "export", db_path=warehouse)
        card = build_card(report.stats_path)

        assert "license: cc-by-4.0" in card
        assert "remain the property of their publishers" in card
        # The opt-out has to be in the card, not only in the repository.
        assert "open an issue" in card

    def test_is_valid_frontmatter(self, warehouse: Path, tmp_path: Path) -> None:
        report = export(out_dir=tmp_path / "export", db_path=warehouse)
        card = build_card(report.stats_path)
        # Hugging Face rejects a card whose frontmatter is not first.
        assert card.startswith("---\n")
        assert card.count("\n---\n") >= 1


class TestUploadGuards:
    def test_refuses_without_a_token(
        self, warehouse: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "export"
        report = export(out_dir=out, db_path=warehouse)
        (out / "README.md").write_text(build_card(report.stats_path), encoding="utf-8")

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

        with pytest.raises(MissingTokenError):
            upload(export_dir=out, repo_id="someone/kanal-test")

    def test_refuses_when_the_export_is_incomplete(self, tmp_path: Path) -> None:
        empty = tmp_path / "export"
        empty.mkdir()
        # Pushing a partial export would leave the dataset in a state where the
        # card and the data disagree.
        with pytest.raises(FileNotFoundError, match="Run `kanal export` first"):
            upload(export_dir=empty, repo_id="someone/kanal-test", dry_run=True)

    def test_dry_run_needs_no_token(
        self, warehouse: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "export"
        report = export(out_dir=out, db_path=warehouse)
        (out / "README.md").write_text(build_card(report.stats_path), encoding="utf-8")

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

        result = upload(export_dir=out, repo_id="someone/kanal-test", dry_run=True)
        assert result.repo_id == "someone/kanal-test"
        assert "articles.parquet" in result.files
        assert result.commit_url is None
