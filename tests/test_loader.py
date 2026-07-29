"""The warehouse loader.

The loader is the only writer in the system, so its two properties are
load-bearing: loading the same files twice must add nothing, and a load must
only read files it has not already consumed. Everything downstream assumes
`raw_articles` has exactly one row per article.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kanal.warehouse.duck import connect, duckdb_settings, reader
from kanal.warehouse.loader import load

FETCHED = datetime(2026, 7, 29, 8, 0, tzinfo=UTC)


def write_partition(root: Path, source: str, name: str, keys: list[int]) -> Path:
    directory = root / f"source={source}" / "dt=2026-07-29"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for k in keys:
            fh.write(
                json.dumps(
                    {
                        "article_key": f"key{k:04d}",
                        "canonical_url": f"https://antaranews.com/berita/{k}/judul",
                        "title_fingerprint": f"fp{k:04d}",
                        "title": f"Judul berita {k}",
                        "summary": f"Ringkasan {k}",
                        "kanal": "ekonomi",
                        "source": source,
                        "channel": "ekonomi",
                        "feed_id": f"{source}:ekonomi",
                        "raw_link": f"https://antaranews.com/berita/{k}/judul",
                        "published_at": "2026-07-29T07:00:00+00:00",
                        "fetched_at": FETCHED.isoformat(),
                        "schema_version": 1,
                        "extra": {"author": "Redaksi"},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


@pytest.fixture
def env(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "raw", tmp_path / "kanal.duckdb"


def count_rows(db: Path) -> int:
    with reader(db) as c:
        return int(c.execute("SELECT count(*) FROM raw_articles").fetchone()[0])


class TestSettings:
    def test_pins_everything_that_defaults_to_a_host_value(self) -> None:
        s = duckdb_settings()
        # Each of these defaults to something derived from the machine, which
        # is wrong inside a runner or a container.
        assert s["memory_limit"]
        assert s["threads"]
        assert Path(s["temp_directory"]).is_absolute()
        assert s["autoload_known_extensions"] == "false"
        assert s["autoinstall_known_extensions"] == "false"

    def test_settings_actually_reach_the_connection(self) -> None:
        # `connect`, not `reader`: an in-memory database cannot be opened
        # read-only, and readers exist to open a *file* the writer owns.
        conn = connect(":memory:")
        try:
            applied = dict(
                conn.execute(
                    "SELECT name, value FROM duckdb_settings() "
                    "WHERE name IN ('threads','autoload_known_extensions')"
                ).fetchall()
            )
        finally:
            conn.close()

        assert applied["threads"] == duckdb_settings()["threads"]
        # duckdb_settings() reports this as the string "false" on some builds
        # and as a real bool on others, so compare on meaning rather than type.
        assert str(applied["autoload_known_extensions"]).lower() == "false"


class TestLoading:
    def test_loads_an_empty_landing_zone_without_error(self, env: tuple[Path, Path]) -> None:
        raw, db = env
        report = load(raw, db)
        assert report.files_seen == 0
        assert report.rows_added == 0

    def test_loads_every_row(self, env: tuple[Path, Path]) -> None:
        raw, db = env
        write_partition(raw, "antara", "a", list(range(10)))
        report = load(raw, db)
        assert report.rows_added == 10
        assert count_rows(db) == 10

    def test_preserves_the_extra_blob(self, env: tuple[Path, Path]) -> None:
        raw, db = env
        write_partition(raw, "antara", "a", [1])
        load(raw, db)
        with reader(db) as c:
            extra = c.execute("SELECT extra FROM raw_articles").fetchone()[0]
        # Whatever the publisher sent survives, so a feed changing shape is
        # recoverable rather than a day of data silently discarded.
        assert "Redaksi" in str(extra)

    def test_records_which_file_each_row_came_from(self, env: tuple[Path, Path]) -> None:
        raw, db = env
        write_partition(raw, "antara", "morning", [1, 2])
        load(raw, db)
        with reader(db) as c:
            rows = c.execute("SELECT DISTINCT _ingest_file FROM raw_articles").fetchall()
        assert {r[0] for r in rows} == {"source=antara/dt=2026-07-29/morning.jsonl"}


class TestIdempotency:
    def test_reloading_the_same_files_adds_nothing(self, env: tuple[Path, Path]) -> None:
        raw, db = env
        write_partition(raw, "antara", "a", list(range(10)))

        first = load(raw, db)
        second = load(raw, db)

        assert first.rows_added == 10
        assert second.files_loaded == 0
        assert second.rows_added == 0
        assert count_rows(db) == 10

    def test_only_unseen_files_are_read(self, env: tuple[Path, Path]) -> None:
        raw, db = env
        write_partition(raw, "antara", "a", [1, 2, 3])
        load(raw, db)

        write_partition(raw, "antara", "b", [4, 5])
        report = load(raw, db)

        assert report.files_seen == 2
        assert report.files_loaded == 1  # only the new one was opened
        assert report.rows_added == 2
        assert count_rows(db) == 5

    def test_the_same_article_in_two_files_lands_once(self, env: tuple[Path, Path]) -> None:
        raw, db = env
        # A wire story syndicated twice, or a feed repeating an item across
        # cycles: the anti-join resolves both to one row.
        write_partition(raw, "antara", "a", [1, 2, 3])
        write_partition(raw, "cnn", "b", [3, 4, 5])

        report = load(raw, db)
        assert report.rows_read == 6
        assert report.rows_added == 5
        assert count_rows(db) == 5

    def test_force_rereads_without_duplicating(self, env: tuple[Path, Path]) -> None:
        raw, db = env
        write_partition(raw, "antara", "a", [1, 2, 3])
        load(raw, db)

        forced = load(raw, db, force=True)
        assert forced.files_loaded == 1  # re-opened
        assert forced.rows_added == 0  # but the anti-join held
        assert count_rows(db) == 3


class TestFreshnessContract:
    """ANTARA mixes evergreen explainers into its news feeds.

    Measured on the first real cycle: 141 of 220 ANTARA items were more than 30
    days old — profiles, "mengenal…" pieces and fixture lists that sit in the
    feed indefinitely — while CNN had none at all.

    This is a property of the source, not a parsing bug, and it has teeth: a
    naive temporal split would push almost all ANTARA rows into train and leave
    a test set dominated by CNN, the leakiest source. The split has to be built
    knowing this, so the warehouse has to be able to state it.
    """

    def test_article_age_is_queryable_per_source(self, env: tuple[Path, Path]) -> None:
        raw, db = env
        write_partition(raw, "antara", "a", [1, 2])
        load(raw, db)

        with reader(db) as c:
            rows = c.execute(
                """
                SELECT source,
                       count(*) AS n,
                       count(*) FILTER (
                           WHERE published_at < now() - INTERVAL 30 DAY
                       ) AS stale
                FROM raw_articles GROUP BY 1
                """
            ).fetchall()

        assert rows == [("antara", 2, 0)]

    def test_null_timestamps_do_not_break_the_age_query(self, env: tuple[Path, Path]) -> None:
        raw, db = env
        directory = raw / "source=antara" / "dt=2026-07-29"
        directory.mkdir(parents=True)
        (directory / "a.jsonl").write_text(
            json.dumps(
                {
                    "article_key": "k1",
                    "canonical_url": "https://a.com/1",
                    "title_fingerprint": "f1",
                    "title": "Judul",
                    "summary": "",
                    "kanal": "ekonomi",
                    "source": "antara",
                    "channel": "ekonomi",
                    "feed_id": "antara:ekonomi",
                    "raw_link": "https://a.com/1",
                    "published_at": None,  # feedparser could not read the date
                    "fetched_at": FETCHED.isoformat(),
                    "schema_version": 1,
                    "extra": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        load(raw, db)

        with reader(db) as c:
            n, missing = c.execute(
                "SELECT count(*), count(*) FILTER (WHERE published_at IS NULL) FROM raw_articles"
            ).fetchone()
        # The row is kept. A guessed timestamp would corrupt the temporal split;
        # a dropped row would lose a real article. Null is the honest option.
        assert (n, missing) == (1, 1)
