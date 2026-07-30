"""The landing zone, and the idempotency the whole pipeline rests on.

RSS is a sliding window: an hour that is not captured cannot be recovered. That
forces hourly polling, and hourly polling means seeing the same articles over
and over. If landing were not idempotent the store would grow duplicates
without bound and every downstream count would be wrong.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kanal.ingest.land import count_articles, existing_keys, land_articles, partition_dir
from kanal.ingest.parse import Article

FETCHED = datetime(2026, 7, 29, 8, 30, tzinfo=UTC)


def make_article(n: int, source: str = "antara") -> Article:
    return Article(
        article_key=f"key{n:04d}",
        canonical_url=f"https://antaranews.com/berita/{n}/judul",
        title_fingerprint=f"fp{n:04d}",
        title=f"Judul berita {n}",
        summary=f"Ringkasan {n}",
        kanal="ekonomi",
        source=source,
        channel="ekonomi",
        feed_id=f"{source}:ekonomi",
        raw_link=f"https://antaranews.com/berita/{n}/judul",
        published_at="2026-07-29T08:00:00+00:00",
        fetched_at=FETCHED.isoformat(),
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "raw"


class TestPartitioning:
    def test_partitions_by_source_and_utc_date(self, root: Path) -> None:
        path = partition_dir("antara", FETCHED, root)
        assert path == root / "source=antara" / "dt=2026-07-29"

    def test_uses_utc_not_local_time(self, root: Path) -> None:
        # 23:30 UTC is already "tomorrow" in Jakarta (UTC+7). The partition must
        # follow UTC, or the same cycle would split across two days depending on
        # where the runner happens to live.
        late = datetime(2026, 7, 29, 23, 30, tzinfo=UTC)
        assert partition_dir("antara", late, root).name == "dt=2026-07-29"


class TestIdempotency:
    def test_second_landing_of_the_same_articles_writes_nothing(self, root: Path) -> None:
        articles = [make_article(i) for i in range(10)]

        first = land_articles(articles, feed_id="antara:ekonomi", fetched_at=FETCHED, root=root)
        assert first.written == 10
        assert first.skipped_duplicates == 0

        second = land_articles(articles, feed_id="antara:ekonomi", fetched_at=FETCHED, root=root)
        assert second.written == 0
        assert second.skipped_duplicates == 10
        assert second.path is None

        assert count_articles(root) == 10

    def test_only_genuinely_new_articles_land(self, root: Path) -> None:
        land_articles(
            [make_article(i) for i in range(5)],
            feed_id="antara:ekonomi",
            fetched_at=FETCHED,
            root=root,
        )
        # An hour later the feed has three new items and keeps two old ones.
        result = land_articles(
            [make_article(i) for i in range(3, 8)],
            feed_id="antara:ekonomi",
            fetched_at=FETCHED,
            root=root,
        )
        assert result.written == 3
        assert result.skipped_duplicates == 2
        assert count_articles(root) == 8

    def test_dedup_survives_a_restart(self, root: Path) -> None:
        # `existing_keys` reads from disk rather than trusting an in-memory set,
        # because a restart is exactly when idempotency has to hold.
        articles = [make_article(i) for i in range(4)]
        land_articles(articles, feed_id="antara:ekonomi", fetched_at=FETCHED, root=root)

        recovered = existing_keys("antara", FETCHED, root)
        assert recovered == {a.article_key for a in articles}

    def test_duplicates_within_one_batch_are_collapsed(self, root: Path) -> None:
        doubled = [make_article(1), make_article(1), make_article(2)]
        result = land_articles(doubled, feed_id="antara:ekonomi", fetched_at=FETCHED, root=root)
        assert result.written == 2
        assert result.skipped_duplicates == 1


class TestAcrossDayBoundaries:
    """The bug that only appeared after midnight.

    Partitions are per UTC day, and the first version of `existing_keys` read
    only the current day's. Within a day it was correct; across one it was not —
    every article still sitting in a feed looked unseen again, and the next
    cycle re-landed all of them.

    Measured on the real data after two days of hourly ingestion: 1,353 lines
    for 824 distinct articles, 39.1% redundant, with 529 articles appearing in
    more than one partition. The unit tests all passed throughout, because none
    of them crossed a date.
    """

    def test_an_article_seen_yesterday_is_not_relanded_today(self, root: Path) -> None:
        yesterday = FETCHED - timedelta(days=1)
        land_articles(
            [make_article(i) for i in range(5)],
            feed_id="antara:ekonomi",
            fetched_at=yesterday,
            root=root,
        )

        # The same five are still in the feed the next day.
        result = land_articles(
            [make_article(i) for i in range(5)],
            feed_id="antara:ekonomi",
            fetched_at=FETCHED,
            root=root,
        )

        assert result.written == 0
        assert result.skipped_duplicates == 5
        assert count_articles(root) == 5

    def test_genuinely_new_articles_still_land_the_next_day(self, root: Path) -> None:
        yesterday = FETCHED - timedelta(days=1)
        land_articles(
            [make_article(i) for i in range(5)],
            feed_id="antara:ekonomi",
            fetched_at=yesterday,
            root=root,
        )
        result = land_articles(
            [make_article(i) for i in range(3, 8)],
            feed_id="antara:ekonomi",
            fetched_at=FETCHED,
            root=root,
        )
        assert result.written == 3
        assert result.skipped_duplicates == 2

    def test_the_lookback_window_is_bounded(self, root: Path) -> None:
        # An article older than the window re-lands. That is the deliberate
        # trade: ANTARA keeps evergreen explainers in its feeds for months, and
        # scanning far enough back to catch those would mean scanning the whole
        # history on every cycle. The warehouse anti-join makes the modelled
        # data unaffected either way.
        long_ago = FETCHED - timedelta(days=10)
        land_articles([make_article(1)], feed_id="antara:ekonomi", fetched_at=long_ago, root=root)
        result = land_articles(
            [make_article(1)], feed_id="antara:ekonomi", fetched_at=FETCHED, root=root
        )
        assert result.written == 1

    def test_lookback_is_configurable(self, root: Path) -> None:
        old = FETCHED - timedelta(days=6)
        land_articles([make_article(1)], feed_id="antara:ekonomi", fetched_at=old, root=root)

        seen_narrow = existing_keys("antara", FETCHED, root, lookback_days=2)
        seen_wide = existing_keys("antara", FETCHED, root, lookback_days=10)

        assert "key0001" not in seen_narrow
        assert "key0001" in seen_wide


class TestIsolation:
    def test_sources_do_not_share_a_partition(self, root: Path) -> None:
        land_articles(
            [make_article(1, "antara")], feed_id="antara:x", fetched_at=FETCHED, root=root
        )
        # Same article_key, different publisher — must not be treated as seen.
        result = land_articles(
            [make_article(1, "cnn")], feed_id="cnn:x", fetched_at=FETCHED, root=root
        )
        assert result.written == 1
        assert count_articles(root) == 2


class TestDurability:
    def test_writes_are_append_only(self, root: Path) -> None:
        # Two cycles produce two files; the first is never rewritten, so an
        # interrupted run cannot corrupt what already landed.
        land_articles([make_article(1)], feed_id="antara:ekonomi", fetched_at=FETCHED, root=root)
        later = FETCHED.replace(hour=9)
        land_articles([make_article(2)], feed_id="antara:ekonomi", fetched_at=later, root=root)

        files = sorted(partition_dir("antara", FETCHED, root).glob("*.jsonl"))
        assert len(files) == 2

    def test_two_cycles_in_the_same_second_do_not_collide(self, root: Path) -> None:
        land_articles([make_article(1)], feed_id="antara:ekonomi", fetched_at=FETCHED, root=root)
        land_articles([make_article(2)], feed_id="antara:ekonomi", fetched_at=FETCHED, root=root)
        files = list(partition_dir("antara", FETCHED, root).glob("*.jsonl"))
        assert len(files) == 2

    def test_a_truncated_line_does_not_abort_the_read(self, root: Path) -> None:
        # A process killed mid-write leaves a partial final line. Losing that
        # one row is acceptable; refusing to read the partition is not.
        land_articles(
            [make_article(i) for i in range(3)],
            feed_id="antara:ekonomi",
            fetched_at=FETCHED,
            root=root,
        )
        target = next(partition_dir("antara", FETCHED, root).glob("*.jsonl"))
        with target.open("a", encoding="utf-8") as fh:
            fh.write('{"article_key": "trunc')

        assert existing_keys("antara", FETCHED, root) == {"key0000", "key0001", "key0002"}

    def test_rows_round_trip_as_utf8_json(self, root: Path) -> None:
        article = make_article(1)
        article.title = "Harga emas naik, rupiah menguat — pasar bereaksi"
        land_articles([article], feed_id="antara:ekonomi", fetched_at=FETCHED, root=root)

        file = next(partition_dir("antara", FETCHED, root).glob("*.jsonl"))
        row = json.loads(file.read_text(encoding="utf-8").strip())
        assert row["title"] == article.title
        assert row["schema_version"] == 1


class TestCounting:
    def test_empty_landing_zone_counts_zero(self, root: Path) -> None:
        assert count_articles(root) == 0

    def test_counts_across_sources_and_days(self, root: Path) -> None:
        land_articles([make_article(1, "antara")], feed_id="a:x", fetched_at=FETCHED, root=root)
        land_articles([make_article(2, "cnn")], feed_id="c:x", fetched_at=FETCHED, root=root)
        tomorrow = FETCHED.replace(day=30)
        land_articles([make_article(3, "antara")], feed_id="a:x", fetched_at=tomorrow, root=root)
        assert count_articles(root) == 3
