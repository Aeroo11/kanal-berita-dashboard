"""The landing zone.

Layout::

    data/raw/source={source}/dt={YYYY-MM-DD}/{feed}-{HHMMSS}.jsonl

Partitioned by source and UTC date so that a day can be reprocessed in
isolation, and so DuckDB and pyarrow can read the tree directly with partition
pruning. One file per feed per cycle keeps writes append-only — nothing is ever
rewritten, so an interrupted run cannot corrupt what came before.

Idempotency lives *here* rather than downstream: a cycle first reads which
`article_key`s the current partition already holds and writes only what is new.
Running an hourly poll twice therefore produces zero new rows the second time,
which is the property `make ingest` twice-in-a-row demonstrates and
`tests/test_land.py` asserts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kanal.config import settings
from kanal.ingest.parse import Article

log = logging.getLogger(__name__)


@dataclass
class LandingResult:
    written: int
    skipped_duplicates: int
    path: Path | None


def partition_dir(source: str, day: datetime, root: Path | None = None) -> Path:
    base = root or settings.raw_dir
    return base / f"source={source}" / f"dt={day.astimezone(UTC).date().isoformat()}"


def existing_keys(
    source: str,
    day: datetime,
    root: Path | None = None,
    lookback_days: int | None = None,
) -> set[str]:
    """Every `article_key` already present in the recent partitions.

    Reading the partitions back is deliberate. The alternative — trusting an
    in-memory set carried across the run — breaks the moment the process
    restarts, which is exactly when idempotency matters.

    **Why a lookback rather than just today.** The first version read only the
    current day's partition, which was correct within a day and wrong across
    one: at midnight UTC every article still sitting in a feed was no longer
    "seen", so the next cycle re-landed all of them. Measured on the real data
    after two days, 39.1% of lines were re-landings — 529 of 824 articles
    appeared in more than one partition.

    A feed is a sliding window of its last 25–100 items, so a window of a few
    days covers everything a publisher is still advertising. ANTARA's evergreen
    explainers sit in its feeds for months and will still re-land once per
    lookback period; bounding that at a few times a year is the point, and the
    warehouse anti-join means the modelled data is unaffected either way.
    """
    lookback = settings.landing_lookback_days if lookback_days is None else lookback_days

    keys: set[str] = set()
    for offset in range(lookback + 1):
        directory = partition_dir(source, day - timedelta(days=offset), root)
        if not directory.exists():
            continue
        for file in directory.glob("*.jsonl"):
            try:
                with file.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            keys.add(json.loads(line)["article_key"])
                        except (json.JSONDecodeError, KeyError):
                            # A partially written final line from a killed
                            # process. Skipping it means we may re-write that one
                            # article; far better than aborting the read.
                            continue
            except OSError as exc:
                log.warning("could not read %s: %s", file, exc)
    return keys


def land_articles(
    articles: Iterable[Article],
    *,
    feed_id: str,
    fetched_at: datetime,
    root: Path | None = None,
    known_keys: set[str] | None = None,
) -> LandingResult:
    """Append new articles to today's partition. Returns what actually landed."""
    articles = list(articles)
    if not articles:
        return LandingResult(0, 0, None)

    source = articles[0].source
    seen = known_keys if known_keys is not None else existing_keys(source, fetched_at, root)

    fresh: list[Article] = []
    duplicates = 0
    for article in articles:
        if article.article_key in seen:
            duplicates += 1
            continue
        seen.add(article.article_key)
        fresh.append(article)

    if not fresh:
        return LandingResult(0, duplicates, None)

    directory = partition_dir(source, fetched_at, root)
    directory.mkdir(parents=True, exist_ok=True)

    safe_feed = feed_id.replace(":", "-").replace("/", "-")
    stamp = fetched_at.astimezone(UTC).strftime("%H%M%S")
    path = directory / f"{safe_feed}-{stamp}.jsonl"

    # A feed polled twice within the same second would otherwise collide.
    suffix = 1
    while path.exists():
        path = directory / f"{safe_feed}-{stamp}-{suffix}.jsonl"
        suffix += 1

    with path.open("w", encoding="utf-8") as fh:
        for article in fresh:
            fh.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")

    log.info("%s: landed %d new (%d dup) → %s", feed_id, len(fresh), duplicates, path.name)
    return LandingResult(len(fresh), duplicates, path)


def count_articles(root: Path | None = None) -> int:
    """Total rows in the landing zone. Used by the CLI summary and asset checks."""
    base = root or settings.raw_dir
    if not base.exists():
        return 0
    total = 0
    for file in base.rglob("*.jsonl"):
        try:
            with file.open("r", encoding="utf-8") as fh:
                total += sum(1 for line in fh if line.strip())
        except OSError:
            continue
    return total
