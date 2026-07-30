"""One ingestion cycle: poll every feed, land what is new, report what happened.

The report is not decoration. RSS is a sliding window — an hour that is not
captured is unrecoverable — so the cycle has to be loud about partial failure.
A run where two sources succeeded and one tripped its breaker is *not* a
success, and the summary says so.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kanal.config import settings
from kanal.ingest.fetch import FeedFetcher
from kanal.ingest.land import existing_keys, land_articles
from kanal.ingest.parse import parse_feed
from kanal.ingest.sources import ALL_FEEDS, Feed

log = logging.getLogger(__name__)


@dataclass
class FeedOutcome:
    feed: Feed
    status: str
    entries: int = 0
    landed: int = 0
    duplicates: int = 0
    parse_failures: int = 0
    error: str | None = None


@dataclass
class CycleReport:
    started_at: datetime
    finished_at: datetime
    outcomes: list[FeedOutcome] = field(default_factory=list)

    @property
    def landed(self) -> int:
        return sum(o.landed for o in self.outcomes)

    @property
    def duplicates(self) -> int:
        return sum(o.duplicates for o in self.outcomes)

    @property
    def failed_feeds(self) -> list[FeedOutcome]:
        return [o for o in self.outcomes if o.status in ("failed", "skipped_breaker")]

    @property
    def sources_seen(self) -> set[str]:
        return {o.feed.source for o in self.outcomes if o.status in ("ok", "not_modified")}

    @property
    def expected_sources(self) -> set[str]:
        """Sources this environment expects to reach.

        Excludes those declared unreachable here — CNN and Tempo answer 403 to
        GitHub's datacentre IPs while serving any request from Indonesia. They
        are still polled and still reported; they just cannot make an otherwise
        healthy cycle fail forever.
        """
        return {f.source for f in ALL_FEEDS} - settings.unreachable_sources

    @property
    def missing_sources(self) -> set[str]:
        """Expected publishers that produced nothing usable."""
        return self.expected_sources - self.sources_seen

    @property
    def unexpectedly_reachable(self) -> set[str]:
        """Sources declared unreachable that answered anyway.

        Worth surfacing: it means the declaration is stale and the exception can
        be removed. An exception nobody revisits is how a temporary workaround
        becomes permanent.
        """
        return self.sources_seen & settings.unreachable_sources

    @property
    def healthy(self) -> bool:
        """Every *expected* source produced at least one usable response.

        Deliberately per source, not per feed: an individual channel going quiet
        is normal, an entire publisher going dark is not.
        """
        return not self.missing_sources

    def by_source(self) -> dict[str, dict[str, int]]:
        """Per-source tallies. A publisher going dark is the failure that matters."""
        out: dict[str, dict[str, int]] = {}
        for o in self.outcomes:
            bucket = out.setdefault(o.feed.source, {"feeds": 0, "ok": 0, "landed": 0, "failed": 0})
            bucket["feeds"] += 1
            bucket["landed"] += o.landed
            if o.status in ("ok", "not_modified"):
                bucket["ok"] += 1
            else:
                bucket["failed"] += 1
        return out

    def summary(self) -> str:
        lines = [
            f"cycle {self.started_at:%Y-%m-%d %H:%M:%S} UTC "
            f"({(self.finished_at - self.started_at).total_seconds():.1f}s)",
            f"  landed {self.landed} new, {self.duplicates} already seen",
        ]

        by_status: dict[str, int] = {}
        for o in self.outcomes:
            by_status[o.status] = by_status.get(o.status, 0) + 1
        lines.append("  feeds: " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))

        lines.append("  per source:")
        for source, tally in sorted(self.by_source().items()):
            declared = source in settings.unreachable_sources
            if tally["ok"]:
                mark = "  "
            elif declared:
                # Failing as declared. Reported, but not a surprise.
                mark = " ~"
            else:
                mark = " !"
            note = "  (declared unreachable here)" if declared else ""
            lines.append(
                f"   {mark} {source:<10} {tally['ok']}/{tally['feeds']} feeds ok, "
                f"{tally['landed']} landed{note}"
            )

        # Every distinct failure reason, with the feeds it affected. Grouping
        # matters: 7 feeds failing for one reason is one problem, not seven, and
        # the reason is the only part worth acting on.
        if self.failed_feeds:
            reasons: dict[str, list[str]] = {}
            for o in self.failed_feeds:
                reasons.setdefault(f"{o.status}: {o.error}", []).append(o.feed.feed_id)
            lines.append("  failures:")
            for reason, feeds in sorted(reasons.items()):
                shown = ", ".join(feeds[:4]) + (
                    f" (+{len(feeds) - 4} more)" if len(feeds) > 4 else ""
                )
                lines.append(f"    {reason}")
                lines.append(f"      affected: {shown}")

        if self.unexpectedly_reachable:
            lines.append(
                f"  ~~ {', '.join(sorted(self.unexpectedly_reachable))} answered despite being "
                f"declared unreachable — the declaration is stale and can be removed"
            )

        if not self.healthy:
            lines.append(
                f"  !! NO usable response from: {', '.join(sorted(self.missing_sources))} "
                f"— this hour is lost for those sources and cannot be backfilled"
            )
        return "\n".join(lines)


def run_cycle(
    feeds: tuple[Feed, ...] = ALL_FEEDS,
    *,
    root: Path | None = None,
    sleep: bool = True,
) -> CycleReport:
    started = datetime.now(UTC)
    outcomes: list[FeedOutcome] = []

    # Read each partition's keys once per cycle rather than once per feed —
    # a source has many feeds and they all land in the same partition.
    key_cache: dict[str, set[str]] = {}

    with FeedFetcher(sleep=sleep) as fetcher:
        for index, feed in enumerate(feeds):
            outcomes.append(_poll_one(fetcher, feed, root, key_cache))

            # Space out requests; the delay belongs *between* feeds, so the
            # last one does not pay for a pause nobody is waiting on.
            if sleep and index < len(feeds) - 1:
                time.sleep(settings.inter_request_delay_s)

    return CycleReport(started, datetime.now(UTC), outcomes)


def _poll_one(
    fetcher: FeedFetcher,
    feed: Feed,
    root: Path | None,
    key_cache: dict[str, set[str]],
) -> FeedOutcome:
    result = fetcher.fetch(feed)

    # `not_modified`, `failed`, `skipped_breaker` — nothing new to land, and
    # only the last two are problems.
    if not result.has_content:
        return FeedOutcome(feed, result.status, error=result.error)

    report = parse_feed(feed, result.body or b"", result.fetched_at)

    if feed.source not in key_cache:
        key_cache[feed.source] = existing_keys(feed.source, result.fetched_at, root)

    landing = land_articles(
        report.articles,
        feed_id=feed.feed_id,
        fetched_at=result.fetched_at,
        root=root,
        known_keys=key_cache[feed.source],
    )

    return FeedOutcome(
        feed=feed,
        status="ok",
        entries=report.total_entries,
        landed=landing.written,
        duplicates=landing.skipped_duplicates,
        parse_failures=report.failures,
    )
