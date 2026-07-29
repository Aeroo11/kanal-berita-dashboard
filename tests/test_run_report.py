"""The cycle report, and the health signal that depends on it.

This exists because of a real failure. On the first scheduled run, CNN returned
nothing at all — 0 of 7 feeds — while ANTARA and Liputan6 were fine. The cycle
correctly exited non-zero, and the run still went green, because the workflow
piped the command into `tee` and bash reports the exit status of the *last*
command in a pipeline.

So the guard was never load-bearing, and an entire publisher went missing in
silence. These tests pin the signal itself; `set -o pipefail` in the workflow
pins the plumbing that carries it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kanal.ingest.run import CycleReport, FeedOutcome
from kanal.ingest.sources import ALL_FEEDS, CNN, LIPUTAN6, feeds_for

START = datetime(2026, 7, 29, 8, 0, tzinfo=UTC)
END = datetime(2026, 7, 29, 8, 0, 27, tzinfo=UTC)


def report(outcomes: list[FeedOutcome]) -> CycleReport:
    return CycleReport(START, END, outcomes)


def all_ok() -> list[FeedOutcome]:
    return [FeedOutcome(f, "ok", entries=50, landed=50) for f in ALL_FEEDS]


class TestHealth:
    def test_a_full_cycle_is_healthy(self) -> None:
        assert report(all_ok()).healthy is True

    def test_not_modified_still_counts_as_a_usable_response(self) -> None:
        # A 304 means the feed genuinely had nothing new. That is a healthy
        # publisher, not a missing one.
        outcomes = [FeedOutcome(f, "not_modified") for f in ALL_FEEDS]
        assert report(outcomes).healthy is True

    def test_one_quiet_channel_does_not_make_a_cycle_unhealthy(self) -> None:
        outcomes = all_ok()
        outcomes[0] = FeedOutcome(outcomes[0].feed, "failed", error="HTTP 500")
        # Its publisher still answered on other feeds.
        assert report(outcomes).healthy is True

    def test_an_entire_publisher_going_dark_is_unhealthy(self) -> None:
        # The exact shape of the first real run: CNN gone, the rest fine.
        outcomes = [
            FeedOutcome(f, "failed", error="HTTP 403")
            if f.source == CNN.name
            else FeedOutcome(f, "ok", entries=50, landed=50)
            for f in ALL_FEEDS
        ]
        rep = report(outcomes)
        assert rep.healthy is False
        assert CNN.name not in rep.sources_seen

    def test_a_tripped_breaker_also_counts_as_missing(self) -> None:
        # Skipping a source because its breaker is open is still an hour of data
        # not collected — the reason does not change the consequence.
        outcomes = [
            FeedOutcome(f, "skipped_breaker", error="cooldown 3400s")
            if f.source == LIPUTAN6.name
            else FeedOutcome(f, "ok", landed=10)
            for f in ALL_FEEDS
        ]
        assert report(outcomes).healthy is False


class TestPerSourceTally:
    def test_counts_feeds_and_rows_per_publisher(self) -> None:
        outcomes = all_ok()
        tally = report(outcomes).by_source()
        assert tally[CNN.name]["feeds"] == len(feeds_for(CNN.name))
        assert tally[CNN.name]["failed"] == 0
        assert tally[CNN.name]["landed"] == 50 * len(feeds_for(CNN.name))

    def test_separates_ok_from_failed_within_a_source(self) -> None:
        cnn_feeds = feeds_for(CNN.name)
        outcomes = [FeedOutcome(cnn_feeds[0], "ok", landed=5)] + [
            FeedOutcome(f, "failed", error="HTTP 403") for f in cnn_feeds[1:]
        ]
        tally = report(outcomes).by_source()[CNN.name]
        assert tally["ok"] == 1
        assert tally["failed"] == len(cnn_feeds) - 1


class TestSummary:
    def test_reports_the_missing_publisher_by_name(self) -> None:
        outcomes = [
            FeedOutcome(f, "failed", error="HTTP 403")
            if f.source == CNN.name
            else FeedOutcome(f, "ok", landed=50)
            for f in ALL_FEEDS
        ]
        text = report(outcomes).summary()
        assert "NO usable response" in text
        assert CNN.name in text
        # The consequence is stated, not left to be inferred.
        assert "cannot be backfilled" in text

    def test_groups_identical_failures_into_one_reason(self) -> None:
        # Seven feeds failing for one reason is one problem, and the reason is
        # the only actionable part.
        cnn_feeds = feeds_for(CNN.name)
        outcomes = [FeedOutcome(f, "failed", error="HTTP 403") for f in cnn_feeds] + [
            FeedOutcome(f, "ok", landed=10) for f in ALL_FEEDS if f.source != CNN.name
        ]
        text = report(outcomes).summary()
        assert text.count("HTTP 403") == 1
        assert "affected:" in text

    def test_a_healthy_cycle_says_nothing_alarming(self) -> None:
        text = report(all_ok()).summary()
        assert "NO usable response" not in text
        assert "failures:" not in text

    def test_always_shows_a_per_source_breakdown(self) -> None:
        text = report(all_ok()).summary()
        # Whether healthy or not — the breakdown is how a partial loss becomes
        # visible at a glance instead of hiding inside a total.
        for source in ("antara", "cnn", "liputan6"):
            assert source in text
