"""Partition definitions.

Days, in UTC, matching the landing zone's `dt=` partition exactly. Anything else
would mean an asset partition covering part of two files, which makes "re-run
2026-07-14" ambiguous — and an ambiguous backfill is worse than none.

The start date is the first day real ingestion ran. Earlier partitions cannot be
materialised at all: RSS is a sliding window with no archive endpoint, so there
is no way to fetch a day that has passed. That is a hard property of the source,
and declaring it here means Dagster's UI shows an honest partition range rather
than offering backfills that could only ever produce empty results.
"""

from __future__ import annotations

from dagster import DailyPartitionsDefinition

# The first day the hourly workflow landed anything.
INGESTION_START = "2026-07-29"

daily_partitions = DailyPartitionsDefinition(
    start_date=INGESTION_START,
    timezone="UTC",
)
