"""The code location: assets, checks, jobs and schedules.

One code location on purpose. Splitting a project this size across several buys
nothing and costs the single lineage view that is the reason for using Dagster
at all.
"""

from __future__ import annotations

from dagster import (
    AssetSelection,
    DefaultScheduleStatus,
    Definitions,
    ScheduleDefinition,
    define_asset_job,
    load_assets_from_modules,
)

from kanal.orchestration import assets_dbt, assets_ingest
from kanal.orchestration.assets_dbt import dbt_resource

ingest_assets = load_assets_from_modules([assets_ingest])
dbt_asset_defs = load_assets_from_modules([assets_dbt])

# ── Jobs ─────────────────────────────────────────────────────────────────
#
# Two jobs rather than one, because the two halves fail for different reasons and
# on different schedules. Ingestion is hourly and time-critical: RSS is a sliding
# window, so a missed hour is gone. Transformation reads what has already landed
# and can be re-run at any time without loss.

ingest_job = define_asset_job(
    name="ingest_job",
    selection=AssetSelection.assets("rss_landing_zone", "raw_articles"),
    description=(
        "Poll every feed and load what is new. Hourly, because a feed is a "
        "sliding window of its last 25-100 items and an hour not captured cannot "
        "be recovered."
    ),
)

transform_job = define_asset_job(
    name="transform_job",
    selection=AssetSelection.all() - AssetSelection.assets("rss_landing_zone"),
    description=(
        "Rebuild the warehouse and every dbt model, then run the contracts. Safe "
        "to re-run: it reads what has already landed."
    ),
)

# ── Schedules ────────────────────────────────────────────────────────────
#
# Stopped by default. The scheduled ingestion that actually runs is the GitHub
# Action, which needs no always-on machine; these exist so `dagster dev` can
# drive the same graph locally without a second definition of *when*.

ingest_schedule = ScheduleDefinition(
    name="hourly_ingest",
    job=ingest_job,
    cron_schedule="7 * * * *",
    default_status=DefaultScheduleStatus.STOPPED,
)

transform_schedule = ScheduleDefinition(
    name="nightly_transform",
    job=transform_job,
    cron_schedule="30 2 * * *",
    default_status=DefaultScheduleStatus.STOPPED,
)

defs = Definitions(
    assets=[*ingest_assets, *dbt_asset_defs],
    jobs=[ingest_job, transform_job],
    schedules=[ingest_schedule, transform_schedule],
    resources={"dbt": dbt_resource},
)
