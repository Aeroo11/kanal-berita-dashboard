"""Ingestion and warehouse assets.

Each asset wraps code that already worked standalone. That is deliberate: the CLI
remains the thing that runs in the scheduled workflow, and Dagster orchestrates
the same functions rather than reimplementing them. An orchestrator that owns
logic nothing else can call is an orchestrator you cannot debug without it.
"""

# No `from __future__ import annotations` here either — see assets_dbt.py.
# Dagster resolves the `context` annotation at decoration time and cannot read it
# once the future import has turned it into a string.

from dagster import (
    AssetCheckResult,
    AssetExecutionContext,
    AssetKey,
    MetadataValue,
    Output,
    asset,
    asset_check,
)

from kanal.config import settings
from kanal.ingest.land import count_articles
from kanal.ingest.run import run_cycle
from kanal.ingest.sources import ALL_FEEDS, SOURCE_BY_NAME
from kanal.orchestration.partitions import daily_partitions
from kanal.warehouse.duck import reader
from kanal.warehouse.loader import default_db_path, load


@asset(
    name="rss_landing_zone",
    group_name="ingestion",
    partitions_def=daily_partitions,
    description=(
        "Raw articles polled from every configured feed, written as partitioned "
        "JSONL. Idempotent: re-materialising a partition adds nothing, because "
        "the cycle reads back which article keys the recent partitions already "
        "hold before writing."
    ),
)
def rss_landing_zone(context: AssetExecutionContext) -> Output[int]:
    """Poll every feed once and land what is new.

    Note what this asset *cannot* do: materialise a past partition. RSS has no
    archive endpoint, so re-running yesterday polls today's feed contents. The
    partition exists to record which day the data belongs to and to make gaps
    visible — not to promise that a gap can be refilled.
    """
    report = run_cycle(ALL_FEEDS)

    context.log.info("\n%s", report.summary())

    if not report.healthy:
        # Surfaced as a failed *check* rather than a raised exception: the
        # articles that did land are real and must still flow downstream.
        context.log.warning(
            "no usable response from: %s", ", ".join(sorted(report.missing_sources))
        )

    return Output(
        report.landed,
        metadata={
            "articles_landed": report.landed,
            "already_seen": report.duplicates,
            "feeds_ok": sum(1 for o in report.outcomes if o.status in ("ok", "not_modified")),
            "feeds_failed": len(report.failed_feeds),
            "sources_reporting": MetadataValue.text(", ".join(sorted(report.sources_seen))),
            "missing_sources": MetadataValue.text(
                ", ".join(sorted(report.missing_sources)) or "none"
            ),
            "landing_zone_total": count_articles(),
            "cycle_report": MetadataValue.md(f"```\n{report.summary()}\n```"),
        },
    )


@asset(
    name="raw_articles",
    group_name="warehouse",
    deps=[AssetKey("rss_landing_zone")],
    description=(
        "The landing zone loaded into DuckDB. The single writer in the system; "
        "everything downstream reads. Incremental — files already recorded in "
        "_load_log are not reopened — and idempotent, via an anti-join on "
        "article_key."
    ),
)
def raw_articles(context: AssetExecutionContext) -> Output[int]:
    report = load()
    context.log.info(report.summary())

    with reader(default_db_path()) as conn:
        total = int(conn.execute("SELECT count(*) FROM raw_articles").fetchone()[0])  # type: ignore[index]

    return Output(
        total,
        metadata={
            "rows_in_warehouse": total,
            "files_loaded_this_run": report.files_loaded,
            "rows_added": report.rows_added,
            "rows_already_present": report.rows_skipped,
        },
    )


# ── Asset checks: the ingestion contracts ────────────────────────────────
#
# dbt already covers the contracts about *modelled* data. These cover ingestion,
# which previously had nowhere to express them except a CLI exit code — and that
# exit code was swallowed by a pipe for a day while a whole publisher went
# missing. A check that lives beside the asset cannot be lost that way.


@asset_check(
    asset=rss_landing_zone,
    name="every_expected_source_reported",
    description=(
        "Each publisher this environment expects to reach produced at least one "
        "usable response. Per source, not per feed: a quiet channel is normal, a "
        "dark publisher is not."
    ),
    blocking=False,
)
def check_every_expected_source_reported() -> AssetCheckResult:
    with reader(default_db_path()) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT source FROM raw_articles
            WHERE fetched_at > now() - INTERVAL 2 HOUR
            """
        ).fetchall()

    seen = {str(r[0]) for r in rows}
    expected = {f.source for f in ALL_FEEDS} - settings.unreachable_sources
    missing = expected - seen

    return AssetCheckResult(
        passed=not missing,
        metadata={
            "expected": MetadataValue.text(", ".join(sorted(expected))),
            "reported_in_last_2h": MetadataValue.text(", ".join(sorted(seen)) or "none"),
            "missing": MetadataValue.text(", ".join(sorted(missing)) or "none"),
            "declared_unreachable": MetadataValue.text(
                ", ".join(sorted(settings.unreachable_sources)) or "none"
            ),
        },
    )


@asset_check(
    asset=raw_articles,
    name="article_keys_are_unique",
    description=(
        "One row per article. The anti-join in the loader is what guarantees "
        "this; the check is what would notice if it ever stopped."
    ),
    blocking=True,
)
def check_article_keys_are_unique() -> AssetCheckResult:
    with reader(default_db_path()) as conn:
        total, distinct = conn.execute(
            "SELECT count(*), count(DISTINCT article_key) FROM raw_articles"
        ).fetchone()  # type: ignore[misc]

    return AssetCheckResult(
        passed=total == distinct,
        metadata={"rows": int(total), "distinct_keys": int(distinct)},
    )


@asset_check(
    asset=raw_articles,
    name="no_feed_has_gone_silent",
    description=(
        "Every configured feed has produced something within the retention "
        "window. A feed that answers HTTP 200 with well-formed but ancient items "
        "is invisible to every other check — see Republika's /rss/kesehatan, "
        "which was still serving 2011."
    ),
    blocking=False,
)
def check_no_feed_has_gone_silent() -> AssetCheckResult:
    with reader(default_db_path()) as conn:
        rows = conn.execute(
            """
            SELECT feed_id,
                   count(*)                                          AS n,
                   min(date_diff('hour', published_at, fetched_at))   AS freshest_age_hours
            FROM raw_articles
            GROUP BY 1
            """
        ).fetchall()

    present = {str(r[0]): (int(r[1]), r[2]) for r in rows}
    configured = {f.feed_id for f in ALL_FEEDS}

    absent = sorted(configured - set(present))
    # Freshest item over a month old: the feed is not quiet, it is abandoned.
    abandoned = sorted(
        feed for feed, (_, age) in present.items() if age is not None and float(age) > 24 * 30
    )

    return AssetCheckResult(
        passed=not absent and not abandoned,
        metadata={
            "configured_feeds": len(configured),
            "feeds_with_data": len(present),
            "absent": MetadataValue.text(", ".join(absent) or "none"),
            "abandoned": MetadataValue.text(", ".join(abandoned) or "none"),
        },
    )


@asset_check(
    asset=raw_articles,
    name="at_least_one_low_leakage_source",
    description=(
        "At least one publisher whose article URLs do not give the label away. "
        "The leakage experiment — train on clean sources, train on everything, "
        "and measure the F1 gap — has no control group without one, and would "
        "still run and still produce a meaningless number."
    ),
    blocking=False,
)
def check_at_least_one_low_leakage_source() -> AssetCheckResult:
    with reader(default_db_path()) as conn:
        rows = conn.execute(
            """
            SELECT source,
                   count(*)                                                       AS n,
                   avg(CASE WHEN contains(lower(raw_link), lower(channel))
                                 OR contains(lower(raw_link), split_part(kanal, '-', 1))
                            THEN 1.0 ELSE 0.0 END)                                AS leak_rate
            FROM raw_articles
            GROUP BY 1
            """
        ).fetchall()

    rates = {str(r[0]): float(r[2]) for r in rows if int(r[1]) > 0}
    clean = sorted(s for s, rate in rates.items() if rate < 0.25)

    return AssetCheckResult(
        passed=bool(clean),
        metadata={
            "low_leakage_sources": MetadataValue.text(", ".join(clean) or "NONE"),
            "leak_rate_by_source": MetadataValue.json(
                {s: round(rate, 3) for s, rate in sorted(rates.items())}
            ),
            "threshold": 0.25,
        },
    )


@asset_check(
    asset=raw_articles,
    name="taxonomy_covers_every_channel",
    description=(
        "Every (source, channel) ingestion produces is known to the registry. A "
        "publisher adding a section would otherwise flow through carrying a label "
        "nobody reviewed, and the damage would be to the labels themselves."
    ),
    blocking=True,
)
def check_taxonomy_covers_every_channel() -> AssetCheckResult:
    with reader(default_db_path()) as conn:
        rows = conn.execute("SELECT DISTINCT source, channel FROM raw_articles").fetchall()

    known = {(f.source, f.channel) for f in ALL_FEEDS}
    unknown = sorted(f"{r[0]}:{r[1]}" for r in rows if (str(r[0]), str(r[1])) not in known)

    return AssetCheckResult(
        passed=not unknown,
        metadata={
            "channels_in_data": len(rows),
            "channels_in_registry": len(known),
            "unknown": MetadataValue.text(", ".join(unknown) or "none"),
        },
    )


@asset_check(
    asset=raw_articles,
    name="no_future_publication_dates",
    description=(
        "No article claims to have been published after it was fetched. Such a "
        "row sorts into the future and lands in the test set regardless of when "
        "it was written — lookahead by definition."
    ),
    blocking=False,
)
def check_no_future_publication_dates() -> AssetCheckResult:
    with reader(default_db_path()) as conn:
        offenders = int(
            conn.execute(
                """
                SELECT count(*) FROM raw_articles
                WHERE published_at > fetched_at + INTERVAL 5 MINUTE
                """
            ).fetchone()[0]  # type: ignore[index]
        )

    return AssetCheckResult(
        passed=offenders == 0,
        metadata={"rows_dated_in_the_future": offenders, "clock_skew_allowance": "5 minutes"},
    )


@asset_check(
    asset=raw_articles,
    name="sources_are_all_registered",
    description="No row carries a source that is not in the registry.",
    blocking=True,
)
def check_sources_are_all_registered() -> AssetCheckResult:
    with reader(default_db_path()) as conn:
        rows = conn.execute("SELECT DISTINCT source FROM raw_articles").fetchall()

    unknown = sorted(str(r[0]) for r in rows if str(r[0]) not in SOURCE_BY_NAME)
    return AssetCheckResult(
        passed=not unknown,
        metadata={"unknown_sources": MetadataValue.text(", ".join(unknown) or "none")},
    )
