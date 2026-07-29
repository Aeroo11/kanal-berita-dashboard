{{ config(materialized='table') }}

-- Per-source, per-day health of the data supply.
--
-- Built after CNN vanished from a scheduled run and the pipeline reported
-- success anyway. The workflow bug that hid it is fixed, but a per-run exit
-- code only tells you about the run in front of you. This table is the record:
-- it makes "CNN has produced nothing since Tuesday" a query rather than an
-- archaeology exercise across workflow logs that expire.
--
-- It also carries the two measured properties that shape the modelling —
-- leakage rate and evergreen share — per source and per day, so drift in
-- either is visible rather than assumed constant.

with daily as (

    select
        source,
        cast(fetched_at as date)                                   as ingest_date,

        count(*)                                                    as articles,
        count(distinct feed_id)                                     as feeds_seen,
        count(distinct kanal)                                       as labels_seen,
        count(distinct cluster_id)                                  as distinct_stories,

        -- Data supply
        min(fetched_at)                                             as first_fetch,
        max(fetched_at)                                             as last_fetch,
        count(distinct date_trunc('hour', fetched_at))              as hours_with_data,

        -- Measured properties, not assumptions
        avg(case when url_leaks_label then 1.0 else 0.0 end)        as url_leak_rate,
        avg(case when is_evergreen then 1.0 else 0.0 end)           as evergreen_rate,
        avg(case when is_cross_source_duplicate then 1.0 else 0.0 end) as cross_source_dup_rate,
        avg(case when missing_published_at then 1.0 else 0.0 end)   as missing_timestamp_rate,
        avg(case when label_is_judgement_call then 1.0 else 0.0 end) as judgement_call_rate,

        -- Shape
        round(avg(title_words), 2)                                  as mean_title_words,
        round(avg(summary_chars), 1)                                as mean_summary_chars

    from {{ ref('fct_articles') }}
    group by 1, 2

)

select
    *,

    -- Ingestion runs hourly. Fewer than 20 hours covered on a completed day
    -- means cycles were missed, and missed cycles cannot be backfilled.
    hours_with_data                                                 as _hours_covered,
    hours_with_data < 20                                            as has_ingestion_gap

from daily
