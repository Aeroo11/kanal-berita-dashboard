{{ config(materialized='table') }}

-- Per-feed health, at feed granularity rather than per source.
--
-- `mart_source_health` answers "is this publisher responding?". It cannot answer
-- "is this *channel* still being published?", and those are different failures
-- with different consequences.
--
-- Discovered while surveying Republika before adding it: of 21 section feeds,
-- nine were stale by years and `/rss/kesehatan` was still serving articles from
-- 2011. It returns HTTP 200 with fifteen well-formed items, so every check that
-- existed at the time would have passed while the store filled with
-- fifteen-year-old content. It also maps *cleanly* onto gaya-hidup-kesehatan,
-- which is what made it dangerous rather than merely useless.
--
-- This is a distinct risk from ANTARA's evergreen mixing. There, a live feed
-- carries some old items. Here, the entire feed stopped being maintained and
-- nothing about the response says so.

with per_feed as (

    select
        source,
        channel,
        feed_id,
        kanal,

        count(*)                                                as articles,
        count(distinct cluster_id)                              as distinct_stories,

        min(published_at)                                       as oldest_published,
        max(published_at)                                       as newest_published,
        max(fetched_at)                                         as last_fetched,

        -- Age of the freshest article this feed has ever given us, measured at
        -- the time we fetched it. A maintained feed keeps this small; an
        -- abandoned one lets it grow without bound.
        min(date_diff('hour', published_at, fetched_at))         as freshest_age_hours,
        median(date_diff('hour', published_at, fetched_at))      as median_age_hours,

        avg(case when is_evergreen then 1.0 else 0.0 end)        as evergreen_rate,
        avg(case when url_leaks_label then 1.0 else 0.0 end)     as url_leak_rate,
        avg(case when missing_published_at then 1.0 else 0.0 end) as missing_timestamp_rate

    from {{ ref('fct_articles') }}
    group by 1, 2, 3, 4

)

select
    *,

    -- Nothing published in the last two days. Either the section genuinely went
    -- quiet, or the feed was abandoned and nobody told us.
    freshest_age_hours > 48                                     as looks_dormant,

    -- Its freshest item is over a month old: not quiet, abandoned.
    freshest_age_hours > 24 * 30                                 as looks_abandoned

from per_feed
