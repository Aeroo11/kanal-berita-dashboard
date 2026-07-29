{{ config(materialized='table') }}

-- The fact table. One row per article, and the only model anything downstream
-- should read — training, the API, the dashboard.
--
-- Materialised rather than a view because every consumer would otherwise
-- re-run the dedup window function. One pass here, many cheap reads after.
--
-- Deliberately still one row per article, not per cluster: dropping duplicates
-- at this layer would decide for every consumer what counts as a duplicate.
-- The flags are carried instead, so the split can partition by cluster while
-- an audit can still see every copy.

select
    -- ── identity ─────────────────────────────────────────────────────────
    article_key,
    cluster_id,

    -- ── the only fields a model may see ──────────────────────────────────
    title,
    summary,

    -- ── label ────────────────────────────────────────────────────────────
    kanal,
    label_is_judgement_call,
    has_label_disagreement,

    -- ── provenance: for auditing, never for features ─────────────────────
    source,
    channel,
    feed_id,
    canonical_url,

    -- ── time ─────────────────────────────────────────────────────────────
    published_at,
    published_date,
    fetched_at,
    age_hours_at_fetch,
    missing_published_at,

    -- ── properties the split and the evaluation need ─────────────────────
    is_evergreen,
    is_cross_source_duplicate,
    is_cluster_representative,
    cluster_size,
    url_leaks_label,

    -- ── shape ────────────────────────────────────────────────────────────
    title_chars,
    title_words,
    summary_chars,

    _ingest_file

from {{ ref('int_articles_labelled') }}
