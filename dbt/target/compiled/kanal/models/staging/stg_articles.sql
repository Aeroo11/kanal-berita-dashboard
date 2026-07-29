

-- Typed, trimmed view over the raw landing table.
--
-- This layer does three things and nothing else: cast, derive, and mark. It
-- does not filter and it does not drop rows. Anything excluded here would be
-- excluded invisibly, and the whole point of a staging layer is that what
-- happened to a row is answerable by reading one file.
--
-- Note `is_evergreen`. ANTARA mixes explainers and profiles into its news
-- feeds, and 64% of its items are more than 30 days old against CNN's 0%. A
-- naive temporal split would therefore put almost all ANTARA in train and leave
-- a test set dominated by CNN — which is also the leakiest source. Marking it
-- here means the split logic downstream can see it instead of being surprised.

with raw as (

    select * from "kanal"."main"."raw_articles"

),

typed as (

    select
        article_key,
        canonical_url,
        title_fingerprint,

        -- The only two fields a model may ever see.
        trim(title)                                    as title,
        coalesce(trim(summary), '')                    as summary,

        kanal,
        source,
        channel,
        feed_id,

        -- Provenance. Kept for auditing, forbidden as features: CNN puts the
        -- section in this URL 86% of the time.
        raw_link,

        published_at,
        fetched_at,
        schema_version,
        extra,
        _ingest_file,
        _loaded_at

    from raw

),

derived as (

    select
        *,

        length(title)                                          as title_chars,
        array_length(string_split(trim(title), ' '))            as title_words,
        length(summary)                                         as summary_chars,

        -- Age at ingest, not age now: a fixed property of the row rather than
        -- one that silently changes every time a query runs.
        date_diff('hour', published_at, fetched_at)             as age_hours_at_fetch,

        published_at is null                                    as missing_published_at,

        -- Older than 30 days when we saw it. See the note above.
        coalesce(
            date_diff('day', published_at, fetched_at) > 30,
            false
        )                                                       as is_evergreen,

        cast(published_at as date)                              as published_date,

        -- Does this row's own URL contain its label? Materialising the leak as
        -- a column is what turns "leakage is a risk" into a number anyone can
        -- query, and lets the ANTARA-only vs all-sources experiment be defined
        -- in SQL rather than described in prose.
        contains(
            lower(raw_link),
            split_part(kanal, '-', 1)
        )                                                       as url_leaks_label

    from typed

)

select * from derived