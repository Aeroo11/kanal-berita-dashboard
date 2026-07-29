{{ config(materialized='view') }}

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

    select * from {{ source('kanal', 'raw_articles') }}

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

        -- Does this row's own URL give its label away?
        --
        -- Two forms, and both count. A URL may carry the *canonical* label
        -- ('ekonomi'), or the publisher's own *channel* name ('bisnis') — and
        -- channel maps to label deterministically, so either one is a complete
        -- giveaway.
        --
        -- The first version of this column only checked the canonical form and
        -- badly understated the problem: Liputan6 measured 1.4% when the real
        -- figure is 98.9%, because its paths read /bisnis/read/... while its
        -- label is 'ekonomi'. Corrected, the finding is sharper than the
        -- original claim — it is not that CNN leaks, it is that ANTARA is the
        -- only source that does not.
        contains(lower(raw_link), lower(channel))               as url_leaks_channel,

        contains(lower(raw_link), split_part(kanal, '-', 1))    as url_leaks_canonical,

        (
            contains(lower(raw_link), lower(channel))
            or contains(lower(raw_link), split_part(kanal, '-', 1))
        )                                                       as url_leaks_label

    from typed

)

select * from derived
