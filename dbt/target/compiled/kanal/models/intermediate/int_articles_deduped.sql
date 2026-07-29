

-- Cross-source duplicate detection.
--
-- ANTARA is a wire agency: its stories are republished near-verbatim by other
-- outlets hours later. Those are *different articles* by URL — different
-- `article_key`, correctly so — but the same story. If a random split puts one
-- copy in train and another in test, the evaluation measures memorisation and
-- reports it as generalisation.
--
-- This layer assigns every article a `cluster_id`, and the split downstream
-- must partition by cluster, never by row.
--
-- The clustering here is the cheap exact-match layer: a normalised title
-- fingerprint, computed at ingest. It catches republication that preserves the
-- headline, which is the common case. It does *not* catch a rewritten headline
-- about the same event — that needs MinHash/LSH, which belongs in the Python
-- layer where the shingling is expressible. This model is deliberately the
-- floor, not the ceiling, and `is_exact_duplicate_group` marks what it found so
-- the gap between the two is measurable rather than assumed away.

with articles as (

    select * from "kanal"."main"."stg_articles"

),

clusters as (

    select
        title_fingerprint                                   as cluster_id,
        count(*)                                            as cluster_size,
        count(distinct source)                              as cluster_sources,
        count(distinct kanal)                               as cluster_labels,
        min(published_at)                                   as cluster_first_seen,
        -- The earliest publisher of a story is usually the originator; the
        -- others are syndicating it.
        arg_min(source, published_at)                       as cluster_origin_source
    from articles
    group by 1

)

select
    a.*,

    c.cluster_id,
    c.cluster_size,
    c.cluster_sources,
    c.cluster_origin_source,
    c.cluster_first_seen,

    c.cluster_size > 1                                      as is_exact_duplicate_group,

    -- Republished across publishers, rather than repeated within one.
    c.cluster_sources > 1                                   as is_cross_source_duplicate,

    -- The same story filed under different sections by different publishers.
    -- This is the label-noise signal: where editors disagree, no model can be
    -- right for everyone, and the disagreement rate is an empirical ceiling on
    -- achievable accuracy.
    c.cluster_labels > 1                                    as has_label_disagreement,

    -- One representative per cluster, chosen deterministically so the choice
    -- does not wander between builds.
    row_number() over (
        partition by c.cluster_id
        order by a.published_at, a.article_key
    ) = 1                                                   as is_cluster_representative

from articles a
join clusters c on c.cluster_id = a.title_fingerprint