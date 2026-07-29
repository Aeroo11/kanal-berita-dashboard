
  
  create view "kanal"."main"."int_articles_labelled__dbt_tmp" as (
    

-- Joins the taxonomy seed onto every article.
--
-- The label already exists on the row — ingestion set it from the feed. This
-- model exists so the *mapping* is a version-controlled artefact rather than a
-- constant buried in Python: `taxonomy_map.csv` carries a `confidence` and a
-- `notes` column for every channel, so a judgement call like "ANTARA's metro
-- feed counts as hukum-kriminal" is written down where it can be argued with.
--
-- The join is also a contract. If ingestion starts emitting a channel the seed
-- has never heard of — a publisher adds a section — the label comes back null
-- and `taxonomy_coverage` fails, instead of the article quietly carrying a
-- label nobody reviewed.

with articles as (

    select * from "kanal"."main"."int_articles_deduped"

),

taxonomy as (

    select
        source,
        channel,
        kanal        as mapped_kanal,
        confidence   as mapping_confidence,
        notes        as mapping_notes
    from "kanal"."main"."taxonomy_map"

)

select
    a.*,

    t.mapped_kanal,
    t.mapping_confidence,
    t.mapping_notes,

    -- Ingestion and the seed should always agree; they are two expressions of
    -- one decision. Disagreement means one of them was changed without the
    -- other, which is a bug worth failing on rather than silently preferring
    -- either side.
    t.mapped_kanal is null                       as taxonomy_missing,
    coalesce(t.mapped_kanal != a.kanal, false)   as taxonomy_conflict,

    -- Rows whose mapping was a judgement call. Excluded from nothing, but
    -- available to weight or audit — these are where label noise concentrates.
    coalesce(t.mapping_confidence != 'high', false) as label_is_judgement_call

from articles a
left join taxonomy t
    on  t.source  = a.source
    and t.channel = a.channel
  );
