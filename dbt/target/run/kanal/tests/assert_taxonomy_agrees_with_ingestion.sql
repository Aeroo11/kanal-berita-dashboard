
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  -- The label ingestion assigned and the label the seed maps to must agree.
--
-- They are two expressions of one decision: `sources.py` holds it in Python so
-- the fetcher knows what it is collecting, `taxonomy_map.csv` holds it in the
-- warehouse so the mapping is reviewable data rather than a buried constant.
--
-- Two copies of one truth will drift. This test is the thing that makes the
-- drift loud instead of leaving the pipeline quietly labelling articles one way
-- and documenting them another.

select
    source,
    channel,
    kanal        as label_from_ingestion,
    mapped_kanal as label_from_seed,
    count(*)     as affected_articles
from "kanal"."main"."int_articles_labelled"
where taxonomy_conflict
group by 1, 2, 3, 4
  
  
      
    ) dbt_internal_test