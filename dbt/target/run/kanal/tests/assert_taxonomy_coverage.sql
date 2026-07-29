
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  -- Every (source, channel) that ingestion produces must exist in the taxonomy
-- seed.
--
-- This is the contract that catches a publisher adding a section. Without it,
-- a new channel would flow through with whatever label ingestion happened to
-- assign, and nobody would have reviewed the mapping. The failure mode is
-- silent and the damage is to the labels themselves, which is the worst place
-- for it.

select
    source,
    channel,
    count(*) as orphaned_articles
from "kanal"."main"."int_articles_labelled"
where taxonomy_missing
group by 1, 2
  
  
      
    ) dbt_internal_test