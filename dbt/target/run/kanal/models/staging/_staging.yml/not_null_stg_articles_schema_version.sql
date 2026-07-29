
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select schema_version
from "kanal"."main"."stg_articles"
where schema_version is null



  
  
      
    ) dbt_internal_test