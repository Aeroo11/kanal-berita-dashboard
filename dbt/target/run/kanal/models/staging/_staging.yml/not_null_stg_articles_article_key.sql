
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select article_key
from "kanal"."main"."stg_articles"
where article_key is null



  
  
      
    ) dbt_internal_test