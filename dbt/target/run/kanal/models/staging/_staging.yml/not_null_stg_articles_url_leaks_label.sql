
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select url_leaks_label
from "kanal"."main"."stg_articles"
where url_leaks_label is null



  
  
      
    ) dbt_internal_test