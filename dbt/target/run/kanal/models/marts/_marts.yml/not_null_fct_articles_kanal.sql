
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select kanal
from "kanal"."main"."fct_articles"
where kanal is null



  
  
      
    ) dbt_internal_test