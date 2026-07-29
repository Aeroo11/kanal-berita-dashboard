
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select source
from "kanal"."main"."fct_articles"
where source is null



  
  
      
    ) dbt_internal_test