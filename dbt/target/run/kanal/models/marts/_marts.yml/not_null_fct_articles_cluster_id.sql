
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select cluster_id
from "kanal"."main"."fct_articles"
where cluster_id is null



  
  
      
    ) dbt_internal_test