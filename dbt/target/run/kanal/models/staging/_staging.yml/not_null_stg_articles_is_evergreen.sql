
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select is_evergreen
from "kanal"."main"."stg_articles"
where is_evergreen is null



  
  
      
    ) dbt_internal_test