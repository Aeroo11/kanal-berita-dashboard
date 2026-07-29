
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select file_path
from "kanal"."main"."_load_log"
where file_path is null



  
  
      
    ) dbt_internal_test