
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  

    select articles
    from "kanal"."main"."mart_source_health"
    where articles is not null
      and (
        false
        
            or articles < 1
        
        
      )


  
  
      
    ) dbt_internal_test