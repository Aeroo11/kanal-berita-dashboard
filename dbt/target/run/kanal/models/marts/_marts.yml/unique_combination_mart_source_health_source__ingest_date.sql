
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  

    
    select source, ingest_date
    from "kanal"."main"."mart_source_health"
    group by source, ingest_date
    having count(*) > 1


  
  
      
    ) dbt_internal_test