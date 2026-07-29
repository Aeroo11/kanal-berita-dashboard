
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  

    select cluster_size
    from "kanal"."main"."fct_articles"
    where cluster_size is not null
      and (
        false
        
            or cluster_size < 1
        
        
      )


  
  
      
    ) dbt_internal_test