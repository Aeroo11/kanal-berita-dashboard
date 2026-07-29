
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

select
    article_key as unique_field,
    count(*) as n_records

from "kanal"."main"."raw_articles"
where article_key is not null
group by article_key
having count(*) > 1



  
  
      
    ) dbt_internal_test