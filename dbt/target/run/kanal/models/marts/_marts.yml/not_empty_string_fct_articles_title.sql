
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  

    -- NOT NULL does not catch '' or '   ', and an empty headline is the one
    -- field the model genuinely cannot work without.
    select title
    from "kanal"."main"."fct_articles"
    where title is not null
      and trim(title) = ''


  
  
      
    ) dbt_internal_test