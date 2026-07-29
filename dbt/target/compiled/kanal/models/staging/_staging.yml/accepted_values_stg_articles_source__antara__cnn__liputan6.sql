
    
    

with all_values as (

    select
        source as value_field,
        count(*) as n_records

    from "kanal"."main"."stg_articles"
    group by source

)

select *
from all_values
where value_field not in (
    'antara','cnn','liputan6'
)


