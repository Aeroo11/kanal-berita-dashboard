
    
    

with all_values as (

    select
        kanal as value_field,
        count(*) as n_records

    from "kanal"."main"."stg_articles"
    group by kanal

)

select *
from all_values
where value_field not in (
    'politik','ekonomi','olahraga','teknologi','hiburan','internasional','hukum-kriminal','gaya-hidup-kesehatan'
)


