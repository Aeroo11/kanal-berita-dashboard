
    
    

select
    article_key as unique_field,
    count(*) as n_records

from "kanal"."main"."fct_articles"
where article_key is not null
group by article_key
having count(*) > 1


