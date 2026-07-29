
    
    

select
    file_path as unique_field,
    count(*) as n_records

from "kanal"."main"."_load_log"
where file_path is not null
group by file_path
having count(*) > 1


