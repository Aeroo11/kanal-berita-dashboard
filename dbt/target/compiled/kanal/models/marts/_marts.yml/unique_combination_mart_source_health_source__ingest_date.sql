

    
    select source, ingest_date
    from "kanal"."main"."mart_source_health"
    group by source, ingest_date
    having count(*) > 1

