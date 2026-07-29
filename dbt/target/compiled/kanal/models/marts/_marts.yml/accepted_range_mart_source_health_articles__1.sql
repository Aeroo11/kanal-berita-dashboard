

    select articles
    from "kanal"."main"."mart_source_health"
    where articles is not null
      and (
        false
        
            or articles < 1
        
        
      )

