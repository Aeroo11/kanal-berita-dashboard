

    select url_leak_rate
    from "kanal"."main"."mart_source_health"
    where url_leak_rate is not null
      and (
        false
        
            or url_leak_rate < 0
        
        
            or url_leak_rate > 1
        
      )

