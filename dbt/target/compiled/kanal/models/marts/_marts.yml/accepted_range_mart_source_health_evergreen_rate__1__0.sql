

    select evergreen_rate
    from "kanal"."main"."mart_source_health"
    where evergreen_rate is not null
      and (
        false
        
            or evergreen_rate < 0
        
        
            or evergreen_rate > 1
        
      )

