

    select cluster_size
    from "kanal"."main"."fct_articles"
    where cluster_size is not null
      and (
        false
        
            or cluster_size < 1
        
        
      )

