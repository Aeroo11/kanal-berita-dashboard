

    select title_words
    from "kanal"."main"."stg_articles"
    where title_words is not null
      and (
        false
        
            or title_words < 1
        
        
            or title_words > 60
        
      )

