-- An article cannot be published after it was fetched.
--
-- Publishers do occasionally emit a timestamp with the wrong timezone, or a
-- scheduled post-dated item. Either way the row is unusable for a temporal
-- split: it would sort into the future and land in the test set regardless of
-- when it was actually written, which is the definition of lookahead.
--
-- A small clock-skew allowance, because a couple of minutes is a clock, not a
-- data problem.

select
    article_key,
    source,
    feed_id,
    published_at,
    fetched_at,
    date_diff('minute', fetched_at, published_at) as minutes_into_the_future
from "kanal"."main"."stg_articles"
where published_at > fetched_at + interval 5 minute