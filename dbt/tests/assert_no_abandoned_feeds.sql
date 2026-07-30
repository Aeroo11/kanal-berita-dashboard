-- No configured feed should be serving content that is over a month old.
--
-- This contract exists because of a near miss. Republika's `/rss/kesehatan`
-- returns HTTP 200 with fifteen well-formed, correctly-dated items — and the
-- newest is from 2011. Nine of its twenty-one section feeds are stale by years.
--
-- Nothing about such a response is detectably wrong: the fetch succeeds, the
-- parse succeeds, the schema validates, the freshness check on the *source*
-- passes because other feeds from the same publisher are healthy. The only
-- symptom is that the articles are ancient, and `/rss/kesehatan` maps cleanly
-- onto gaya-hidup-kesehatan, so it would have been added without a second
-- thought and quietly filled the store with fifteen-year-old text.
--
-- Warn rather than error: a section going quiet is the publisher's business,
-- and failing the build would not restart their editorial calendar. But it must
-- be *visible*, because the alternative is discovering it in a confusion matrix
-- six weeks later.

{{ config(severity='warn') }}

select
    feed_id,
    kanal,
    articles,
    round(freshest_age_hours / 24.0, 1) as freshest_age_days,
    newest_published
from {{ ref('mart_feed_health') }}
where looks_abandoned
order by freshest_age_hours desc
