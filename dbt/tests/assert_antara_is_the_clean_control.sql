-- ANTARA must remain the low-leakage control source.
--
-- The whole leakage experiment rests on one source whose URLs do not give the
-- answer away: train on ANTARA alone, train on everything, and the F1 gap
-- measures what leakage manufactures. If ANTARA ever starts leaking, that
-- experiment silently becomes meaningless — it would still run, still produce
-- a number, and the number would mean nothing.
--
-- Measured at the time of writing: ANTARA 4.1%, Liputan6 98.9%, CNN 100.0%.
-- The 4.1% is coincidental word matches (a slug that happens to contain
-- "metro"), not structure. 25% is far above that noise and far below the
-- structural leak, so crossing it means something changed.
--
-- Warn rather than error: this is a signal about the data, not a broken
-- pipeline, and stopping the build would not fix the publisher.

{{ config(severity='warn') }}

select
    source,
    count(*)                                                        as articles,
    round(100.0 * avg(case when url_leaks_label then 1 else 0 end), 1) as leak_pct
from {{ ref('fct_articles') }}
where source = 'antara'
group by 1
having avg(case when url_leaks_label then 1 else 0 end) > 0.25
