-- Every configured publisher must have produced articles on the most recent
-- ingest day.
--
-- This is the warehouse-level counterpart to the exit code in `kanal ingest`,
-- and it exists because that exit code was not enough. On the first scheduled
-- run CNN returned nothing, the cycle exited non-zero exactly as designed, and
-- the workflow reported success anyway because the command was piped into
-- `tee` — bash reports the status of the last command in a pipeline.
--
-- The pipe is fixed. This test is the second line of defence: a per-run signal
-- can only be missed once per run, whereas a missing publisher is visible here
-- on every build until someone deals with it.
--
-- Warn rather than error: a publisher being blocked is a decision to make, not
-- a reason to stop the pipeline from processing the sources that *are*
-- working. Losing the good data too would repeat the original mistake.

{{ config(severity='warn') }}

with expected as (
    select unnest(['antara', 'cnn', 'liputan6']) as source
),

latest_day as (
    select max(ingest_date) as ingest_date from {{ ref('mart_source_health') }}
),

reported as (
    select h.source
    from {{ ref('mart_source_health') }} h
    join latest_day d on d.ingest_date = h.ingest_date
)

select
    e.source as silent_source,
    (select ingest_date from latest_day) as on_date
from expected e
left join reported r on r.source = e.source
where r.source is null
