"""Dagster orchestration.

Why an orchestrator at this scale, stated plainly: at ~10^5 rows you do not need
one. A Makefile and two cron jobs would move the same bytes, and for a while
that is exactly what this project used.

The complexity buys three things a Makefile cannot:

1.  **Asset lineage as a graph.** feeds → landing zone → warehouse → dbt models
    → marts, rendered and clickable. Every failure so far in this project was a
    question about where data came from, and a lineage view answers that class
    of question directly.

2.  **Asset checks that live next to the asset.** The data contracts already
    exist in dbt; the ones about *ingestion* had nowhere to live except a CLI
    exit code, which is how a whole publisher went missing in silence for a day.

3.  **Partitions, which give backfill for free.** Days are the natural partition
    and re-running one is a click. That also forces the idempotency discipline
    the landing zone already has, and makes it demonstrable rather than claimed.

Deliberately not used: Dagster Cloud, Kubernetes executors, custom IO managers,
sensors that poll external systems. `dagster dev` locally, `dagster job execute`
in CI.
"""
