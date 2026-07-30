# ADR-003: Dagster, and whether an orchestrator is warranted at all

**Status:** accepted · **Date:** 2026-07-30

## Context

At the point this was decided the pipeline already worked: a CLI, a GitHub Action
on a cron, and `dbt build`. Roughly 10⁵ rows. Nothing was broken.

So the first question was not "which orchestrator" but "why one".

## Decision

Dagster, capped at one code location and around fifteen assets. Timeboxed to 1.5
days, with a documented fallback to Prefect if it overran.

## Why an orchestrator at all

Stated plainly: **at this scale it is not needed.** A Makefile and two cron jobs
move the same bytes. Three things justify the complexity:

**1. Lineage as a rendered graph.** Every failure in this project so far has been
a question about where data came from — a publisher missing, a feed serving 2011,
a leakage rate measured against the wrong string. A lineage view answers that
class of question directly, and it is also the single artifact that most
efficiently communicates the work.

**2. Checks that live beside the asset they guard.** dbt already covers contracts
about *modelled* data. Contracts about *ingestion* had nowhere to live except a
CLI exit code — and that exit code was swallowed by a shell pipe for a full day
while an entire publisher went missing in silence. A check attached to an asset
cannot be lost that way.

**3. Partitions, which make backfill a click** and force the idempotency
discipline to be demonstrable rather than claimed.

## Why Dagster over Prefect

Prefect is lighter and easier to adopt, and models **tasks**. Dagster models
**assets** — the tables and files themselves — which is what makes lineage and
asset checks possible. Since lineage is the deliverable, that difference decides
it.

`dagster-dbt` also pulls every dbt model in as an asset and every dbt test as an
asset check, so one graph spans ingestion and transformation. Two separate DAGs —
one in a Makefile, one inside dbt — would each be correct, and neither could
answer "where did this number come from" without a human joining them by hand.

Airflow was not seriously considered: it needs a real deployment, and its
`execution_date` semantics are a well-known footgun.

## Consequences

**A dependency with real weight**, and one immediate trap: `from __future__
import annotations` breaks Dagster's context detection, because the annotation
becomes a string and the asset is rejected. Both orchestration modules omit it
with a comment.

**It introduced its own bug, of the worst kind.** dbt names its source
`kanal.raw_articles`, which becomes the asset key `kanal/raw_articles`, while the
Python asset writing that table is keyed `raw_articles`. Both nodes existed,
nothing connected them, and the lineage view showed two clusters that each looked
healthy. Every asset materialised. Every check passed. And the one capability the
orchestrator was added for stopped working, silently.

Fixed with a translator, and pinned by `scripts/check_lineage.py`, which walks
from the ingestion root in CI and requires everything reachable. Verified by
removing the translator and watching it fail.

**Schedules default to STOPPED.** The scheduled ingestion that actually runs is
the GitHub Action, which needs no always-on machine. The Dagster schedules exist
so `dagster dev` drives the same graph locally rather than requiring a second
definition of *when*.

**Explicitly not adopted:** Dagster Cloud, Kubernetes executors, custom IO
managers, sensors polling external systems.
