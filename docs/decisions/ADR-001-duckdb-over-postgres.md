# ADR-001: DuckDB as the warehouse, not Postgres

**Status:** accepted · **Date:** 2026-07-29

## Context

The project needs an analytical store for a fact table of news articles, queried
by dbt and by Python. Expected size after a year of hourly ingestion is on the
order of 10⁵–10⁶ rows.

## Decision

DuckDB, as a single file in `data/`.

## Why

**No server to operate.** Postgres would add a process to install, start, secure
and back up, in exchange for zero analytical benefit at this scale. Every
contributor and every CI runner gets a working warehouse from `uv sync` alone.

**Columnar, which is what the queries are.** Almost every query is an aggregate
over a scan — counts by source, leak rates, per-bucket medians. That is DuckDB's
home turf and Postgres's weak spot.

**Reads Parquet directly.** The landing zone can be queried without a load step,
and the published export is produced by a single `COPY ... TO`.

**Already understood.** I had built the same pattern before, including the
expensive lessons: pinning `temp_directory`, `memory_limit`, `threads` and
extension autoloading, because every DuckDB default is derived from the *host* and
is wrong inside a container.

## Consequences

**One writer.** DuckDB permits a single writable connection. This shapes the
architecture rather than merely constraining it: exactly one asset loads into the
warehouse, everything else reads. `writer()` and `reader()` make that visible in
code instead of discovered at runtime.

**Not a service.** Nothing can query the warehouse over a network. When serving
arrives in a later stage, the model artifact travels with the API rather than the
API reaching back into the warehouse.

**Concurrency limits.** Fine here — the orchestrator serialises writes and readers
open their own short-lived connections.

## Alternatives considered

| Option | Why not |
|---|---|
| **Postgres** | A server to operate for no analytical gain at 10⁵ rows |
| **SQLite** | Row-oriented; aggregate scans are exactly its weakness |
| **BigQuery / Snowflake** | Needs a credit card, and the project must run on free tiers |
| **Parquet + pandas only** | No SQL, so no dbt — and dbt is a deliberate goal, not incidental |
