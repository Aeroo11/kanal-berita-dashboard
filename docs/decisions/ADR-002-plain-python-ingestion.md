# ADR-002: Plain Python for ingestion, not dlt

**Status:** accepted · **Date:** 2026-07-29

## Context

The pipeline polls 27 RSS feeds hourly and lands new articles idempotently.
`dlt` exists for exactly this shape of job and would supply incremental loading,
schema inference and state management out of the box.

## Decision

Plain Python: `httpx` for transport, `feedparser` for parsing, and hand-written
idempotency.

## Why

**The idempotency logic is the thing being demonstrated.** This is a portfolio
project targeting data engineering, and "how do you make ingestion idempotent
over a source that cannot be backfilled?" is the question it exists to answer.
Delegating that to a framework would mean answering "I imported a library that
does it", and being unable to explain the mechanism in an interview.

That is not hypothetical. The mechanism has already needed two corrections that
only understanding it could produce:

- keys are hashed from the *canonicalised* URL, after stripping tracking
  parameters — otherwise the same article lands under several identities;
- deduplication scans a **window of previous day-partitions**, not just today's.
  The first version was correct within a day and wrong across one: at midnight
  UTC every article still in a feed looked unseen, and 39.1% of landed lines were
  re-landings before it was caught.

**The whole module is ~150 lines.** The framework would be larger than the code
it replaced.

**Failure handling here is domain-specific.** `403` and `404` are verdicts about
who we are and must not be retried; `408/429/5xx` are transient and must be. A
publisher that fails repeatedly is tripped out for a cooldown so it cannot cost
us the healthy feeds — because RSS is a sliding window and a lost hour is
unrecoverable. That policy is not a configuration option in any framework.

## Consequences

**More code to own**, including the backoff, the circuit breaker and the
conditional-request state. All of it is tested.

**No free schema evolution.** Handled explicitly instead: the raw layer stores
every scalar the publisher sent in an `extra` blob alongside a `schema_version`,
so a feed changing shape degrades rather than losing a day.

**Reconsider if the source count grows.** At twenty publishers with varied
protocols the arithmetic changes and a framework starts earning its complexity.
At four it does not.

## Alternatives considered

| Option | Why not |
|---|---|
| **dlt** | Would hide the one mechanism the project exists to demonstrate |
| **Airbyte / Meltano** | A service to operate, for four RSS feeds |
| **Scrapy** | Built for crawling; there is no crawl here, and adding one is out of scope |
