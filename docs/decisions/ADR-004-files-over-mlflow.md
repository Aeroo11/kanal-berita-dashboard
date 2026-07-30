# ADR-004: A directory of files instead of MLflow

**Status:** accepted · **Date:** 2026-07-31

## Context

The plan for this project named MLflow for experiment tracking and the model
registry, on the reasoning that it is the largest MLOps gap in the author's
portfolio and that champion/challenger aliases give clean promotion and
rollback.

By the time Stage 3 was implemented, the registry's actual job had become clear.
It is three operations:

1. resolve an alias to an artifact id
2. move an alias, keeping the outgoing value as `previous`
3. append a decision to a log

Each is a handful of lines. The serving path additionally needs to re-read an
alias on a timer, which is one file read.

## Decision

Implement the registry as a directory: content-addressed artifact folders, one
small JSON file per alias, and an append-only `decisions.jsonl`.

Do not run MLflow.

## Why

**The dependency is load-bearing at the worst moment.** The API resolves the
champion alias on every poll. With MLflow that read becomes a call to a tracking
server — so the serving path acquires a runtime dependency on a second process,
and the failure mode is that a rollback cannot be read *during the incident the
rollback exists for*. A file on the same filesystem cannot be unreachable in a
way the API is not already dead from.

**Nothing here uses what MLflow is for.** Its value is in experiment comparison
across many runs, parameter search, and artifact browsing across a team. This
project fits four candidates on one split and records the comparison in a JSON
file the harness already writes. Adding a service to store three values buys a
UI for data that is legible as text.

**The promotion log has a requirement MLflow does not meet naturally.** It must
record *refusals* — the challengers that were evaluated and turned down, with the
reason. That is the entry worth reading. MLflow's registry models transitions
between stages, and a challenger that never transitioned leaves no first-class
record of having been considered and rejected.

**A file registry is inspectable without running anything.** `cat aliases/champion.json`
answers "what is serving" during an incident, on a machine with no Python
environment. That property matters more than it sounds.

## What this gives up

- **No experiment UI.** Comparing ten runs means reading ten JSON files or
  writing the query. Acceptable at this scale; annoying at fifty runs.
- **No artifact browser**, no parameter search, no run comparison plots.
- **No multi-user story.** Two people promoting at once would race. The atomic
  alias write makes that safe rather than corrupting, but there is no locking
  and no audit of *who*.
- **The nudge value is lost.** MLflow on a CV is a recognised line item; "I wrote
  a registry" is not, until someone reads the code. That is a real cost for a
  portfolio project, and it is being paid deliberately.

## When to revisit

Move to MLflow when any of these becomes true:

- more than one person promotes models
- runs accumulate past the point where reading JSON is faster than a UI
- experiment comparison across many hyperparameter settings starts to matter,
  which would arrive with a real tuning phase

The migration is mechanical because everything is content-addressed: an artifact
id is a hash of its provenance, so importing the directory into MLflow is a walk
over `artifacts/` plus one `set_registered_model_alias` per alias. Nothing about
the storage layout would have to be reverse-engineered.

## Related

- [ADR-002](ADR-002-plain-python-over-dlt.md) made the same trade for ingestion:
  the logic being demonstrated is the logic worth writing.
- The atomic alias write in `registry/store.py` was added after a concurrency
  test caught `write_text` truncating the file a serving process was reading.
