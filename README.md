# KANAL

A model-lifecycle platform for Indonesian news classification: **find the best
model, deploy it, keep it honest.**

> **Status: Stage 1 in progress — ingestion.**
> This README describes only what is built and running today. It gains a
> section when the code does, and not before.

---

## What exists right now

An ingestion layer that polls 25 RSS feeds across three Indonesian publishers
and lands new articles into a partitioned, append-only store.

```bash
uv sync
uv run kanal status      # feed registry and landing-zone state
uv run kanal ingest      # one polling cycle
uv run kanal ingest      # run it again — lands zero new rows
```

That second run landing nothing is the point, not a coincidence: see
*Idempotency* below.

---

## The task (where this is going)

Classify an Indonesian news article into one of eight *kanal* — `politik`,
`ekonomi`, `olahraga`, `teknologi`, `hiburan`, `internasional`,
`hukum-kriminal`, `gaya-hidup-kesehatan` — **from the headline and RSS summary
alone**, never the article body.

Headline-only is a deliberate difficulty setting. Ten to twenty-five tokens is
little enough that the task stays genuinely hard, which leaves room for the
comparison this project is actually about: a TF-IDF model costing fractions of
a cent against an LLM costing four orders of magnitude more, and the question
of when that difference is worth paying.

It also means no article-body scraping. Nothing is stored beyond what a
publisher chose to syndicate.

---

## Why RSS, and where the labels come from

The label of an article is **the section its publisher filed it under**. An
article arriving on `antaranews.com/rss/ekonomi.xml` is labelled `ekonomi`.

That single decision is what makes the rest of the project possible. Labels
arrive automatically, continuously, and free, so retraining has something new
to learn from — rather than being a scheduled job that re-fits the same frozen
dataset and calls itself MLOps.

### Sources

| Publisher | Feeds | Section in article URL | Item-level `<category>` |
|---|---|---|---|
| **ANTARA** (state wire agency) | 11 | no | no |
| **CNN Indonesia** | 7 | **yes** | no |
| **Liputan6** | 7 | **yes** | **yes** |

Those differences are not an inconvenience — they are the experiment. See
*Label leakage*.

---

## Engineering notes

### Idempotency, over a source that cannot be backfilled

An RSS feed is a sliding window of the last 25–100 items. There is no archive
endpoint and no `?since=` parameter: **an hour that is not captured is gone
permanently.** That single fact drives most of the design.

Polling hourly means seeing the same article many times. So identity is a hash
of the *canonicalised* URL, not the raw one — the same article arrives as

```
https://www.antaranews.com/berita/123/judul?utm_source=rss
https://www.antaranews.com/berita/123/judul/
http://antaranews.com/berita/123/judul#top
```

and all three must reduce to one `article_key`. Canonicalisation forces https,
drops `www.` and default ports, strips tracking parameters, sorts what remains,
discards the fragment, and normalises trailing slashes — while deliberately
*not* lowercasing the path, since some CMSes do serve case-sensitive slugs and
folding them would merge genuinely different articles.

Before writing, a cycle reads back which keys its target partition already
holds and writes only what is new. Re-running a poll is therefore a no-op, and
`kanal ingest` twice in a row lands zero rows the second time. That check reads
from disk rather than from memory on purpose: an in-memory set is worthless
across the process restart that is exactly when idempotency matters.

### Being a good guest

Feeds are published for syndication, and the way to stay welcome is to behave
like a well-mannered consumer:

- conditional requests via `ETag` / `If-Modified-Since`, so an unchanged feed
  costs the publisher a `304` and no body,
- one request at a time with a pause between them — never a parallel fan-out,
- a descriptive `User-Agent` with a contact route,
- exponential backoff that honours `Retry-After`,
- headline, summary, canonical URL, publisher and timestamp stored — **never
  article bodies**, never republished text.

See [`DATA.md`](DATA.md).

### One bad source must not cost the others

Two publishers are known to be unreliable — Tempo returns `403` to unfamiliar
user agents, Detik resets connections intermittently. A cycle that dies on the
first failure loses the healthy feeds too, and that hour is unrecoverable.

So every feed is isolated, and a source failing repeatedly trips a circuit
breaker and is skipped for a cooldown rather than retried every cycle. Failures
are also classified rather than lumped together: `408/429/5xx` are transient
and retried, while `403` and `404` are *verdicts* about who we are or what we
asked for, and retrying them is pure noise.

A cycle exits non-zero when an entire publisher produced no usable response, so
a scheduled run goes red instead of quietly succeeding with a hole in the data.
An individual channel going quiet is normal; a whole publisher going dark is
not.

### Label leakage, which is real here

The label *is* feed provenance, so it leaks through several channels at once:

- CNN puts the section in the article URL (`/nasional/2026...`),
- Liputan6 puts it in the URL **and** an item-level `<category>`,
- ANTARA does neither — its URLs are `/berita/{id}/{slug}`.

Any model shown a URL, a feed id, or a category scores ~100% and has learnt
nothing at all. Those fields are stored for provenance and auditing, and are
excluded from features by construction.

Publisher boilerplate is a subtler version of the same problem. `"Liputan6.com,
Jakarta - "` prefixed to a summary identifies the publisher, and publishers
have different section mixes — so the prefix alone is a partial label. It is
stripped at ingest.

The three-source contrast is what makes this demonstrable rather than merely
asserted: ANTARA is clean enough to train on in isolation, so the F1 gap
between an ANTARA-only model and an all-sources model *measures* what leakage
manufactures.

### Schema evolution

Feeds change shape without warning. Liputan6 sends `<category>`, ANTARA does
not; date formats differ; a publisher may start emitting `media:content` next
month.

If parsing only kept the fields understood today, the day a feed changed would
be a day of data silently discarded. Instead the raw layer keeps every scalar
the publisher sent in an `extra` object alongside a `schema_version`, derives
what it can, and records per-item parse failures rather than aborting a batch.
A timestamp that cannot be parsed becomes `null` rather than a guess — a wrong
publication time would corrupt the temporal split that the whole evaluation
will rest on.

---

## Layout

```
src/kanal/
├── config.py              # env-driven settings; polling policy lives here
├── cli.py                 # kanal ingest | status
└── ingest/
    ├── sources.py         # the feed registry + taxonomy mapping, with reasoning
    ├── normalize.py       # URL canonicalisation, article_key, text cleaning
    ├── fetch.py           # conditional GETs, backoff, circuit breaker
    ├── parse.py           # feed bytes → Article records, failure-tolerant
    ├── land.py            # partitioned append-only writes, idempotent
    └── run.py             # one cycle, and an honest report of it
```

Landing zone: `data/raw/source={source}/dt={YYYY-MM-DD}/{feed}-{HHMMSS}.jsonl`

Partitioned by source and UTC date so a day can be reprocessed in isolation,
and so DuckDB and pyarrow can read the tree directly with partition pruning.
One file per feed per cycle keeps writes append-only — nothing is ever
rewritten, so an interrupted run cannot corrupt what came before.

---

## Configuration

Every setting reads from the environment with a `KANAL_` prefix; the defaults
are what the scheduled job uses.

| Variable | Default | Purpose |
|---|---|---|
| `KANAL_DATA_DIR` | `./data` | Landing-zone root |
| `KANAL_REQUEST_TIMEOUT_S` | `20` | Per-request timeout |
| `KANAL_INTER_REQUEST_DELAY_S` | `1.0` | Pause between feeds |
| `KANAL_MAX_RETRIES` | `3` | Attempts per feed on transient failure |
| `KANAL_BREAKER_FAILURE_THRESHOLD` | `3` | Failures before a feed is tripped out |
| `KANAL_BREAKER_COOLDOWN_S` | `3600` | How long a tripped feed stays skipped |

---

## Not built yet

Named here so the scope is legible, and so this README cannot be mistaken for a
description of a finished system:

warehouse and dbt models · data-quality contracts · orchestration · the four
model candidates · the evaluation harness · promotion gates · serving ·
the confidence cascade · drift detection · the dashboard.

## Licence

MIT — see [LICENSE](LICENSE). Article text belongs to its publishers; this
repository stores only syndicated metadata and links back. See `DATA.md`.
