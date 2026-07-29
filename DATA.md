# Data sources, licensing, and polling policy

This project reads publicly syndicated RSS feeds from three Indonesian news
publishers. This document states exactly what is collected, what is not, and
how the collection behaves — because a research project that quietly takes more
than it should is one that deserves to be blocked.

## Sources

| Publisher | Feeds | RSS index |
|---|---|---|
| ANTARA (state news agency) | 11 | `https://www.antaranews.com/rss/` |
| CNN Indonesia | 7 | `https://www.cnnindonesia.com/{section}/rss` |
| Liputan6 | 7 | `https://feed.liputan6.com/rss/{channel}` |

## What is stored

Only what the publisher chose to place in a syndication feed:

- headline
- the feed's own summary or description
- canonical article URL (the link back)
- publisher and channel, as provenance
- publication timestamp
- fetch timestamp and feed identifier

## What is deliberately not stored

- **Article bodies.** No page beyond the feed is ever requested. There is no
  scraper in this repository and adding one is explicitly out of scope.
- **Images.** Thumbnail markup embedded in summaries is stripped at ingest.
- **Anything behind a paywall, login, or `robots.txt` disallow.**

Article text remains the property of its publisher. This repository stores
syndicated metadata and links back to the original. Nothing here is a substitute
for reading the source, and no full text is republished.

## Polling policy

RSS exists to be consumed, and the way to stay welcome is to behave like a
well-mannered consumer:

| Behaviour | Value |
|---|---|
| Poll interval | hourly, per feed |
| Concurrency | **one request at a time** — never a parallel fan-out |
| Pause between requests | 1 second |
| Conditional requests | `If-None-Match` / `If-Modified-Since` on every poll |
| Timeout | 20 s |
| Retries | 3, exponential backoff, `Retry-After` honoured |
| Retried statuses | 408, 425, 429, 5xx only |
| Not retried | 403, 404 — these are verdicts, and retrying them is noise |
| Circuit breaker | 3 consecutive failures → feed skipped for 1 hour |
| User-Agent | descriptive, with a contact route |

The current `User-Agent`:

```
kanal-research/0.1 (+https://github.com/Aeroo11/kanal-berita-dashboard)
Indonesian news classification research; contact via GitHub issues
```

Conditional requests matter more than they look: a feed polled hourly is
unchanged most of the time, so the publisher answers `304` with no body. That
is cheaper for them than for us.

## Why hourly, and why it cannot be less

An RSS feed is a sliding window of its last 25–100 items. There is no archive
endpoint and no `?since=` parameter. **An hour that is not captured cannot be
recovered.** Hourly polling is the minimum rate that keeps the window from
overtaking us on a busy news day; it is not an attempt to be first to anything.

## Labels

The label of an article is the channel it was syndicated on — an article from
`antaranews.com/rss/ekonomi.xml` is labelled `ekonomi`. Publisher channels are
mapped onto eight canonical classes in `src/kanal/ingest/sources.py`, where each
judgement call is recorded with its reasoning.

These labels are the publishers' editorial decisions, not ground truth. They
disagree with one another on syndicated stories, and that disagreement is
measured rather than assumed away — see the label-noise ceiling in the
evaluation.

## If you are a publisher

If you would prefer this project not to poll your feeds, open an issue on the
repository and the source will be removed. No further justification needed.

## Redistribution

The derived dataset published to Hugging Face contains headlines, summaries,
links and labels — the same syndicated metadata described above. It is intended
for research and is not a substitute for the publishers' own content.

Seed corpora used for cold-start smoke tests carry their own licences:

- `indonesian-nlp/id_newspapers_2018` — CC-BY-4.0
- `fahadh4ilyas/indonesian_news_datasets` — CC-BY-NC-4.0 (non-commercial)
