---
language:
  - id
license: cc-by-4.0
task_categories:
  - text-classification
task_ids:
  - topic-classification
pretty_name: KANAL — Indonesian news headlines by section
size_categories:
  - n<10K
tags:
  - indonesian
  - news
  - rss
  - label-leakage
configs:
  - config_name: default
    data_files: articles.parquet
---

# KANAL — Indonesian news headlines, labelled by section

Indonesian news headlines and RSS summaries, labelled with the *kanal* (section)
their publisher filed them under. Collected hourly from public RSS feeds.

**1,295 articles · 4 publishers ·
8 classes**
Published between `2025-09-30 09:21:14` and
`2026-07-30 11:55:49`.

Built as the data layer of [KANAL](https://github.com/Aeroo11/kanal-berita-dashboard),
a model-lifecycle platform. Every figure below is computed from the file in this
repository, not asserted by hand.

## What is here

One row per article. Headline and summary only — **no article bodies**. Nothing
is stored beyond what each publisher chose to place in a syndication feed, and
every row links back to the original.

| column | notes |
|---|---|
| `title`, `summary` | the only fields a model should see |
| `kanal` | the label: the section the publisher filed it under |
| `source`, `channel`, `canonical_url` | provenance — see the warning below |
| `published_at`, `fetched_at` | UTC |
| `is_evergreen` | over 30 days old when fetched |
| `url_leaks_label` | the URL gives the label away |
| `cluster_id`, `cluster_size` | articles sharing a normalised headline |
| `is_cross_source_duplicate` | the same story republished by another publisher |
| `has_label_disagreement` | that story was filed under different sections |
| `label_is_judgement_call` | the channel→kanal mapping was not a direct match |

## Read this before training on it

### The URL gives the label away on most rows

| source | share of URLs containing the label |
|---|---|
| `antara` | 2.5% |
| `cnn` | 100.0% |
| `liputan6` | 98.6% |
| `republika` | 35.6% |

Publishers put the section in the article path, or in a subdomain. **Any model
shown `canonical_url`, `channel` or `source` will score near-perfectly and have
learnt nothing.** Those columns are here so the leakage can be *measured* and
reproduced, not used.

These are per-source averages, and they hide structure: Republika's rate is the
mean over its three feeds, where `ekonomi` leaks through its subdomain at 100%
while `nasional` and `internasional` are served from a neutral host and leak
almost nothing. So prefer the per-row column over the source:

```python
clean = df[~df.url_leaks_label]  # 229 of 1,295 rows
```

That clean subset is the control group. Training on it versus on everything and
comparing macro-F1 *measures* what leakage manufactures, rather than assuming it.
At source level the low-leakage publisher is
`antara`.

### Some feeds mix in evergreen content

| source | share over 30 days old at fetch |
|---|---|
| `antara` | 59.0% |
| `cnn` | 0.0% |
| `liputan6` | 11.4% |
| `republika` | 0.0% |

ANTARA files explainers, profiles and fixture lists into its live news feeds, so
a naive temporal split pushes most of its rows into train and leaves a test set
dominated by the publishers that leak. Split on `published_at`, and check the
source mix on both sides afterwards.

### Split by cluster, never by row

ANTARA is a wire agency; its stories reappear near-verbatim elsewhere hours
later, with different URLs and correctly different `article_key`s. A random split
puts one copy in train and another in test, and then measures memorisation.
`cluster_id` groups them — partition on it.

### The labels are editorial decisions, not ground truth

`has_label_disagreement` marks stories that different publishers filed under
different sections. That disagreement rate is an empirical ceiling: no model can
beat the labels' own inconsistency. `label_is_judgement_call` marks rows whose
channel→kanal mapping was a judgement rather than a direct match — the mapping,
with its reasoning, is
[in the repository](https://github.com/Aeroo11/kanal-berita-dashboard/blob/main/dbt/seeds/taxonomy_map.csv).

## Distribution

| kanal | articles |
|---|---|
| `gaya-hidup-kesehatan` | 240 |
| `ekonomi` | 185 |
| `internasional` | 185 |
| `olahraga` | 170 |
| `hiburan` | 170 |
| `teknologi` | 170 |
| `politik` | 135 |
| `hukum-kriminal` | 40 |

| source | articles |
|---|---|
| `antara` | 200 |
| `cnn` | 700 |
| `liputan6` | 350 |
| `republika` | 45 |

## Collection

Hourly polling with conditional requests, one request at a time, exponential
backoff, and a per-source circuit breaker. Full policy:
[`DATA.md`](https://github.com/Aeroo11/kanal-berita-dashboard/blob/main/DATA.md).

RSS is a sliding window of the last 25–100 items with no archive endpoint, so an
hour not captured cannot be recovered. The dataset therefore grows forward only;
it cannot be backfilled.

One publisher (CNN Indonesia) sits behind Cloudflare and answers HTTP 403 to
datacentre IPs, so it is absent from data collected in CI while being present in
local runs. That is a property of the collection environment, and it is recorded
rather than smoothed over.

## Licence and attribution

Metadata in this dataset: **CC-BY-4.0**.

Article headlines and summaries remain the property of their publishers —
ANTARA, Liputan6, Republika and CNN Indonesia. This dataset stores syndicated
metadata and links back to the originals; no article text is republished. If you
are a publisher and would prefer your feeds not be collected, open an issue on
the repository and the source will be removed.
