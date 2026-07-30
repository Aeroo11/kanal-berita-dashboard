"""Generate the Hugging Face dataset card.

Generated rather than hand-written, because the numbers in it must match the
Parquet beside it. A card claiming 1,270 articles next to a file holding 824 is
worse than a card with no numbers at all — and hand-maintained figures drift the
moment ingestion runs again.

The prose is fixed; every figure is read from the export.
"""

from __future__ import annotations

import json
from pathlib import Path


def _table(rows: dict[str, int | float], header: tuple[str, str], pct: bool = False) -> str:
    lines = [f"| {header[0]} | {header[1]} |", "|---|---|"]
    for key, value in rows.items():
        rendered = f"{value * 100:.1f}%" if pct else f"{value:,}"
        lines.append(f"| `{key}` | {rendered} |")
    return "\n".join(lines)


def build_card(stats_path: Path) -> str:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    leak = stats["url_leak_rate_by_source"]
    evergreen = stats["evergreen_rate_by_source"]
    clean_sources = sorted(s for s, rate in leak.items() if rate < 0.25)
    clean_rows = int(stats["rows_without_url_leak"])

    return f"""---
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

**{stats["articles"]:,} articles · {stats["sources"]} publishers ·
{stats["kanal_classes"]} classes**
Published between `{stats["published_range"]["oldest"]}` and
`{stats["published_range"]["newest"]}`.

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

{_table(leak, ("source", "share of URLs containing the label"), pct=True)}

Publishers put the section in the article path, or in a subdomain. **Any model
shown `canonical_url`, `channel` or `source` will score near-perfectly and have
learnt nothing.** Those columns are here so the leakage can be *measured* and
reproduced, not used.

These are per-source averages, and they hide structure: Republika's rate is the
mean over its three feeds, where `ekonomi` leaks through its subdomain at 100%
while `nasional` and `internasional` are served from a neutral host and leak
almost nothing. So prefer the per-row column over the source:

```python
clean = df[~df.url_leaks_label]      # {clean_rows:,} of {stats["articles"]:,} rows
```

That clean subset is the control group. Training on it versus on everything and
comparing macro-F1 *measures* what leakage manufactures, rather than assuming it.
At source level the low-leakage publisher is
{", ".join(f"`{s}`" for s in clean_sources) or "none"}.

### Some feeds mix in evergreen content

{_table(evergreen, ("source", "share over 30 days old at fetch"), pct=True)}

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

{_table(stats["articles_by_kanal"], ("kanal", "articles"))}

{_table(stats["articles_by_source"], ("source", "articles"))}

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
"""


def write_card(stats_path: Path, out_path: Path | None = None) -> Path:
    target = out_path or (stats_path.parent / "README.md")
    target.write_text(build_card(stats_path), encoding="utf-8")
    return target
