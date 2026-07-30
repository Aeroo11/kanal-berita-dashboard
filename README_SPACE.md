---
title: KANAL
emoji: 📰
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# KANAL — Indonesian news section classifier

Classifies an Indonesian news article into one of eight sections from the
**headline and RSS summary alone** — never the article body, because none is
collected.

- **Try it:** [`/docs`](./docs) — Swagger, clickable
- **Dataset:** [`aeroo11/kanal-berita`](https://huggingface.co/datasets/aeroo11/kanal-berita)
- **Source:** [github.com/Aeroo11/kanal-berita-dashboard](https://github.com/Aeroo11/kanal-berita-dashboard)

```bash
curl -X POST https://aeroo11-kanal.hf.space/predict \
  -H 'content-type: application/json' \
  -d '{"articles":[{"title":"Bank Indonesia pertahankan suku bunga acuan"}]}'
```

Every response names the model that answered, its confidence, the measured
latency, and what a thousand such calls cost. The champion is resolved from an
alias on a 60-second timer, so a rollback takes effect without a redeploy.

## What is worth knowing before trusting it

The label is the feed section the article was published under, which means it
**leaks through the URL** — 100% of the time for CNN, 98.6% for Liputan6, 35.6%
for Republika, 4.1% for ANTARA. The model never sees a URL: the input struct
carries the title and summary and does not have a field for anything else.

The evaluation protocol was written and committed **before any model was run**,
and it records two findings that were withdrawn under control:

- a leakage experiment that looked like +0.29 macro-F1 turned out to be
  publisher composition, not leakage — leakage rate and publisher are nearly the
  same variable in this corpus
- a temporal-versus-random split gap of 0.43 was mostly training-set size

See [`docs/evaluation.md`](https://github.com/Aeroo11/kanal-berita-dashboard/blob/main/docs/evaluation.md).

Current scores are **provisional**: the corpus has only a few days of collection
history, so the temporal split has too few training rows to mean much. It grows
hourly.
