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

Classifies Indonesian news into one of eight sections from the headline and
optional RSS summary. The demo serves a TF-IDF model with a linear SVM.

Open `/docs` on the Space's app URL to try Swagger. The same API can be called
with curl. Set `BASE_URL` to the app URL first.

```bash
curl -X POST "$BASE_URL/predict" \
  -H 'content-type: application/json' \
  -d '{"articles":[{"title":"Persib melaju ke final setelah menang lewat adu penalti"}]}'
```

The response includes the category, its probability, the model ID, and
latency. The serving process checks the model alias every 60 seconds.

[Dataset on Hugging Face](https://huggingface.co/datasets/aeroo11/kanal-berita) ·
[Source and local setup](https://github.com/Aeroo11/kanal-berita-dashboard)

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

On the pinned 1,295-article snapshot, TF-IDF achieved 0.420 macro-F1 versus
0.026 for the majority baseline on 517 test articles at a one-day horizon.
F1 ranged from 0.083 for law and crime to 0.839 for sports. These scores
describe the snapshot and split used for this demo.
