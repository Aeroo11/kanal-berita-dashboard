# Evaluation protocol

**Written before any model was run.** Committed on 2026-07-30, when the only
numbers in this repository described the *data* — not a single classifier had been
fitted.

That ordering is the point. A protocol written after seeing results is not a
protocol; it is a rationalisation, and it is indistinguishable from one written in
advance once the commit history is squashed. So: every threshold, every test, and
every rule about what counts as a win is fixed here, in advance, and the git
history proves when.

If a decision below turns out to be wrong, it gets **changed in a commit that
says so and explains why** — not quietly adjusted until the numbers look better.

---

## The task

Assign an Indonesian news article to one of eight *kanal* from its **headline and
RSS summary only**. Never the article body — none is collected.

Eight classes: `politik`, `ekonomi`, `olahraga`, `teknologi`, `hiburan`,
`internasional`, `hukum-kriminal`, `gaya-hidup-kesehatan`.

## The metric

**Primary: macro-F1.** Chosen before looking at the class distribution, and the
distribution justifies it: `hukum-kriminal` has roughly a sixth of the support of
`gaya-hidup-kesehatan`. Accuracy and micro-F1 would let a model ignore the small
classes and still look good; macro-F1 weights every class equally.

**Reported alongside, always:**

- per-class F1, precision and recall — because macro-F1 hides *which* class
  collapsed
- the confusion matrix
- micro-F1 and accuracy, for comparability with other work
- **p95 latency** per prediction, measured warm at batch size 1
- **USD per 1000 predictions**
- **ECE** (expected calibration error), 15 equal-mass bins

A candidate is never described by macro-F1 alone. The whole reason this project
exists is that a single number hides the trade-off.

---

## Splitting

### Temporal, not random

```
train  published_at <= T - 14 days
val    T - 14 days  <  published_at <= T - 7 days
test   published_at >  T - 7 days
```

where `T` is the split creation time, recorded in the manifest.

Random splitting is dishonest for this task. News topics emerge and fade; a model
tested on the same week it trained on is asked an easier question than the one it
will face in production. **Both numbers will be reported** — the random-split
score alongside the temporal one — and the gap quantified explicitly, because
that gap is what most portfolio numbers are unknowingly quoting.

### Cluster-aware, never row-aware

ANTARA is a wire agency; its stories reappear near-verbatim on other sites hours
later, with different URLs and correctly different `article_key`s. A split that
partitions by row puts one copy in train and another in test, and then measures
memorisation while calling it generalisation.

**Splitting partitions by `cluster_id`.** Every article in a cluster lands on the
same side. This is asserted by a test, not assumed.

### Evergreen-aware

ANTARA mixes explainers, profiles and fixture lists into its live news feeds:
59% of its articles were over 30 days old when fetched, against 0% for CNN and
Republika. A naive temporal split therefore pushes almost all ANTARA rows into
train and leaves a test set dominated by the publishers that leak their label
through the URL.

So after splitting, **the source composition of each split is reported and
checked**. If the test set is more than 80% a single publisher, the split is
rejected and the reason recorded — a test set that is effectively one publisher
measures that publisher, not the task.

> #### Correction, 2026-07-31 — the window assumed history the corpus does not have
>
> The 14/7-day windows above were fixed when the assumption was weeks of
> collection history. The corpus has days of it, and the age distribution shows
> why the default measures the wrong side of a cliff:
>
> | percentile | article age when fetched |
> |---|---|
> | 50th | 1.26 days |
> | 75th | 3.35 days |
> | **90th** | **85.4 days** |
> | 95th | 214.9 days |
>
> Three quarters of the corpus is under 3.4 days old, and then it jumps to
> months — that jump is ANTARA's evergreen explainers, not history.
>
> **The fallback window is derived from the operational question, not from a
> table of scores:** *classify the next day's news given everything before it.*
> That gives a one-day test window, a one-day validation window, and training on
> everything earlier. No part of those numbers refers to a result.
>
> That distinction is load-bearing, because the windows were also swept and the
> sweep is exactly what a protocol exists to stop anyone selecting from:
>
> | train/val | train | test | macro-F1 | ECE |
> |---|---|---|---|---|
> | 14/7 | 163 | 1077 | 0.3015 | 0.1007 |
> | 3/1 | 347 | 517 | 0.3451 | 0.0673 |
> | 2/1 | 470 | 517 | 0.4200 | 0.0683 |
> | 1/0.5 | 778 | 341 | 0.7102 | 0.0835 |
>
> **Those macro-F1 values are not comparable with each other.** Narrowing the
> window does not only hand the model more training data — it makes the task
> easier, because predicting twelve hours ahead is a different problem from
> predicting seven days ahead. The prediction horizon is part of the task
> definition, and every run records the horizon it used so nobody later reads
> 0.71 and 0.30 as a model improvement.
>
> The default stays 14/7. The fallback fires only when the default produces a
> split with more test rows than training rows, it is logged loudly, and the run
> report carries `horizon_days` and a warning that says so.

### Frozen manifest

Every split is written to a manifest containing the article keys per split, the
cluster assignment, `T`, the code version, and a SHA-256 over the whole thing.

An evaluation result is only valid if it names the manifest hash it ran against.
Two candidates compared on different splits are not compared at all, and without
this the mistake is invisible.

---

## Leakage

The label **is** feed provenance, and it leaks: CNN's article URLs contain their
section 100% of the time, Liputan6's 98.6%, Republika's 35.6%, ANTARA's 4.1%.

**Features may only ever be derived from `title` and `summary`.**

Forbidden as features, permanently: `canonical_url`, `raw_link`, `source`,
`channel`, `feed_id`, `article_key`, `cluster_id`, `published_at`, `fetched_at`,
and every `url_leaks_*` column. A test asserts the feature vector is invariant to
all of them.

### The leakage experiment

Because ANTARA is nearly clean and the others are not, the cost of leakage can be
*measured* rather than asserted:

1. train on the clean subset (`NOT url_leaks_label`) only
2. train on everything
3. report the macro-F1 gap on a common test set

That gap is the headline finding this dataset makes possible.

> #### Correction, 2026-07-30 — this experiment does not measure what it claims
>
> Run for the first time, it produced a gap of **+0.29 macro-F1**. Two controls
> reduced it to nothing worth reporting.
>
> **The first confound was training-set size.** The clean subset holds 156 rows
> against 905 for the full pool. Holding *n* fixed at 156 and drawing ten
> size-matched samples from the full pool cut the gap from +0.29 to **+0.14** —
> the naive comparison overstated it by more than double.
>
> **The second confound removes the finding entirely.** The composition of the
> clean subset is:
>
> | | ANTARA | CNN | Liputan6 | Republika |
> |---|---|---|---|---|
> | clean training set | **84.6%** | **0%** | 0.6% | 14.7% |
> | test set | 15.2% | **59.1%** | 22.7% | 3.0% |
>
> CNN contributes **zero** rows to the clean subset, because CNN leaks its label
> 100% of the time. So "train on clean rows" means "train almost entirely on
> ANTARA", and the test set is 59% a publisher the model has never seen a single
> example of. The remaining +0.14 is a **publisher shift**, not a leakage effect.
>
> **Leakage rate and publisher identity are very nearly the same variable in this
> corpus**, so this experiment cannot separate them — not with more data, and not
> with a better model. That is a property of the data supply, not of the sample.
>
> The experiment that *can* work is a within-publisher one, on the only source
> with a genuine mix: Republika, at 36% leaky. It is small, and it will stay
> provisional for a while. It is also the only version of this that would mean
> anything.
>
> The original design is left above rather than deleted, because a protocol that
> quietly edits away its own mistakes is not a protocol.

---

## What counts as a win

### Minimum detectable effect: 0.01 macro-F1

Declared here, in advance. A challenger must beat the incumbent by more than this
for the difference to be treated as real. The figure is deliberately larger than
the noise floor a few hundred test rows can resolve, and small enough that a
genuine improvement is not dismissed.

### Significance, not point estimates

Two tests, both required:

- **Paired bootstrap**, 10 000 resamples over the test set, giving a 95%
  confidence interval on Δmacro-F1. Promotion requires the **lower bound** to
  exceed the MDE — not the point estimate.
- **McNemar's exact test** on paired predictions, requiring p < 0.05.

Paired, because both candidates see the same test rows and unpaired tests throw
away that structure.

### No per-class regression

No canonical class may lose more than 0.05 F1 relative to the incumbent, even if
macro-F1 improves. Macro-F1 can rise while a small class collapses entirely, and
a model that has stopped recognising `hukum-kriminal` is not an improvement.

### Calibration is a hard requirement

**ECE ≤ 0.08**, measured on validation after temperature scaling.

This is not decoration. The confidence-gated cascade planned for Stage 4 escalates
to the expensive model when the cheap one is unsure — and that gate is meaningless
if the confidence is not calibrated. Fine-tuned transformers are systematically
overconfident, so this must be measured rather than hoped for.

---

## Cost accounting

Cost is a first-class metric here, and there are two standard ways to lie about
it. Both are ruled out in advance.

**LLM calls are priced at published list price, even when a free tier is used.**
A free-tier call costs $0, and pricing it at $0 would make the comparison
meaningless. The price book lives in a table, versioned, next to the data.

**Self-hosted models are priced at amortised compute, never zero.** "Self-hosted
is free" is the most common falsehood in cost comparisons. TF-IDF and IndoBERT
inference cost vCPU-seconds; those are multiplied by a stated cloud rate and the
rate is recorded with the result.

**Failures count.** A call that returns unparseable output is scored **wrong**,
not dropped, and its tokens are still billed. Silently discarding malformed LLM
responses is the standard way an LLM baseline gets flattered.

**Latency is measured, not estimated**, warm, at batch size 1, on the same
machine as the other candidates.

### The cost gate has two limbs, and the first was missing

> Added 2026-07-31, after an end-to-end run.

Promotion passes the cost condition if the challenger is **either** within an
absolute budget of **USD 0.10 per 1,000 predictions**, **or** within 20% of what
the champion costs.

The tolerance limb alone was implemented first, and it was unusable. The majority
baseline costs about 8×10⁻¹² USD per thousand, so 120% of it still rounds to
nothing — and the gate refused a challenger that was **+0.69 macro-F1 better**,
with a bootstrap CI lower bound of +0.64 and McNemar at p = 7×10⁻²⁹. A baseline
that does no work would have blocked every real model forever.

The budget is a **policy number, not a derived one**: it is what a prediction is
worth to the product, and it is the figure the Stage 4 cascade has to tune
against. Stating it here rather than choosing it later is the point.

---

## The candidates

Four, not two. Two candidates is an A/B test; four is a comparison.

| candidate | why it is in the set |
|---|---|
| **majority class** | Makes every other number credible. Without it, "0.84 macro-F1" is unanchored |
| **TF-IDF + LinearSVC** | The honest baseline. If it lands within a few points of a transformer at 1/1000th the cost, that is the finding |
| **IndoBERT-lite** | A real fine-tune, on CPU, to keep the cost story real |
| **LLM zero-shot** | No training data at all, four orders of magnitude more expensive |

All four implement one `Candidate` protocol, so the harness cannot accidentally
treat them differently.

---

## Reproducibility

Every evaluation run records: the manifest hash, the git commit, the candidate
configuration, every metric above, the measured latency and cost, and the random
seed. A result that cannot state its manifest hash is not a result.

---

## Known limitations of this protocol

Stated now, before they can become excuses later.

**The clean subset is small.** At the time of writing, 229 of 1,295 rows had a URL
that does not leak — and all eight classes survive in that subset, but thinly. The
leakage experiment will be underpowered until more data accumulates. Sample sizes
will be reported with every result, and a result on fewer than 150 test rows will
be labelled provisional.

**The labels are not ground truth.** They are editorial decisions, and publishers
disagree about syndicated stories. That disagreement rate is an empirical ceiling
on achievable macro-F1, and it will be measured on cross-source duplicate
clusters rather than assumed away.

**CNN is absent from CI-collected data.** It answers HTTP 403 to datacentre IPs.
Any result computed on CI data describes three publishers, not four, and will say
so.

**One task, one language.** Nothing here generalises to another taxonomy or
another language without re-measuring.
