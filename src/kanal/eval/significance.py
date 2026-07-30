"""Is the challenger actually better, or did it get a luckier test set?

The protocol requires two tests, both passing, before anything is promoted: a
paired bootstrap whose 95% CI *lower bound* clears the declared MDE, and
McNemar's exact test at p < 0.05. Point estimates are not evidence — with a few
hundred test rows, a 0.02 macro-F1 gap is comfortably within noise.

Both are paired. The two candidates see identical test rows, and an unpaired
test throws that structure away, which costs far more power than it looks like
it should.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kanal.eval.metrics import evaluate

BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE = 0.95


@dataclass
class BootstrapResult:
    """A confidence interval on the difference between two candidates."""

    observed_delta: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    resampled_unit: str

    def clears(self, mde: float) -> bool:
        """Whether the *lower bound* clears the minimum detectable effect.

        The lower bound, not the point estimate. A challenger whose interval is
        (-0.01, +0.08) might be better by a lot or slightly worse, and promoting
        on the midpoint is how a coin flip becomes a decision.
        """
        return self.lower > mde

    def summary(self) -> str:
        return (
            f"Δ={self.observed_delta:+.4f}  "
            f"{self.confidence:.0%} CI [{self.lower:+.4f}, {self.upper:+.4f}]  "
            f"({self.resamples:,} resamples over {self.resampled_unit}s)"
        )


def paired_bootstrap(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    *,
    clusters: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = 42,
) -> BootstrapResult:
    """Confidence interval on `macro_f1(b) - macro_f1(a)`.

    **Resamples clusters, not rows, when clusters are supplied.** A wire story
    republished by three outlets is three rows carrying one story's worth of
    independent information. Resampling rows treats them as three independent
    draws, which understates the variance and produces a confidence interval
    narrower than the data supports — the exact failure mode that turns a
    coin-flip result into a confident promotion.

    Resamples the *units*, then rebuilds the row set from whichever units were
    drawn, so a cluster is always present whole or absent whole.
    """
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError(f"length mismatch: {len(y_true)} true, {len(pred_a)} a, {len(pred_b)} b")
    if not y_true:
        raise ValueError("cannot bootstrap an empty test set")

    truth = list(y_true)
    a = list(pred_a)
    b = list(pred_b)

    observed = (
        evaluate(truth, b, labels=labels).macro_f1 - evaluate(truth, a, labels=labels).macro_f1
    )

    if clusters is not None:
        if len(clusters) != len(truth):
            raise ValueError("clusters must be the same length as the test set")
        rows_of: dict[str, list[int]] = {}
        for i, cid in enumerate(clusters):
            rows_of.setdefault(cid, []).append(i)
        units = list(rows_of.values())
        unit_name = "cluster"
    else:
        units = [[i] for i in range(len(truth))]
        unit_name = "row"

    rng = random.Random(seed)
    n_units = len(units)
    deltas = np.empty(resamples, dtype=float)

    for r in range(resamples):
        drawn: list[int] = []
        for _ in range(n_units):
            drawn.extend(units[rng.randrange(n_units)])

        t = [truth[i] for i in drawn]
        # A resample can miss a class entirely. `evaluate` excludes absent
        # classes from macro-F1 by design, so both candidates are scored over
        # the same denominator within a resample and the difference stays
        # meaningful.
        deltas[r] = (
            evaluate(t, [b[i] for i in drawn], labels=labels).macro_f1
            - evaluate(t, [a[i] for i in drawn], labels=labels).macro_f1
        )

    tail = (1.0 - confidence) / 2.0
    return BootstrapResult(
        observed_delta=observed,
        lower=float(np.quantile(deltas, tail)),
        upper=float(np.quantile(deltas, 1.0 - tail)),
        confidence=confidence,
        resamples=resamples,
        resampled_unit=unit_name,
    )


@dataclass
class McNemarResult:
    """Exact McNemar on paired predictions.

    Only the *disagreements* carry information. Rows both candidates get right,
    or both get wrong, say nothing about which is better — and including them, as
    a naive proportion test would, is what makes two candidates that differ on
    six of a thousand rows look indistinguishable when they are not.
    """

    b_only_challenger_right: int
    c_only_champion_right: int
    both_right: int
    both_wrong: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.b_only_challenger_right + self.c_only_champion_right

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def summary(self) -> str:
        return (
            f"McNemar exact p={self.p_value:.4g}  "
            f"(challenger-only {self.b_only_challenger_right}, "
            f"champion-only {self.c_only_champion_right}, "
            f"{self.discordant} discordant of "
            f"{self.discordant + self.both_right + self.both_wrong})"
        )


def mcnemar(
    y_true: Sequence[str],
    champion: Sequence[str],
    challenger: Sequence[str],
) -> McNemarResult:
    """Exact two-sided McNemar test.

    Under the null hypothesis the two candidates are equally likely to be the one
    that gets a disagreement right, so the count of challenger-only wins follows
    Binomial(b + c, 0.5). The exact test computes that tail directly rather than
    using the chi-squared approximation, which is unreliable below roughly 25
    discordant pairs — and on a test set of a few hundred rows, two similar
    candidates routinely disagree on fewer than that.
    """
    if not (len(y_true) == len(champion) == len(challenger)):
        raise ValueError("all three sequences must be the same length")
    if not y_true:
        raise ValueError("cannot test an empty test set")

    b = c = both_right = both_wrong = 0
    for truth, champ, chal in zip(y_true, champion, challenger, strict=True):
        champ_ok, chal_ok = champ == truth, chal == truth
        if champ_ok and chal_ok:
            both_right += 1
        elif not champ_ok and not chal_ok:
            both_wrong += 1
        elif chal_ok:
            b += 1
        else:
            c += 1

    n = b + c
    if n == 0:
        # No disagreements at all: the candidates are identical on this test set,
        # and there is no evidence either way.
        p = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)

    return McNemarResult(
        b_only_challenger_right=b,
        c_only_champion_right=c,
        both_right=both_right,
        both_wrong=both_wrong,
        p_value=p,
    )
