"""One protocol every candidate implements.

Four candidates spanning four orders of magnitude in cost only compare honestly
if the harness treats them identically. A protocol makes that structural: the
evaluation code cannot special-case a candidate it cannot distinguish.

Each must be able to state three things about itself, and the third is the one
usually left out:

- `predict` — the label, with a calibrated probability over all classes
- `describe` — enough configuration to reproduce the run
- `unit_cost_usd` — what 1,000 predictions cost, *including* when self-hosted

That last one is not optional. "Self-hosted is free" is the most common falsehood
in cost comparisons: TF-IDF inference costs vCPU-seconds, and pricing them at
zero makes the whole comparison meaningless. Every candidate prices itself, and
`docs/evaluation.md` fixes the rate.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from kanal.features.text import Example

# Amortised compute, so self-hosted candidates cannot price themselves at zero.
# A shared 2 vCPU box at roughly USD 0.03/hour is USD 8.3e-6 per vCPU-second.
# The figure is stated here rather than buried, and any result quoting a cost
# records it.
USD_PER_VCPU_SECOND = 8.3e-6


@dataclass(frozen=True)
class Prediction:
    """One classification, with the confidence a cascade can gate on."""

    label: str
    confidence: float
    probabilities: dict[str, float]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")


@dataclass
class Timing:
    """Measured latency, never estimated.

    p95 at batch size 1 and warm, because that is what a serving path
    experiences. A throughput number from a large batch would flatter every
    candidate unequally — the transformer most of all.
    """

    samples_ms: list[float] = field(default_factory=list)

    def record(self, ms: float) -> None:
        self.samples_ms.append(ms)

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.samples_ms, 50)) if self.samples_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.samples_ms, 95)) if self.samples_ms else 0.0

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.samples_ms)) if self.samples_ms else 0.0

    def summary(self) -> str:
        return (
            f"p50={self.p50_ms:.2f}ms  p95={self.p95_ms:.2f}ms  "
            f"(n={len(self.samples_ms)}, batch=1, warm)"
        )


@runtime_checkable
class Candidate(Protocol):
    """What every candidate must provide.

    `runtime_checkable` so a test can assert conformance rather than trusting
    that four separate classes stayed in step.
    """

    name: str

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        """Fit on training text. A candidate that does not train implements this
        as a no-op recording the label set — the majority baseline does exactly
        that, and it still has to be *told* what the classes are."""
        ...

    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...

    def describe(self) -> dict[str, object]:
        """Configuration sufficient to reproduce this run."""
        ...

    def unit_cost_usd(self, n: int = 1000) -> float:
        """Cost of `n` predictions, in USD. Never zero for self-hosted."""
        ...


def measure_latency(
    candidate: Candidate,
    examples: Sequence[Example],
    *,
    warmup: int = 5,
    to_text_fn: object = None,
) -> Timing:
    """Time one prediction at a time, after a warm-up.

    The warm-up is not ceremony. The first call through a fitted sklearn
    pipeline, or a transformer's first forward pass, pays one-off costs that a
    serving process pays once at boot and never again — including them would
    measure the wrong thing.
    """
    from kanal.features.text import to_text as default_to_text

    convert = to_text_fn if callable(to_text_fn) else default_to_text
    texts = [convert(e) for e in examples]
    if not texts:
        return Timing()

    for i in range(min(warmup, len(texts))):
        candidate.predict([texts[i]])

    timing = Timing()
    for text in texts:
        start = time.perf_counter()
        candidate.predict([text])
        timing.record((time.perf_counter() - start) * 1000.0)
    return timing
