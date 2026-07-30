"""The gate that decides whether a challenger replaces the champion.

Eight conditions, all declared in `docs/evaluation.md` before any model existed.
Every one of them can refuse, every refusal carries a reason in the words the
promotion log will use, and the decision is a value rather than a side effect —
so it can be tested, replayed, and disagreed with.

The design point worth defending: **this returns REJECT, HOLD or PROMOTE, not a
boolean.** A challenger that fails the integrity check is not merely "not better"
— its evaluation is invalid, and re-running it on more data will not help. A
challenger that fails on quality might well pass next week. Collapsing those into
one answer loses the distinction that decides what to do next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kanal.eval.harness import CandidateResult, Comparison

# All from docs/evaluation.md, fixed in advance.
MDE = 0.01
ALPHA = 0.05
MAX_ECE = 0.08
MAX_P95_MS = 300.0
MAX_CLASS_REGRESSION = 0.05
MIN_TEST_ROWS = 150
COOLDOWN_HOURS = 72

# G6 has two limbs, and the first one was missing until an end-to-end run caught
# it. A tolerance alone is unusable when the incumbent is free: the majority
# baseline costs about 8e-12 USD per thousand, and 120% of that still rounds to
# nothing, so the gate refused a challenger that was 0.69 macro-F1 better with
# both significance tests passing. Any real model would have been blocked
# forever.
#
# So: within an absolute budget, OR within tolerance of the champion. The budget
# is a policy number rather than a derived one — it is what a prediction is worth
# to the product, and it is the figure the Stage 4 cascade tunes against.
MAX_USD_PER_1000 = 0.10
COST_TOLERANCE = 1.20  # or up to 20% more than the champion, whichever helps


class Verdict(StrEnum):
    PROMOTE = "PROMOTE"
    HOLD = "HOLD"
    REJECT = "REJECT"


@dataclass
class GateResult:
    """The decision, and why.

    `reasons` is never empty for HOLD or REJECT. A gate that refuses without
    saying why is indistinguishable from a bug.
    """

    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        return self.verdict is Verdict.PROMOTE

    def summary(self) -> str:
        lines = [f"{self.verdict}"]
        for reason in self.reasons:
            lines.append(f"  refused: {reason}")
        for name in self.passed:
            lines.append(f"  passed:  {name}")
        return "\n".join(lines)


@dataclass
class Incumbent:
    """What the champion scored, on the split the challenger must also have used."""

    name: str
    split_hash: str
    macro_f1: float
    per_class_f1: dict[str, float]
    usd_per_1000: float
    promoted_at: datetime | None = None
    violating_slo: bool = False


def evaluate_gate(
    challenger: CandidateResult,
    comparison: Comparison,
    incumbent: Incumbent | None,
    *,
    split_hash: str,
    n_test: int,
    cluster_leak: bool = False,
    now: datetime | None = None,
) -> GateResult:
    """Run all eight conditions and return one decision.

    `incumbent` of None means there is no champion yet — the first model to be
    evaluated. It still has to clear the absolute gates (integrity, calibration,
    latency, sample size); it simply has nothing to be better *than*.
    """
    moment = now or datetime.now(tz=UTC)
    reasons: list[str] = []
    passed: list[str] = []

    # G2 — integrity. Checked first and fatal: if a story sits in both train and
    # test, every number downstream is measuring memorisation and no amount of
    # further evaluation redeems it.
    if cluster_leak:
        return GateResult(
            verdict=Verdict.REJECT,
            reasons=[
                "G2 integrity: a cluster appears in both train and test, so this "
                "evaluation measures memorisation rather than generalisation. The "
                "result is invalid, not merely insufficient"
            ],
        )
    passed.append("G2 integrity: no cluster spans train and test")

    # G1 — provenance. A result that cannot name its split is not a result, and
    # two candidates scored on different splits were never compared.
    if not split_hash:
        reasons.append("G1 provenance: the evaluation names no split manifest hash")
    elif incumbent is not None and incumbent.split_hash != split_hash:
        return GateResult(
            verdict=Verdict.REJECT,
            reasons=[
                f"G1 provenance: challenger evaluated on split {split_hash[:12]} "
                f"but the champion's score comes from {incumbent.split_hash[:12]}. "
                f"These numbers were never compared, they were merely printed "
                f"next to each other"
            ],
        )
    else:
        passed.append(f"G1 provenance: split {split_hash[:12]}")

    if n_test < MIN_TEST_ROWS:
        reasons.append(
            f"G1 provenance: {n_test} test rows is below the {MIN_TEST_ROWS} the "
            f"protocol requires; any result here is provisional"
        )
    else:
        passed.append(f"G1 provenance: {n_test} test rows")

    # G7 — calibration. Absolute, and it applies to a first model too: the
    # cascade escalates on low confidence, and an uncalibrated gate escalates on
    # noise.
    if challenger.ece > MAX_ECE:
        direction = "over" if challenger.overconfident_by > 0 else "under"
        reasons.append(
            f"G7 calibration: ECE {challenger.ece:.4f} exceeds {MAX_ECE} "
            f"({direction}confident by {abs(challenger.overconfident_by):.3f}). "
            f"A cascade gated on this confidence would escalate on noise"
        )
    else:
        passed.append(f"G7 calibration: ECE {challenger.ece:.4f}")

    # G5 — latency, measured in the serving container at batch 1, warm.
    if challenger.p95_ms > MAX_P95_MS:
        reasons.append(f"G5 latency: p95 {challenger.p95_ms:.1f}ms exceeds {MAX_P95_MS:.0f}ms")
    else:
        passed.append(f"G5 latency: p95 {challenger.p95_ms:.1f}ms")

    if incumbent is None:
        # No champion to beat. The comparative gates cannot apply, and saying so
        # is more honest than passing them vacuously.
        verdict = Verdict.PROMOTE if not reasons else Verdict.HOLD
        passed.append("G3/G4/G6/G8: no incumbent — comparative gates not applicable")
        return GateResult(verdict=verdict, reasons=reasons, passed=passed)

    # G3 — quality. Both tests, not either. The lower bound, not the estimate.
    if comparison.bootstrap.clears(MDE) and comparison.mcnemar.significant(ALPHA):
        passed.append(
            f"G3 quality: Δ={comparison.bootstrap.observed_delta:+.4f}, "
            f"CI lower {comparison.bootstrap.lower:+.4f} > {MDE}, "
            f"McNemar p={comparison.mcnemar.p_value:.4g}"
        )
    else:
        if not comparison.bootstrap.clears(MDE):
            reasons.append(
                f"G3 quality: challenger is {comparison.bootstrap.observed_delta:+.4f} "
                f"macro-F1 ahead, but the 95% CI lower bound is "
                f"{comparison.bootstrap.lower:+.4f}, which does not clear the MDE of "
                f"{MDE} declared in advance"
            )
        if not comparison.mcnemar.significant(ALPHA):
            reasons.append(
                f"G3 quality: McNemar p={comparison.mcnemar.p_value:.4g} is not below "
                f"{ALPHA} — the {comparison.mcnemar.discordant} disagreements split "
                f"too evenly to carry a signal"
            )

    # G4 — no per-class regression. Macro-F1 can rise while a small class
    # collapses entirely, and a model that has stopped recognising
    # hukum-kriminal is not an improvement.
    regressions: list[str] = []
    for label, champ_f1 in incumbent.per_class_f1.items():
        chal_f1 = challenger.report.f1_of(label)
        drop = champ_f1 - chal_f1
        if drop > MAX_CLASS_REGRESSION:
            regressions.append(f"{label} {champ_f1:.3f} -> {chal_f1:.3f} (-{drop:.3f})")
    if regressions:
        reasons.append(
            f"G4 no-regress: {', '.join(regressions)} — a class may not lose more "
            f"than {MAX_CLASS_REGRESSION} F1 even when macro-F1 improves"
        )
    else:
        passed.append("G4 no-regress: no class lost more than 0.05 F1")

    # G6 — cost. Within the absolute budget, OR within tolerance of the champion.
    # Either limb suffices, and the first one is what makes the gate usable when
    # the incumbent is a free baseline.
    tolerance_ceiling = incumbent.usd_per_1000 * COST_TOLERANCE
    within_budget = challenger.usd_per_1000 <= MAX_USD_PER_1000
    within_tolerance = challenger.usd_per_1000 <= tolerance_ceiling

    if within_budget or within_tolerance:
        limb = "within budget" if within_budget else "within tolerance of the champion"
        passed.append(f"G6 cost: ${challenger.usd_per_1000:.6f}/1k, {limb}")
    else:
        reasons.append(
            f"G6 cost: ${challenger.usd_per_1000:.6f}/1k is over both limbs — "
            f"above the ${MAX_USD_PER_1000:.4f}/1k budget, and above the "
            f"${tolerance_ceiling:.6f}/1k that is {COST_TOLERANCE:.0%} of the "
            f"champion's ${incumbent.usd_per_1000:.6f}/1k"
        )

    # G8 — cooldown, waived when the champion is violating its own SLO. Promoting
    # repeatedly on noise is how a registry fills with churn; refusing to promote
    # while the champion is actively failing is worse.
    if incumbent.promoted_at is not None and not incumbent.violating_slo:
        elapsed = moment - incumbent.promoted_at
        if elapsed < timedelta(hours=COOLDOWN_HOURS):
            hours = elapsed.total_seconds() / 3600
            reasons.append(
                f"G8 cooldown: the champion was promoted {hours:.1f}h ago, inside "
                f"the {COOLDOWN_HOURS}h cooldown"
            )
        else:
            passed.append(f"G8 cooldown: {elapsed.days}d since the last promotion")
    elif incumbent.violating_slo:
        passed.append("G8 cooldown: waived, the champion is violating its SLO")
    else:
        passed.append("G8 cooldown: no previous promotion")

    return GateResult(
        verdict=Verdict.PROMOTE if not reasons else Verdict.HOLD,
        reasons=reasons,
        passed=passed,
    )
