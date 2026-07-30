"""The promotion gate, with every condition shown refusing.

A gate whose refusal path has never executed is a gate nobody knows works. So
there is one test per condition that constructs a challenger failing exactly
that condition and asserts both the verdict and the reason.

The distinction the tests protect: REJECT means the evaluation is invalid and
re-running it on more data will not help. HOLD means it might pass next week.
Collapsing them into a boolean loses what decides the next action.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kanal.eval.harness import CandidateResult, Comparison
from kanal.eval.metrics import ClassScore, Report
from kanal.eval.significance import BootstrapResult, McNemarResult
from kanal.registry.promote import (
    COOLDOWN_HOURS,
    MAX_ECE,
    MDE,
    GateResult,
    Incumbent,
    Verdict,
    evaluate_gate,
)

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
SPLIT = "a" * 64
CLASSES = ["ekonomi", "olahraga", "politik"]


def result(
    *,
    macro_f1: float = 0.85,
    per_class: dict[str, float] | None = None,
    ece: float = 0.05,
    p95_ms: float = 50.0,
    usd: float = 0.00002,
) -> CandidateResult:
    scores = per_class or dict.fromkeys(CLASSES, macro_f1)
    return CandidateResult(
        name="challenger",
        report=Report(
            macro_f1=macro_f1,
            micro_f1=macro_f1,
            accuracy=macro_f1,
            per_class=[
                ClassScore(label=k, precision=v, recall=v, f1=v, support=100)
                for k, v in scores.items()
            ],
            confusion={},
            classes_evaluated=len(scores),
            n=500,
        ),
        ece=ece,
        overconfident_by=0.01,
        p50_ms=p95_ms / 2,
        p95_ms=p95_ms,
        usd_per_1000=usd,
        config={},
    )


def comparison(*, delta: float = 0.05, lower: float = 0.03, p: float = 0.001) -> Comparison:
    return Comparison(
        champion="champion",
        challenger="challenger",
        bootstrap=BootstrapResult(
            observed_delta=delta,
            lower=lower,
            upper=lower + 0.05,
            confidence=0.95,
            resamples=10_000,
            resampled_unit="cluster",
        ),
        mcnemar=McNemarResult(
            b_only_challenger_right=40,
            c_only_champion_right=10,
            both_right=400,
            both_wrong=50,
            p_value=p,
        ),
    )


def incumbent(
    *,
    macro_f1: float = 0.80,
    per_class: dict[str, float] | None = None,
    usd: float = 0.00002,
    promoted_at: datetime | None = None,
    violating_slo: bool = False,
    split_hash: str = SPLIT,
) -> Incumbent:
    return Incumbent(
        name="champion",
        split_hash=split_hash,
        macro_f1=macro_f1,
        per_class_f1=per_class or dict.fromkeys(CLASSES, macro_f1),
        usd_per_1000=usd,
        promoted_at=promoted_at,
        violating_slo=violating_slo,
    )


def run(**kw: object) -> GateResult:
    kw.setdefault("challenger", result())
    kw.setdefault("comparison", comparison())
    kw.setdefault("incumbent", incumbent())
    kw.setdefault("split_hash", SPLIT)
    kw.setdefault("n_test", 500)
    kw.setdefault("now", NOW)
    return evaluate_gate(**kw)  # type: ignore[arg-type]


class TestThePathThatPromotes:
    def test_a_clearly_better_challenger_is_promoted(self) -> None:
        gate = run()
        assert gate.verdict is Verdict.PROMOTE
        assert gate.reasons == []
        assert gate.promoted

    def test_every_gate_is_recorded_as_passed(self) -> None:
        # So the log shows what was checked, not only what failed.
        passed = " ".join(run().passed)
        for name in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"):
            assert name in passed


class TestIntegrityIsFatal:
    def test_a_cluster_leak_rejects_rather_than_holds(self) -> None:
        """REJECT, not HOLD — the distinction that decides what to do next.

        If a story sits in both train and test, the evaluation measures
        memorisation. Re-running it next week on more data does not redeem it;
        the split has to be rebuilt.
        """
        gate = run(cluster_leak=True)
        assert gate.verdict is Verdict.REJECT
        assert "memorisation" in gate.reasons[0]
        assert "invalid, not merely insufficient" in gate.reasons[0]

    def test_integrity_is_checked_before_anything_else(self) -> None:
        # A leaking evaluation must not be reported as "failed on calibration" —
        # that would send someone off to fix the wrong thing.
        gate = run(cluster_leak=True, challenger=result(ece=0.5, p95_ms=9999))
        assert len(gate.reasons) == 1
        assert "integrity" in gate.reasons[0]


class TestProvenance:
    def test_a_different_split_rejects(self) -> None:
        """Two candidates scored on different splits were never compared.

        Without this check the mistake is invisible: the numbers still print
        next to each other and still look comparable.
        """
        gate = run(incumbent=incumbent(split_hash="b" * 64))
        assert gate.verdict is Verdict.REJECT
        assert "never compared" in gate.reasons[0]
        assert "printed" in gate.reasons[0]

    def test_a_missing_split_hash_holds(self) -> None:
        gate = run(split_hash="")
        assert gate.verdict is Verdict.HOLD
        assert any("names no split manifest" in r for r in gate.reasons)

    def test_too_few_test_rows_holds(self) -> None:
        gate = run(n_test=100)
        assert gate.verdict is Verdict.HOLD
        assert any("provisional" in r for r in gate.reasons)


class TestQuality:
    def test_a_positive_delta_inside_the_noise_is_refused(self) -> None:
        """The rule the whole protocol exists to enforce."""
        gate = run(comparison=comparison(delta=0.02, lower=-0.005, p=0.001))
        assert gate.verdict is Verdict.HOLD
        assert any("does not clear the MDE" in r for r in gate.reasons)
        assert any(str(MDE) in r for r in gate.reasons)

    def test_an_insignificant_mcnemar_is_refused(self) -> None:
        gate = run(comparison=comparison(delta=0.05, lower=0.03, p=0.40))
        assert gate.verdict is Verdict.HOLD
        assert any("split too evenly" in r for r in gate.reasons)

    def test_both_tests_are_required(self) -> None:
        gate = run(comparison=comparison(delta=0.02, lower=-0.005, p=0.40))
        quality = [r for r in gate.reasons if r.startswith("G3")]
        assert len(quality) == 2


class TestNoRegression:
    def test_a_collapsed_small_class_blocks_a_better_macro_f1(self) -> None:
        """Macro-F1 can rise while a class disappears.

        A model that has stopped recognising one section is not an improvement,
        however good the average looks.
        """
        gate = run(
            challenger=result(
                macro_f1=0.88,
                per_class={"ekonomi": 0.95, "olahraga": 0.95, "politik": 0.20},
            ),
            incumbent=incumbent(
                macro_f1=0.80, per_class={"ekonomi": 0.80, "olahraga": 0.80, "politik": 0.80}
            ),
        )
        assert gate.verdict is Verdict.HOLD
        assert any("politik" in r and "no-regress" in r for r in gate.reasons)

    def test_a_small_drop_is_tolerated(self) -> None:
        gate = run(
            challenger=result(
                macro_f1=0.86,
                per_class={"ekonomi": 0.90, "olahraga": 0.90, "politik": 0.77},
            ),
            incumbent=incumbent(
                macro_f1=0.80, per_class={"ekonomi": 0.80, "olahraga": 0.80, "politik": 0.80}
            ),
        )
        assert gate.verdict is Verdict.PROMOTE


class TestCalibration:
    def test_a_poorly_calibrated_challenger_is_refused(self) -> None:
        gate = run(challenger=result(ece=0.15))
        assert gate.verdict is Verdict.HOLD
        assert any("escalate on noise" in r for r in gate.reasons)

    def test_the_boundary_is_inclusive(self) -> None:
        assert run(challenger=result(ece=MAX_ECE)).verdict is Verdict.PROMOTE
        assert run(challenger=result(ece=MAX_ECE + 0.001)).verdict is Verdict.HOLD


class TestLatencyAndCost:
    def test_a_slow_challenger_is_refused(self) -> None:
        gate = run(challenger=result(p95_ms=500))
        assert gate.verdict is Verdict.HOLD
        assert any("G5 latency" in r for r in gate.reasons)

    def test_a_challenger_far_more_expensive_is_refused(self) -> None:
        gate = run(
            challenger=result(usd=0.10),
            incumbent=incumbent(usd=0.00002),
        )
        assert gate.verdict is Verdict.HOLD
        assert any("G6 cost" in r for r in gate.reasons)

    def test_a_slightly_more_expensive_challenger_is_allowed(self) -> None:
        # Within the declared 20% tolerance.
        gate = run(
            challenger=result(usd=0.000022),
            incumbent=incumbent(usd=0.00002),
        )
        assert gate.verdict is Verdict.PROMOTE


class TestCooldown:
    def test_a_recent_promotion_blocks_another(self) -> None:
        gate = run(incumbent=incumbent(promoted_at=NOW - timedelta(hours=10)))
        assert gate.verdict is Verdict.HOLD
        assert any("cooldown" in r for r in gate.reasons)

    def test_the_cooldown_expires(self) -> None:
        gate = run(incumbent=incumbent(promoted_at=NOW - timedelta(hours=COOLDOWN_HOURS + 1)))
        assert gate.verdict is Verdict.PROMOTE

    def test_a_champion_violating_its_slo_waives_the_cooldown(self) -> None:
        """Refusing to promote while the champion is actively failing is worse
        than promoting inside the cooldown window."""
        gate = run(incumbent=incumbent(promoted_at=NOW - timedelta(hours=1), violating_slo=True))
        assert gate.verdict is Verdict.PROMOTE
        assert any("waived" in p for p in gate.passed)


class TestFirstModel:
    def test_with_no_incumbent_the_absolute_gates_still_apply(self) -> None:
        assert run(incumbent=None).verdict is Verdict.PROMOTE
        assert run(incumbent=None, challenger=result(ece=0.2)).verdict is Verdict.HOLD
        assert run(incumbent=None, challenger=result(p95_ms=9999)).verdict is Verdict.HOLD

    def test_the_comparative_gates_are_declared_inapplicable_not_passed(self) -> None:
        # Saying "not applicable" is more honest than passing them vacuously, and
        # a reader of the log can tell the two apart.
        gate = run(incumbent=None)
        assert any("not applicable" in p for p in gate.passed)


class TestReasonsAreAlwaysGiven:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"cluster_leak": True},
            {"n_test": 10},
            {"challenger": result(ece=0.5)},
            {"challenger": result(p95_ms=5000)},
            {"comparison": comparison(lower=-0.1)},
            {"incumbent": incumbent(split_hash="c" * 64)},
        ],
    )
    def test_no_refusal_is_ever_silent(self, kwargs: dict[str, object]) -> None:
        # A gate that refuses without saying why is indistinguishable from a bug.
        gate = run(**kwargs)
        assert gate.verdict is not Verdict.PROMOTE
        assert gate.reasons
        assert all(r.strip() for r in gate.reasons)
