"""Significance testing, checked against hand-computed exact values.

McNemar's exact p-value is a binomial tail, so every expected number below was
worked out from `comb(n, k) / 2**n` rather than read off a library. That is the
same discipline as TokenWatch's reconciliation test: two independent routes to a
number must agree, or neither is trusted.
"""

from __future__ import annotations

import math

import pytest

from kanal.eval.significance import mcnemar, paired_bootstrap


def paired(both_right: int, both_wrong: int, b: int, c: int) -> tuple[list[str], ...]:
    """Build a test set with exactly the requested agreement pattern.

    b = challenger right, champion wrong.  c = the reverse.
    """
    truth = ["x"] * (both_right + both_wrong + b + c)
    champ: list[str] = []
    chal: list[str] = []

    champ += ["x"] * both_right
    chal += ["x"] * both_right
    champ += ["y"] * both_wrong
    chal += ["y"] * both_wrong
    champ += ["y"] * b
    chal += ["x"] * b
    champ += ["x"] * c
    chal += ["y"] * c

    return truth, champ, chal


class TestMcNemarExact:
    def test_no_disagreement_is_no_evidence(self) -> None:
        truth, champ, chal = paired(both_right=50, both_wrong=50, b=0, c=0)
        r = mcnemar(truth, champ, chal)
        assert r.discordant == 0
        assert r.p_value == 1.0
        assert not r.significant()

    def test_six_nil_matches_the_hand_computed_tail(self) -> None:
        # n=6, k=0:  2 * comb(6,0) / 2^6  =  2/64  =  0.03125
        truth, champ, chal = paired(both_right=100, both_wrong=0, b=6, c=0)
        r = mcnemar(truth, champ, chal)
        assert r.p_value == pytest.approx(2 / 64)
        assert r.significant()

    def test_five_nil_falls_just_short(self) -> None:
        # n=5, k=0:  2 * 1 / 32  =  0.0625  — above 0.05, so not significant.
        # The boundary is worth pinning: five straight wins is not enough.
        truth, champ, chal = paired(both_right=100, both_wrong=0, b=5, c=0)
        r = mcnemar(truth, champ, chal)
        assert r.p_value == pytest.approx(2 / 32)
        assert not r.significant()

    def test_fifteen_to_five_matches_the_hand_computed_tail(self) -> None:
        # n=20, k=5:  sum(comb(20,i) for i in 0..5) = 21700
        #             2 * 21700 / 2^20 = 0.0413895...
        expected = 2 * sum(math.comb(20, i) for i in range(6)) / 2**20
        assert expected == pytest.approx(0.0413895, abs=1e-6)

        truth, champ, chal = paired(both_right=200, both_wrong=0, b=15, c=5)
        r = mcnemar(truth, champ, chal)
        assert r.p_value == pytest.approx(expected)
        assert r.significant()

    def test_is_symmetric_in_direction(self) -> None:
        # The p-value answers "are these different", not "which is better".
        a = mcnemar(*paired(both_right=50, both_wrong=0, b=12, c=3))
        b = mcnemar(*paired(both_right=50, both_wrong=0, b=3, c=12))
        assert a.p_value == pytest.approx(b.p_value)

    def test_concordant_rows_do_not_affect_the_p_value(self) -> None:
        """Only disagreements carry information.

        This is the whole point of the test. Two candidates differing on 6 of
        1,000 rows look identical to a naive proportion test — 99.4% agreement —
        while McNemar correctly reports the difference as significant.
        """
        few = mcnemar(*paired(both_right=10, both_wrong=0, b=6, c=0))
        many = mcnemar(*paired(both_right=100_000, both_wrong=0, b=6, c=0))
        assert few.p_value == pytest.approx(many.p_value)

    def test_counts_are_reported(self) -> None:
        r = mcnemar(*paired(both_right=7, both_wrong=3, b=4, c=2))
        assert r.both_right == 7
        assert r.both_wrong == 3
        assert r.b_only_challenger_right == 4
        assert r.c_only_champion_right == 2
        assert r.discordant == 6
        assert "discordant" in r.summary()

    def test_rejects_malformed_input(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            mcnemar(["a"], ["a"], ["a", "b"])
        with pytest.raises(ValueError, match="empty"):
            mcnemar([], [], [])


class TestPairedBootstrap:
    def test_identical_candidates_give_a_zero_interval(self) -> None:
        truth = ["a", "b", "c"] * 20
        r = paired_bootstrap(truth, truth, truth, resamples=200)
        assert r.observed_delta == 0.0
        assert r.lower == 0.0
        assert r.upper == 0.0
        assert not r.clears(0.01)

    def test_a_clearly_better_challenger_clears_the_mde(self) -> None:
        truth = ["a", "b"] * 100
        champion = ["a"] * 200  # gets every 'b' wrong
        challenger = list(truth)  # perfect

        r = paired_bootstrap(truth, champion, challenger, resamples=400)
        assert r.observed_delta > 0.3
        assert r.clears(0.01)

    def test_promotion_uses_the_lower_bound_not_the_point_estimate(self) -> None:
        """A challenger ahead on average but inside the noise must not promote.

        This is the single rule the protocol exists to enforce, so the case is
        constructed rather than hoped for: on 100 rows the challenger gets two
        more right than the champion. The point estimate is positive and the
        interval straddles zero, and promoting on the midpoint is how a coin
        flip becomes a decision.

        The two error sets must *cross* rather than nest. A first attempt gave
        the challenger a strict subset of the champion's errors, and the lower
        bound came out at exactly 0.0 — correctly, because a challenger that is
        never worse on any row cannot be worse in any resample either. Crossing
        errors are what real candidates have, and what produces a straddling
        interval.
        """
        truth = ["a"] * 50 + ["b"] * 50
        # Champion misses b[0:4]; challenger misses b[4:6]. Disjoint, so each is
        # right where the other is wrong.
        champion = ["a"] * 50 + ["a"] * 4 + ["b"] * 46
        challenger = ["a"] * 50 + ["b"] * 4 + ["a"] * 2 + ["b"] * 44

        r = paired_bootstrap(truth, champion, challenger, resamples=800, seed=1)

        assert r.observed_delta > 0, "the challenger is ahead on the point estimate"
        assert r.lower < 0, "and the interval straddles zero"
        assert not r.clears(0.01), "so it must not be promoted"

    def test_is_reproducible_from_the_seed(self) -> None:
        truth = ["a", "b", "c"] * 30
        champ = ["a"] * 90
        chal = ["a", "b", "a"] * 30
        first = paired_bootstrap(truth, champ, chal, resamples=200, seed=7)
        second = paired_bootstrap(truth, champ, chal, resamples=200, seed=7)
        assert (first.lower, first.upper) == (second.lower, second.upper)

    def test_the_interval_brackets_the_observed_delta(self) -> None:
        truth = ["a", "b"] * 60
        champ = ["a"] * 120
        chal = ["a", "b"] * 55 + ["a", "a"] * 5
        r = paired_bootstrap(truth, champ, chal, resamples=400, seed=3)
        assert r.lower <= r.observed_delta <= r.upper

    def test_cluster_resampling_widens_the_interval(self) -> None:
        """The reason the bootstrap resamples clusters rather than rows.

        A wire story republished by five outlets is five rows carrying one
        story's worth of independent information. Resampling rows counts it five
        times and produces an interval narrower than the data supports — which
        is exactly how a coin-flip result becomes a confident promotion.
        """
        truth: list[str] = []
        champ: list[str] = []
        chal: list[str] = []
        clusters: list[str] = []

        # 40 stories, each duplicated across 5 outlets.
        for i in range(40):
            label = "a" if i % 2 else "b"
            wrong = "b" if label == "a" else "a"
            for _ in range(5):
                truth.append(label)
                clusters.append(f"story{i}")
                champ.append(label if i % 3 else wrong)
                chal.append(label)

        by_row = paired_bootstrap(truth, champ, chal, resamples=400, seed=5)
        by_cluster = paired_bootstrap(truth, champ, chal, clusters=clusters, resamples=400, seed=5)

        assert (by_cluster.upper - by_cluster.lower) > (by_row.upper - by_row.lower)
        assert by_cluster.resampled_unit == "cluster"
        assert by_row.resampled_unit == "row"

    def test_cluster_resampling_keeps_clusters_whole(self) -> None:
        # If a cluster could be partially drawn, the correlation it represents
        # would leak back in and the widening above would be an illusion.
        truth = ["a"] * 10 + ["b"] * 10
        clusters = ["c1"] * 10 + ["c2"] * 10
        r = paired_bootstrap(truth, truth, truth, clusters=clusters, resamples=50, seed=2)
        assert r.resampled_unit == "cluster"
        assert r.observed_delta == 0.0

    def test_rejects_malformed_input(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            paired_bootstrap(["a"], ["a"], ["a", "b"], resamples=10)
        with pytest.raises(ValueError, match="empty"):
            paired_bootstrap([], [], [], resamples=10)
        with pytest.raises(ValueError, match="same length"):
            paired_bootstrap(["a"], ["a"], ["a"], clusters=["x", "y"], resamples=10)


class TestTheGateSaysNo:
    """Both tests together, on a case a naive comparison would get wrong.

    A gate that has only ever said yes is a gate nobody knows works. This is the
    worked example from the README: a challenger whose *point estimate beats the
    declared MDE* and which is still, correctly, refused.
    """

    @staticmethod
    def _corpus() -> tuple[list[str], list[str], list[str]]:
        import numpy as np

        rng = np.random.default_rng(11)
        labels = ["politik", "ekonomi", "olahraga", "hiburan"]
        truth = list(rng.choice(labels, size=500))

        def corrupt(n_wrong: int, offset: int) -> list[str]:
            out = list(truth)
            for i in range(offset, offset + n_wrong):
                idx = i % len(out)
                wrong = [c for c in labels if c != out[idx]]
                out[idx] = wrong[i % len(wrong)]
            return out

        return truth, corrupt(60, 0), corrupt(54, 30)

    def test_a_challenger_past_the_mde_on_the_point_estimate_is_still_refused(
        self,
    ) -> None:
        """champion 0.8795, challenger 0.8923, Δ +0.0128 — and the answer is no.

        The point estimate clears the 0.01 MDE. Both remaining tests refuse it:
        the bootstrap lower bound is negative, so the challenger might be worse;
        and McNemar returns p ≈ 0.50, meaning the disagreements split almost
        evenly and carry no signal at all.

        Promoting this would be indistinguishable from promoting a coin flip.
        """
        truth, champion, challenger = self._corpus()
        labels = sorted(set(truth))

        boot = paired_bootstrap(
            truth, champion, challenger, labels=labels, resamples=2_000, seed=42
        )
        mc = mcnemar(truth, champion, challenger)

        assert boot.observed_delta > 0.01, "the point estimate clears the MDE"
        assert boot.lower < 0, "but the interval says it might be worse"
        assert not mc.significant(), "and the disagreements carry no signal"
        assert not (boot.clears(0.01) and mc.significant()), "so: no promotion"

    def test_a_genuinely_better_challenger_passes_both(self) -> None:
        # The same machinery must not simply refuse everything.
        import numpy as np

        rng = np.random.default_rng(11)
        labels = ["politik", "ekonomi", "olahraga", "hiburan"]
        truth = list(rng.choice(labels, size=500))

        def corrupt(n_wrong: int, offset: int) -> list[str]:
            out = list(truth)
            for i in range(offset, offset + n_wrong):
                idx = i % len(out)
                wrong = [c for c in labels if c != out[idx]]
                out[idx] = wrong[i % len(wrong)]
            return out

        boot = paired_bootstrap(
            truth,
            corrupt(120, 0),
            corrupt(40, 200),
            labels=labels,
            resamples=2_000,
            seed=42,
        )
        mc = mcnemar(truth, corrupt(120, 0), corrupt(40, 200))

        assert boot.clears(0.01)
        assert mc.significant()
