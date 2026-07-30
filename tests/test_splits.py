"""Temporal, cluster-aware splitting.

One invariant matters more than everything else here: no cluster may appear in
two splits. A wire story republished by three outlets is three rows and one
story, and if one copy lands in train while another lands in test, the
evaluation measures memorisation and reports it as generalisation.

Every other test in this file is about not producing a split that is technically
valid and practically meaningless.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kanal.data.splits import (
    MIN_TEST_ROWS,
    Article,
    assert_no_cluster_leak,
    random_split,
    temporal_split,
)

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


def article(
    key: str,
    days_ago: float,
    *,
    cluster: str | None = None,
    source: str = "antara",
    kanal: str = "ekonomi",
) -> Article:
    return Article(
        article_key=key,
        cluster_id=cluster or key,
        published_at=NOW - timedelta(days=days_ago),
        source=source,
        kanal=kanal,
    )


def spread(
    n: int,
    days_ago: float,
    *,
    source: str = "antara",
    kanal: str = "ekonomi",
) -> list[Article]:
    """`n` distinct articles at the same age.

    The key includes source and kanal, not just the age. An earlier version did
    not, so two calls at the same age produced colliding keys and the second
    silently overwrote the first — which made a deliberately balanced test set
    come out as a single publisher. Same class of mistake as the CI fixture that
    suffixed titles per source rather than per feed.
    """
    return [
        article(f"k{days_ago}_{source}_{kanal}_{i}", days_ago, source=source, kanal=kanal)
        for i in range(n)
    ]


class TestBoundaries:
    def test_assigns_by_the_declared_windows(self) -> None:
        articles = [
            article("old", 20),
            article("mid", 10),
            article("new", 2),
        ]
        m = temporal_split(articles, anchor=NOW)
        assert m.assignment["old"] == "train"
        assert m.assignment["mid"] == "val"
        assert m.assignment["new"] == "test"

    def test_boundaries_are_inclusive_on_the_earlier_side(self) -> None:
        # Exactly 14 days old is train; a moment later is val. Stated so the
        # edge is a decision rather than an accident of comparison operators.
        articles = [article("exactly14", 14), article("justunder14", 13.99)]
        m = temporal_split(articles, anchor=NOW)
        assert m.assignment["exactly14"] == "train"
        assert m.assignment["justunder14"] == "val"

    def test_anchor_defaults_to_the_newest_article(self) -> None:
        articles = [article("a", 30), article("b", 0)]
        m = temporal_split(articles)
        assert m.anchor == articles[1].published_at.isoformat()

    def test_empty_corpus_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty corpus"):
            temporal_split([])


class TestClusterIntegrity:
    def test_no_cluster_spans_two_splits(self) -> None:
        articles = [
            *spread(20, 20),
            *spread(20, 10),
            *spread(20, 2),
        ]
        m = temporal_split(articles, anchor=NOW)
        assert_no_cluster_leak(m)

    def test_a_cluster_spanning_a_boundary_goes_to_the_later_split(self) -> None:
        """The direction is not arbitrary.

        Sending it to the later split puts an older copy into test, which is
        harmless. Sending it to the earlier split would put a test-period article
        into training — a model learning from the period it is about to be
        evaluated on. Only one of those inflates a score.
        """
        articles = [
            article("train_era", 20, cluster="wire"),
            article("test_era", 2, cluster="wire"),
        ]
        m = temporal_split(articles, anchor=NOW)

        assert m.assignment["train_era"] == "test"
        assert m.assignment["test_era"] == "test"
        assert m.clusters_spanning_boundary == 1

    def test_the_leak_check_actually_catches_a_leak(self) -> None:
        # A check never seen to fail is a check nobody knows works.
        articles = [*spread(10, 20), *spread(10, 2)]
        m = temporal_split(articles, anchor=NOW)

        # Force two members of one cluster onto different sides.
        keys = sorted(m.assignment)
        m.cluster_of[keys[0]] = "shared"
        m.cluster_of[keys[-1]] = "shared"
        m.assignment[keys[0]] = "train"
        m.assignment[keys[-1]] = "test"

        with pytest.raises(AssertionError, match="memorisation"):
            assert_no_cluster_leak(m)

    def test_random_split_is_also_cluster_aware(self) -> None:
        # Comparing a cluster-aware temporal split against a row-wise random one
        # would conflate two effects, and the interesting one is the ordering.
        articles = [article(f"a{i}", i % 30, cluster=f"c{i // 3}") for i in range(60)]
        assert_no_cluster_leak(random_split(articles))


class TestCompositionChecks:
    def test_warns_when_the_test_set_is_effectively_one_publisher(self) -> None:
        # The evergreen problem, made concrete: ANTARA's months-old explainers
        # land in train, leaving a test set that is almost entirely CNN — which
        # also leaks its label 100% of the time, against ANTARA's 4%.
        articles = [
            *spread(30, 20, source="antara"),
            *spread(30, 2, source="cnn"),
            *spread(2, 2, source="antara"),
        ]
        m = temporal_split(articles, anchor=NOW)
        assert any("effectively one publisher" in w for w in m.warnings)

    def test_no_warning_on_a_balanced_test_set(self) -> None:
        articles = [
            *spread(20, 20, source="antara"),
            *spread(15, 2, source="cnn"),
            *spread(15, 2, source="liputan6"),
        ]
        m = temporal_split(articles, anchor=NOW)
        assert not any("effectively one publisher" in w for w in m.warnings)

    def test_warns_when_a_split_is_missing_classes(self) -> None:
        articles = [*spread(10, 20, kanal="ekonomi"), *spread(10, 2, kanal="politik")]
        m = temporal_split(articles, anchor=NOW)
        assert any("only" in w and "of 8 classes" in w for w in m.warnings)

    def test_marks_a_small_test_set_provisional(self) -> None:
        # The protocol's own rule, enforced rather than remembered.
        articles = [*spread(200, 20), *spread(10, 2)]
        m = temporal_split(articles, anchor=NOW)
        assert m.is_provisional
        assert str(MIN_TEST_ROWS) in m.summary()

    def test_a_large_test_set_is_not_provisional(self) -> None:
        articles = [*spread(300, 20), *spread(MIN_TEST_ROWS + 10, 2)]
        m = temporal_split(articles, anchor=NOW)
        assert not m.is_provisional

    def test_warns_when_test_is_larger_than_train(self) -> None:
        """The pathology the composition check cannot see.

        Measured on the first real run: train=163, val=55, test=1077. The split
        was technically correct — every assignment right, no cluster leaking —
        and practically useless, because the *collection* period was two days
        while published_at ranged over a year. That range came from ANTARA's
        evergreen explainers, and old publication dates are not history.
        """
        articles = [*spread(20, 20), *spread(200, 2)]
        m = temporal_split(articles, anchor=NOW)
        assert any("larger than train" in w for w in m.warnings)
        assert any("not history" in w for w in m.warnings)

    def test_no_such_warning_on_a_corpus_with_real_history(self) -> None:
        articles = [*spread(200, 30), *spread(60, 10), *spread(60, 2)]
        m = temporal_split(articles, anchor=NOW)
        assert not any("larger than train" in w for w in m.warnings)


class TestManifest:
    def test_hash_is_stable_for_the_same_split(self) -> None:
        articles = [*spread(10, 20), *spread(10, 2)]
        a = temporal_split(articles, anchor=NOW)
        b = temporal_split(articles, anchor=NOW)
        assert a.hash == b.hash

    def test_hash_ignores_creation_time(self) -> None:
        # Otherwise the identifier is a timestamp wearing a hash's clothes, and
        # two runs of the same split would look like different splits.
        articles = [*spread(10, 20), *spread(10, 2)]
        m = temporal_split(articles, anchor=NOW)
        before = m.hash
        m.created_at = "1999-01-01T00:00:00+00:00"
        assert m.hash == before

    def test_hash_changes_when_the_assignment_changes(self) -> None:
        articles = [*spread(10, 20), *spread(10, 2)]
        m = temporal_split(articles, anchor=NOW)
        before = m.hash
        m.assignment[sorted(m.assignment)[0]] = "test"
        assert m.hash != before

    def test_hash_changes_with_the_anchor(self) -> None:
        articles = [*spread(10, 30), *spread(10, 2)]
        a = temporal_split(articles, anchor=NOW)
        b = temporal_split(articles, anchor=NOW - timedelta(days=5))
        assert a.hash != b.hash

    def test_writes_a_readable_manifest(self, tmp_path: Path) -> None:
        articles = [*spread(10, 20), *spread(10, 2)]
        m = temporal_split(articles, anchor=NOW)
        path = m.write(tmp_path / "split.json")

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["hash"] == m.hash
        assert loaded["counts"]["train"] == 10
        assert "assignment" in loaded
        # A manifest without its parameters cannot be reproduced from itself.
        assert loaded["anchor"] == m.anchor
        assert loaded["train_cutoff"] == m.train_cutoff

    def test_keys_returns_only_that_split(self) -> None:
        articles = [*spread(10, 20), *spread(5, 2)]
        m = temporal_split(articles, anchor=NOW)
        assert len(m.keys("train")) == 10
        assert len(m.keys("test")) == 5
        assert set(m.keys("train")).isdisjoint(m.keys("test"))


class TestRandomSplitComparison:
    def test_is_reproducible_from_the_seed(self) -> None:
        articles = [article(f"a{i}", i % 30) for i in range(60)]
        assert random_split(articles, seed=7).hash == random_split(articles, seed=7).hash

    def test_differs_by_seed(self) -> None:
        articles = [article(f"a{i}", i % 30) for i in range(60)]
        assert random_split(articles, seed=1).hash != random_split(articles, seed=2).hash

    def test_is_labelled_as_a_measuring_tool_not_a_result(self) -> None:
        # It exists to quantify the inflation. Reporting a number from it as a
        # result is exactly the mistake the protocol is meant to prevent.
        articles = [article(f"a{i}", i % 30) for i in range(60)]
        m = random_split(articles)
        assert any("never for reporting" in w for w in m.warnings)

    def test_mixes_eras_where_the_temporal_split_does_not(self) -> None:
        # The whole reason the two differ: a random split lets recent articles
        # into train, so the model is tested on the period it trained on.
        articles = [*spread(40, 20), *spread(40, 2)]

        temporal = temporal_split(articles, anchor=NOW)
        shuffled = random_split(articles, seed=42)

        def newest_in_train(m: object) -> float:
            keys = {k for k, s in m.assignment.items() if s == "train"}  # type: ignore[attr-defined]
            ages = [(NOW - a.published_at).days for a in articles if a.article_key in keys]
            return min(ages) if ages else 999

        assert newest_in_train(temporal) >= 14
        assert newest_in_train(shuffled) < 14
