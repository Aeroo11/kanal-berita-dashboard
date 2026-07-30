"""The registry: aliases, rollback, and a log that records refusals.

The properties that matter during an incident, not during a demo:

- rollback is one operation and is itself reversible
- an alias move is a complete rollback, needing no redeploy
- the log contains what was turned down, not only what was accepted
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kanal.registry.artifact import save
from kanal.registry.promote import GateResult, Verdict
from kanal.registry.store import CHAMPION, PREVIOUS, AliasNotSet, Decision, Registry

TEXTS = [
    "Presiden resmikan bendungan baru di provinsi itu",
    "DPR sahkan undang-undang baru soal pemilu",
    "Harga emas menguat di tengah ketidakpastian pasar",
    "Bank sentral pertahankan suku bunga acuan",
    "Timnas futsal menang atas Thailand di final",
    "Persib segel tiket lolos ke semifinal turnamen",
]
LABELS = ["politik", "politik", "ekonomi", "ekonomi", "olahraga", "olahraga"]


def make_artifact(tmp: Path, name: str, *, c: float = 1.0) -> tuple[Path, str]:
    from kanal.models.tfidf import TfidfLinearSVC

    model = TfidfLinearSVC(min_df=1, c=c)
    model.fit(TEXTS, LABELS)
    path = tmp / name
    meta = save(model, path, split_hash="split-1")
    return path, meta.id


def decision(verdict: Verdict, artifact_id: str, reasons: list[str] | None = None) -> Decision:
    return Decision(
        at="2026-07-30T12:00:00+00:00",
        verdict=str(verdict),
        challenger_id=artifact_id,
        challenger_name="tfidf-linearsvc",
        champion_id=None,
        split_hash="split-1",
        macro_f1=0.8,
        reasons=reasons or [],
        passed=[],
    )


class TestAliases:
    def test_an_unset_alias_raises_rather_than_returning_none(self, tmp_path: Path) -> None:
        # A None here would become a crash somewhere further away, at the point
        # the API tries to serve with no model.
        with pytest.raises(AliasNotSet, match="never been set"):
            Registry(tmp_path).resolve(CHAMPION)

    def test_setting_an_alias_to_an_unknown_artifact_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no artifact"):
            Registry(tmp_path).set_alias(CHAMPION, "does-not-exist")

    def test_registering_is_idempotent(self, tmp_path: Path) -> None:
        # Content-addressed, so the same id is the same model. Ingestion learnt
        # this already: an operation that is not idempotent cannot be retried.
        reg = Registry(tmp_path / "reg")
        path, art_id = make_artifact(tmp_path, "a")
        first = reg.register(path, art_id)
        second = reg.register(path, art_id)
        assert first == second

    def test_a_registered_artifact_loads_through_its_alias(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg")
        path, art_id = make_artifact(tmp_path, "a")
        reg.register(path, art_id)
        reg.set_alias(CHAMPION, art_id)

        artifact = reg.load_alias(CHAMPION)
        assert artifact.meta.id == art_id
        assert artifact.predict(["Harga emas naik"])


class TestPromotionAndRollback:
    def test_promoting_keeps_the_outgoing_champion_as_previous(self, tmp_path: Path) -> None:
        # What makes rollback one operation instead of archaeology during an
        # incident.
        reg = Registry(tmp_path / "reg")
        p1, id1 = make_artifact(tmp_path, "a", c=1.0)
        p2, id2 = make_artifact(tmp_path, "b", c=2.0)
        reg.register(p1, id1)
        reg.register(p2, id2)

        reg.promote(id1)
        reg.promote(id2)

        assert reg.resolve(CHAMPION) == id2
        assert reg.resolve(PREVIOUS) == id1

    def test_rollback_swaps_rather_than_overwrites(self, tmp_path: Path) -> None:
        """Rolling back twice must return to where it started.

        A rollback that cannot itself be undone is a second outage waiting for
        someone to press it by mistake.
        """
        reg = Registry(tmp_path / "reg")
        p1, id1 = make_artifact(tmp_path, "a", c=1.0)
        p2, id2 = make_artifact(tmp_path, "b", c=2.0)
        reg.register(p1, id1)
        reg.register(p2, id2)
        reg.promote(id1)
        reg.promote(id2)

        assert reg.rollback() == id1
        assert reg.resolve(CHAMPION) == id1

        assert reg.rollback() == id2
        assert reg.resolve(CHAMPION) == id2

    def test_rollback_with_no_previous_champion_says_so(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg")
        path, art_id = make_artifact(tmp_path, "a")
        reg.register(path, art_id)
        reg.promote(art_id)

        with pytest.raises(AliasNotSet, match="nothing to roll back to"):
            reg.rollback()

    def test_promoting_the_same_artifact_twice_does_not_self_reference(
        self, tmp_path: Path
    ) -> None:
        # Otherwise previous == champion, and a later rollback becomes a no-op
        # that looks like it worked.
        reg = Registry(tmp_path / "reg")
        path, art_id = make_artifact(tmp_path, "a")
        reg.register(path, art_id)
        reg.promote(art_id)
        reg.promote(art_id)

        assert reg.resolve(CHAMPION) == art_id
        with pytest.raises(AliasNotSet):
            reg.resolve(PREVIOUS)

    def test_an_alias_move_needs_no_redeploy(self, tmp_path: Path) -> None:
        # The property the API depends on: the alias is a file, and the serving
        # process re-reads it. Nothing about rollback touches the process.
        reg = Registry(tmp_path / "reg")
        p1, id1 = make_artifact(tmp_path, "a", c=1.0)
        p2, id2 = make_artifact(tmp_path, "b", c=2.0)
        reg.register(p1, id1)
        reg.register(p2, id2)
        reg.promote(id1)

        before = reg.resolve(CHAMPION)
        reg.promote(id2)
        assert reg.resolve(CHAMPION) != before


class TestTheLogRecordsRefusals:
    def test_refusals_are_logged_not_only_promotions(self, tmp_path: Path) -> None:
        """A promotion log containing only successes reads as a system that has
        never once said no."""
        reg = Registry(tmp_path / "reg")
        reg.record(decision(Verdict.PROMOTE, "aaa"))
        reg.record(decision(Verdict.HOLD, "bbb", ["G3 quality: CI lower bound -0.004"]))
        reg.record(decision(Verdict.REJECT, "ccc", ["G2 integrity: cluster leak"]))

        assert len(reg.decisions()) == 3
        assert len(reg.refusals()) == 2
        assert {d.verdict for d in reg.refusals()} == {"HOLD", "REJECT"}

    def test_a_refusal_carries_its_reason(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg")
        reg.record(decision(Verdict.HOLD, "bbb", ["G3 quality: does not clear the MDE"]))
        assert "does not clear the MDE" in reg.refusals()[0].summary()

    def test_the_log_is_append_only_and_line_delimited(self, tmp_path: Path) -> None:
        # So a crash mid-write loses one entry rather than the file.
        reg = Registry(tmp_path / "reg")
        for i in range(3):
            reg.record(decision(Verdict.HOLD, f"id{i}", ["reason"]))

        lines = reg.log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        assert all(json.loads(line)["verdict"] == "HOLD" for line in lines)

    def test_an_empty_log_reads_as_empty_not_as_an_error(self, tmp_path: Path) -> None:
        assert Registry(tmp_path / "reg").decisions() == []


class TestApplyIsTheOnlyEntryPoint:
    def test_a_promote_verdict_registers_and_moves_the_alias(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg")
        path, art_id = make_artifact(tmp_path, "a")

        reg.apply(
            GateResult(verdict=Verdict.PROMOTE, passed=["G3 quality"]),
            artifact_dir=path,
            artifact_id=art_id,
            artifact_name="tfidf",
            split_hash="split-1",
            macro_f1=0.85,
        )

        assert reg.resolve(CHAMPION) == art_id
        assert reg.decisions()[0].verdict == "PROMOTE"

    def test_a_hold_verdict_logs_but_changes_nothing(self, tmp_path: Path) -> None:
        """The important half. A refused challenger must leave no trace in what
        serves — only in what was decided."""
        reg = Registry(tmp_path / "reg")
        p1, id1 = make_artifact(tmp_path, "a", c=1.0)
        reg.register(p1, id1)
        reg.promote(id1)

        p2, id2 = make_artifact(tmp_path, "b", c=2.0)
        reg.apply(
            GateResult(verdict=Verdict.HOLD, reasons=["G3 quality: inside the noise"]),
            artifact_dir=p2,
            artifact_id=id2,
            artifact_name="tfidf",
            split_hash="split-1",
            macro_f1=0.81,
        )

        assert reg.resolve(CHAMPION) == id1, "a refused challenger must not serve"
        assert not (reg.artifacts / id2).exists(), "nor be registered"
        assert reg.refusals()[0].challenger_id == id2

    def test_the_decision_names_the_champion_it_was_measured_against(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg")
        p1, id1 = make_artifact(tmp_path, "a", c=1.0)
        reg.register(p1, id1)
        reg.promote(id1)

        p2, id2 = make_artifact(tmp_path, "b", c=2.0)
        d = reg.apply(
            GateResult(verdict=Verdict.HOLD, reasons=["nope"]),
            artifact_dir=p2,
            artifact_id=id2,
            artifact_name="tfidf",
            split_hash="split-1",
            macro_f1=0.5,
        )
        assert d.champion_id == id1
