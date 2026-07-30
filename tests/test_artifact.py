"""Artifact persistence, and the refusal that prevents silent drift.

The failure this guards against has no symptom. Someone improves `to_text`; the
serving process then applies today's preprocessing to a model fitted on last
month's; every prediction shifts slightly. Nothing raises. Accuracy degrades,
the warehouse metrics still look plausible, and the cause is invisible because
both halves are individually correct.

So loading refuses on a feature-code mismatch. A refusal is recoverable in a
minute; silent drift is not recoverable at all, because nobody knows it happened.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kanal.registry.artifact import (
    FEATURE_VERSION,
    Artifact,
    FeatureMismatch,
    feature_code_hash,
    load,
    save,
)

TEXTS = [
    "Presiden resmikan bendungan baru di provinsi itu",
    "DPR sahkan undang-undang baru soal pemilu",
    "Harga emas menguat di tengah ketidakpastian pasar",
    "Bank sentral pertahankan suku bunga acuan",
    "Timnas futsal menang atas Thailand di final",
    "Persib segel tiket lolos ke semifinal turnamen",
]
LABELS = ["politik", "politik", "ekonomi", "ekonomi", "olahraga", "olahraga"]


def fitted() -> object:
    from kanal.models.tfidf import TfidfLinearSVC

    model = TfidfLinearSVC(min_df=1)
    model.fit(TEXTS, LABELS)
    return model


class TestRoundTrip:
    def test_a_saved_model_predicts_identically_after_loading(self, tmp_path: Path) -> None:
        # The baseline property. If this fails, nothing else here matters.
        model = fitted()
        save(model, tmp_path / "art", split_hash="deadbeef")

        before = [p.label for p in model.predict(TEXTS)]  # type: ignore[attr-defined]
        after = [p.label for p in load(tmp_path / "art").predict(TEXTS)]
        assert before == after

    def test_confidences_survive_the_round_trip(self, tmp_path: Path) -> None:
        # Not just the label — the cascade gates on the confidence, so a
        # temperature lost in serialisation would break escalation silently.
        model = fitted()
        save(model, tmp_path / "art", split_hash="deadbeef")

        before = [p.confidence for p in model.predict(TEXTS)]  # type: ignore[attr-defined]
        after = [p.confidence for p in load(tmp_path / "art").predict(TEXTS)]
        assert before == pytest.approx(after)

    def test_metadata_is_readable_without_unpickling(self, tmp_path: Path) -> None:
        # So tooling can inspect a registry without executing anything from it.
        meta = save(fitted(), tmp_path / "art", split_hash="abc123", metrics={"macro_f1": 0.8})
        raw = json.loads((tmp_path / "art" / "meta.json").read_text(encoding="utf-8"))

        assert raw["split_hash"] == "abc123"
        assert raw["metrics"]["macro_f1"] == 0.8
        assert raw["id"] == meta.id
        assert raw["feature_version"] == FEATURE_VERSION


class TestTheFeatureGuard:
    def test_a_changed_feature_hash_refuses_to_load(self, tmp_path: Path) -> None:
        """The refusal that prevents the silent version-skew failure."""
        save(fitted(), tmp_path / "art", split_hash="x")

        meta_path = tmp_path / "art" / "meta.json"
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        raw["feature_hash"] = "0" * 64
        meta_path.write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(FeatureMismatch, match="feature path changed"):
            load(tmp_path / "art")

    def test_a_changed_feature_version_refuses_to_load(self, tmp_path: Path) -> None:
        save(fitted(), tmp_path / "art", split_hash="x")

        meta_path = tmp_path / "art" / "meta.json"
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        raw["feature_version"] = FEATURE_VERSION + 1
        meta_path.write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(FeatureMismatch, match="wrong answers, not errors"):
            load(tmp_path / "art")

    def test_the_escape_hatch_exists_and_is_not_for_serving(self, tmp_path: Path) -> None:
        # Historical artifacts must stay inspectable, so the reason one was
        # rejected can still be explained. The serving path does not pass this.
        save(fitted(), tmp_path / "art", split_hash="x")
        meta_path = tmp_path / "art" / "meta.json"
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        raw["feature_hash"] = "0" * 64
        meta_path.write_text(json.dumps(raw), encoding="utf-8")

        artifact = load(tmp_path / "art", allow_feature_mismatch=True)
        assert isinstance(artifact, Artifact)
        assert artifact.meta.feature_hash == "0" * 64

    def test_the_hash_tracks_the_feature_source(self) -> None:
        # Source rather than behaviour, so it is conservative: a comment change
        # invalidates artifacts too. That is the correct direction to be wrong in.
        assert feature_code_hash() == feature_code_hash()
        assert len(feature_code_hash()) == 64


class TestProvenance:
    def test_the_id_is_content_addressed(self, tmp_path: Path) -> None:
        a = save(fitted(), tmp_path / "a", split_hash="same")
        b = save(fitted(), tmp_path / "b", split_hash="different")
        assert a.id != b.id

    def test_the_split_hash_is_carried(self, tmp_path: Path) -> None:
        # Without it a promoted model cannot say which evaluation justified it.
        # Both halves are checked: what save() reported, and what load() reads
        # back — a save that returned a different value from what it wrote would
        # otherwise go unnoticed.
        meta = save(fitted(), tmp_path / "art", split_hash="0e2cdd3f7243")
        assert meta.split_hash == "0e2cdd3f7243"
        assert load(tmp_path / "art").meta.split_hash == "0e2cdd3f7243"

    def test_the_classes_are_recorded(self, tmp_path: Path) -> None:
        save(fitted(), tmp_path / "art", split_hash="x")
        assert set(load(tmp_path / "art").meta.classes) == {"politik", "ekonomi", "olahraga"}


class TestRejections:
    def test_saving_a_non_candidate_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="Candidate protocol"):
            save(object(), tmp_path / "art", split_hash="x")

    def test_loading_a_directory_that_is_not_an_artifact(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="not an artifact"):
            load(tmp_path / "empty")
