"""Saving a fitted candidate, with its preprocessing inside it.

The temptation is to save the model and rebuild the preprocessing at load time
from the current code. That works until the day someone improves `to_text` — and
then the serving process silently applies today's preprocessing to a model fitted
on last month's, and every prediction shifts a little. Nothing errors. Accuracy
degrades, the warehouse metrics still look plausible, and the cause is invisible
because both halves are individually correct.

So the artifact records the **version and content hash** of the feature code it
was fitted with, and loading refuses when they no longer match. A refusal at load
is recoverable in a minute; silent drift is not recoverable at all, because
nobody knows it happened.

Pickle is used for the sklearn estimator, which is what sklearn supports. It is
therefore only ever loaded from paths this project wrote — a constraint the
loader enforces rather than documents.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kanal.features import text as feature_module

# Bumped by hand whenever the meaning of the feature path changes in a way that
# invalidates fitted models. The content hash below catches accidental changes;
# this catches deliberate ones where a human knows the models must be refitted.
FEATURE_VERSION = 1


def feature_code_hash() -> str:
    """SHA-256 of the feature module's source.

    Catches the case the version number cannot: someone edits `to_text` and does
    not think to bump anything. Source rather than behaviour, so it is
    conservative — a comment change invalidates artifacts too, which is the
    correct direction to be wrong in.
    """
    source = inspect.getsource(feature_module)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@dataclass
class ArtifactMeta:
    """Everything needed to decide whether this model may be trusted, and served."""

    name: str
    created_at: str
    feature_version: int
    feature_hash: str
    split_hash: str
    config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    git_commit: str | None = None
    notes: str = ""

    @property
    def id(self) -> str:
        """Content address. Two artifacts with the same id are the same model."""
        payload = json.dumps(
            {
                "name": self.name,
                "feature_hash": self.feature_hash,
                "split_hash": self.split_hash,
                "config": self.config,
                "created_at": self.created_at,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Artifact:
    """A fitted model plus the provenance that makes it usable."""

    model: Any
    meta: ArtifactMeta

    def predict(self, texts: list[str]) -> list[Any]:
        return list(self.model.predict(texts))


class FeatureMismatch(RuntimeError):
    """Raised when an artifact was fitted with different feature code.

    Deliberately fatal. The alternative is a serving process quietly applying
    today's preprocessing to a model fitted on last month's — which produces
    wrong answers rather than an error, and leaves nothing to find.
    """


def save(
    model: Any,
    path: Path,
    *,
    split_hash: str,
    metrics: dict[str, float] | None = None,
    git_commit: str | None = None,
    notes: str = "",
) -> ArtifactMeta:
    """Write a fitted candidate and its provenance to `path`."""
    if not hasattr(model, "predict") or not hasattr(model, "describe"):
        raise TypeError(f"{type(model).__name__} does not satisfy the Candidate protocol")

    config = model.describe()
    meta = ArtifactMeta(
        name=getattr(model, "name", type(model).__name__),
        created_at=datetime.now(tz=UTC).isoformat(),
        feature_version=FEATURE_VERSION,
        feature_hash=feature_code_hash(),
        split_hash=split_hash,
        config=config,
        metrics=dict(metrics or {}),
        classes=list(config.get("classes", [])),
        git_commit=git_commit,
        notes=notes,
    )

    path.mkdir(parents=True, exist_ok=True)
    (path / "model.pkl").write_bytes(pickle.dumps(model))
    (path / "meta.json").write_text(
        json.dumps(asdict(meta) | {"id": meta.id}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return meta


def load(path: Path, *, allow_feature_mismatch: bool = False) -> Artifact:
    """Load an artifact, refusing one fitted with different feature code.

    `allow_feature_mismatch` exists for exactly one caller: the tooling that
    inspects historical artifacts to explain why they were rejected. It is not
    for serving, and the serving path does not pass it.
    """
    meta_path = path / "meta.json"
    model_path = path / "model.pkl"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"{path} is not an artifact directory")

    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw.pop("id", None)
    meta = ArtifactMeta(**raw)

    if not allow_feature_mismatch:
        current = feature_code_hash()
        if meta.feature_version != FEATURE_VERSION:
            raise FeatureMismatch(
                f"artifact {meta.id} was fitted with feature version "
                f"{meta.feature_version}, this code is version {FEATURE_VERSION}. "
                f"Refit before serving — applying today's preprocessing to a model "
                f"fitted on another version produces wrong answers, not errors."
            )
        if meta.feature_hash != current:
            raise FeatureMismatch(
                f"artifact {meta.id} was fitted with feature code "
                f"{meta.feature_hash[:12]}, this process has {current[:12]}. "
                f"The feature path changed since this model was fitted; refit it."
            )

    return Artifact(model=pickle.loads(model_path.read_bytes()), meta=meta)
