"""The registry: which artifact serves, and the log of how it got there.

Deliberately a directory of JSON plus an alias file rather than MLflow. The
registry's whole job here is three operations — resolve an alias, move an alias,
append to a log — and each is a few lines. Adding a service to perform them would
put a process between the API and the decision it depends on, for no capability
this project uses.

That trade is recorded in ADR-004 alongside what is given up: no experiment UI,
no artifact browser, no parameter search. If those become the bottleneck, the
migration is mechanical, because everything below is content-addressed.

## Aliases, not paths

The API resolves `champion` on a timer instead of loading a path at boot. Moving
an alias is therefore a complete rollback: no redeploy, no restart, and the
running process picks it up within one poll interval. A registry where rollback
requires a deploy is a registry that will not be used during an incident.

## The log records refusals

`decisions.jsonl` appends every gate outcome, not only promotions. A promotion
log containing nothing but successes reads as a system that has never once said
no — which is either untrue or much worse than it sounds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from kanal.registry.artifact import Artifact, load
from kanal.registry.promote import GateResult, Verdict

CHAMPION = "champion"
CHALLENGER = "challenger"
PREVIOUS = "previous"


class AliasNotSet(RuntimeError):
    """Raised when an alias has never pointed at anything."""


@dataclass
class Decision:
    """One gate outcome, promoted or not."""

    at: str
    verdict: str
    challenger_id: str
    challenger_name: str
    champion_id: str | None
    split_hash: str
    macro_f1: float
    reasons: list[str]
    passed: list[str]

    def summary(self) -> str:
        head = f"{self.at[:19]}  {self.verdict:<8} {self.challenger_name} ({self.challenger_id})"
        return head + "".join(f"\n    {r}" for r in self.reasons)


class Registry:
    """A directory of artifacts, a set of aliases, and an append-only log."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifacts = root / "artifacts"
        self.aliases = root / "aliases"
        self.log_path = root / "decisions.jsonl"

    def _alias_file(self, alias: str) -> Path:
        return self.aliases / f"{alias}.json"

    def register(self, artifact_dir: Path, artifact_id: str) -> Path:
        """Take an artifact directory into the registry under its content id."""
        self.artifacts.mkdir(parents=True, exist_ok=True)
        target = self.artifacts / artifact_id
        if target.exists():
            # Content-addressed: the same id is the same model, so re-registering
            # is a no-op rather than an error. Ingestion learnt this lesson
            # already — an operation that is not idempotent cannot be retried.
            return target
        target.mkdir(parents=True)
        for item in artifact_dir.iterdir():
            target.joinpath(item.name).write_bytes(item.read_bytes())
        return target

    def set_alias(self, alias: str, artifact_id: str) -> None:
        if not (self.artifacts / artifact_id).exists():
            raise FileNotFoundError(f"no artifact {artifact_id} in the registry")
        self.aliases.mkdir(parents=True, exist_ok=True)
        self._alias_file(alias).write_text(
            json.dumps(
                {"artifact_id": artifact_id, "set_at": datetime.now(tz=UTC).isoformat()},
                indent=2,
            ),
            encoding="utf-8",
        )

    def resolve(self, alias: str) -> str:
        path = self._alias_file(alias)
        if not path.exists():
            raise AliasNotSet(f"alias {alias!r} has never been set")
        return str(json.loads(path.read_text(encoding="utf-8"))["artifact_id"])

    def alias_set_at(self, alias: str) -> str | None:
        path = self._alias_file(alias)
        if not path.exists():
            return None
        return str(json.loads(path.read_text(encoding="utf-8")).get("set_at"))

    def load_alias(self, alias: str) -> Artifact:
        return load(self.artifacts / self.resolve(alias))

    def promote(self, artifact_id: str) -> None:
        """Move `champion` to a new artifact, keeping the old one as `previous`.

        Recording the outgoing champion is what makes `rollback` a single
        operation rather than an archaeology exercise during an incident.
        """
        try:
            outgoing = self.resolve(CHAMPION)
        except AliasNotSet:
            outgoing = None

        if outgoing is not None and outgoing != artifact_id:
            self.set_alias(PREVIOUS, outgoing)
        self.set_alias(CHAMPION, artifact_id)

    def rollback(self) -> str:
        """Swap `champion` and `previous`.

        Swap rather than overwrite, so rolling back twice returns to where it
        started. A rollback that cannot itself be undone is a second outage
        waiting for someone to press it by mistake.
        """
        try:
            current = self.resolve(CHAMPION)
            previous = self.resolve(PREVIOUS)
        except AliasNotSet as err:
            raise AliasNotSet("nothing to roll back to — no previous champion") from err

        self.set_alias(CHAMPION, previous)
        self.set_alias(PREVIOUS, current)
        return previous

    def record(self, decision: Decision) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")

    def decisions(self) -> list[Decision]:
        if not self.log_path.exists():
            return []
        return [
            Decision(**json.loads(line))
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def refusals(self) -> list[Decision]:
        """Every decision that was not a promotion.

        Exposed as a first-class query because a promotion log worth reading is
        one that shows what it turned down.
        """
        return [d for d in self.decisions() if d.verdict != Verdict.PROMOTE]

    def apply(
        self,
        gate: GateResult,
        *,
        artifact_dir: Path,
        artifact_id: str,
        artifact_name: str,
        split_hash: str,
        macro_f1: float,
    ) -> Decision:
        """Record the gate's decision, and act on it only if it was PROMOTE.

        One entry point, so a decision cannot be logged without being applied or
        applied without being logged.
        """
        try:
            champion_id: str | None = self.resolve(CHAMPION)
        except AliasNotSet:
            champion_id = None

        decision = Decision(
            at=datetime.now(tz=UTC).isoformat(),
            verdict=str(gate.verdict),
            challenger_id=artifact_id,
            challenger_name=artifact_name,
            champion_id=champion_id,
            split_hash=split_hash,
            macro_f1=macro_f1,
            reasons=list(gate.reasons),
            passed=list(gate.passed),
        )

        if gate.verdict is Verdict.PROMOTE:
            self.register(artifact_dir, artifact_id)
            self.promote(artifact_id)

        self.record(decision)
        return decision
