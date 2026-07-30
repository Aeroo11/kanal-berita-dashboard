"""Resolving the champion alias on a timer, rather than locking it at boot.

The obvious implementation loads the model once at startup and keeps it. It is
simpler, it is faster, and it makes rollback useless: reverting a bad promotion
would need a redeploy, which during an incident is the one thing nobody wants to
be waiting on.

So the alias is re-read on a TTL. Moving it is a complete rollback — the running
process notices within one interval and swaps the model underneath itself, with
no restart and no dropped request.

Three properties this has to get right, and each is a way it could quietly fail:

**A failed reload must not take down a working model.** If a newly promoted
artifact will not load, the process keeps serving the one it already has and says
so in `/health`. Serving a slightly stale model beats serving nothing.

**The swap must be atomic from a request's point of view.** A request that begins
during a reload sees either the old model or the new one, never a half-built
state. The reference is replaced in one assignment; nothing mutates in place.

**Only one reload runs at a time.** Without a lock, ten concurrent requests
arriving after the TTL expires would unpickle ten copies of the same artifact.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kanal.registry.artifact import Artifact
from kanal.registry.store import CHAMPION, AliasNotSet, Registry

# How stale the alias may be. Sixty seconds is the number the README promises for
# rollback, and it is a deliberate trade: shorter means more filesystem reads for
# a value that changes perhaps weekly.
DEFAULT_TTL_SECONDS = 60.0


class NoChampion(RuntimeError):
    """No model has ever been promoted, so there is nothing to serve."""


@dataclass
class LoaderState:
    """What the loader is currently serving, and how that is going.

    Exposed through `/health` so an operator can see whether a promotion
    actually took effect, rather than inferring it from prediction quality.
    """

    artifact_id: str | None
    loaded_at: float | None
    alias_checked_at: float | None
    reloads: int
    failed_reloads: int
    last_error: str | None


class ChampionLoader:
    """Holds the current champion, re-resolving the alias on a TTL."""

    def __init__(
        self,
        registry_root: Path,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.registry = Registry(registry_root)
        self.ttl = ttl_seconds
        # Injectable so the tests can advance time instead of sleeping through
        # a sixty-second TTL. Monotonic rather than wall-clock: an NTP step
        # backwards must not make the alias look permanently fresh.
        self._now: Callable[[], float] = clock or time.monotonic

        self._lock = threading.Lock()
        self._artifact: Artifact | None = None
        self._artifact_id: str | None = None
        self._loaded_at: float | None = None
        self._checked_at: float | None = None
        self._reloads = 0
        self._failed = 0
        self._last_error: str | None = None

    @property
    def state(self) -> LoaderState:
        return LoaderState(
            artifact_id=self._artifact_id,
            loaded_at=self._loaded_at,
            alias_checked_at=self._checked_at,
            reloads=self._reloads,
            failed_reloads=self._failed,
            last_error=self._last_error,
        )

    def _stale(self) -> bool:
        if self._checked_at is None:
            return True
        return (self._now() - self._checked_at) >= self.ttl

    def _refresh(self) -> None:
        """Re-resolve the alias and swap the model if it changed.

        Holds the lock, so concurrent requests do not each unpickle a copy.
        """
        with self._lock:
            # Another thread may have refreshed while this one waited.
            if not self._stale() and self._artifact is not None:
                return

            self._checked_at = self._now()

            try:
                target = self.registry.resolve(CHAMPION)
            except AliasNotSet:
                # Not an error state to record — there has simply never been a
                # champion. `get()` turns this into a clear 503.
                return

            if target == self._artifact_id and self._artifact is not None:
                return

            try:
                loaded = self.registry.load_alias(CHAMPION)
            except Exception as err:
                # A newly promoted artifact that will not load must not take down
                # the model already serving. Serving a slightly stale champion
                # beats serving nothing, and /health reports the discrepancy.
                self._failed += 1
                self._last_error = f"{type(err).__name__}: {err}"
                return

            # One assignment, so a request in flight sees the old artifact or the
            # new one and never a half-built state.
            self._artifact = loaded
            self._artifact_id = target
            self._loaded_at = self._now()
            self._reloads += 1
            self._last_error = None

    def get(self) -> Artifact:
        """The current champion, refreshing first if the alias is stale.

        The `_artifact is None` half of this condition is load-bearing, and its
        absence was a real bug. `_refresh` stamps `_checked_at` before the
        unpickle finishes, so a second thread arriving during the very first load
        saw "not stale", skipped the refresh, and found `_artifact` still unset —
        returning 503 while a champion existed. In production that would have hit
        the first few concurrent requests after every boot.

        Refreshing whenever there is no artifact sends that thread into
        `_refresh`, where it blocks on the lock until the first load completes
        and then returns the loaded model.
        """
        if self._artifact is None or self._stale():
            self._refresh()

        if self._artifact is None:
            raise NoChampion(
                "no model has been promoted — the champion alias is unset, so "
                "there is nothing to serve"
            )
        return self._artifact

    def force_refresh(self) -> None:
        """Re-resolve immediately, ignoring the TTL.

        For the tests, and for an operator who has just rolled back and does not
        want to wait out the interval.
        """
        self._checked_at = None
        self._refresh()
