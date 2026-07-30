"""Rollback must take effect without a redeploy.

The obvious loader reads the alias once at boot. It is simpler, faster, and it
makes rollback useless — reverting a bad promotion would need a redeploy, which
during an incident is the one thing nobody wants to be waiting on.

These tests pin the three ways a polling loader could quietly fail instead: a
failed reload evicting a working model, a torn read during the swap, and a
thundering herd of concurrent reloads.

Time is injected rather than slept through, so a sixty-second TTL costs nothing
to test.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from kanal.registry.artifact import save
from kanal.registry.store import CHAMPION, Registry
from kanal.serving.loader import ChampionLoader, NoChampion

TRAIN = [
    ("Presiden resmikan bendungan baru", "politik"),
    ("DPR sahkan undang-undang pemilu", "politik"),
    ("Harga emas menguat di pasar", "ekonomi"),
    ("Bank sentral tahan suku bunga", "ekonomi"),
    ("Timnas menang di partai final", "olahraga"),
    ("Persib lolos ke semifinal", "olahraga"),
]


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def build(tmp: Path, name: str, *, c: float = 1.0) -> tuple[Path, str]:
    from kanal.models.tfidf import TfidfLinearSVC

    model = TfidfLinearSVC(min_df=1, c=c)
    model.fit([t for t, _ in TRAIN], [k for _, k in TRAIN])
    path = tmp / name
    meta = save(model, path, split_hash="s1")
    return path, meta.id


def registry_with(tmp: Path, *names: str) -> tuple[Registry, list[str]]:
    reg = Registry(tmp / "registry")
    ids: list[str] = []
    for i, name in enumerate(names):
        path, art_id = build(tmp, name, c=1.0 + i)
        reg.register(path, art_id)
        ids.append(art_id)
    return reg, ids


class TestThePollPickingUpChanges:
    def test_a_promotion_is_picked_up_within_the_ttl(self, tmp_path: Path) -> None:
        """The property the whole design exists for.

        No restart, no redeploy — the running process notices the alias moved
        and swaps the model underneath itself.
        """
        reg, (id1, id2) = registry_with(tmp_path, "a", "b")
        reg.promote(id1)

        clock = FakeClock()
        loader = ChampionLoader(tmp_path / "registry", ttl_seconds=60.0, clock=clock)
        assert loader.get().meta.id == id1

        reg.promote(id2)

        # Still inside the TTL: the loader has not looked again.
        clock.advance(30)
        assert loader.get().meta.id == id1

        # Past it: picked up, with no restart.
        clock.advance(31)
        assert loader.get().meta.id == id2
        assert loader.state.reloads == 2

    def test_a_rollback_is_picked_up_the_same_way(self, tmp_path: Path) -> None:
        reg, (id1, id2) = registry_with(tmp_path, "a", "b")
        reg.promote(id1)
        reg.promote(id2)

        clock = FakeClock()
        loader = ChampionLoader(tmp_path / "registry", ttl_seconds=60.0, clock=clock)
        assert loader.get().meta.id == id2

        reg.rollback()
        clock.advance(61)
        assert loader.get().meta.id == id1

    def test_an_unchanged_alias_does_not_reload(self, tmp_path: Path) -> None:
        # Unpickling a model on every TTL expiry would be a slow leak of CPU for
        # a value that changes perhaps weekly.
        reg, (id1,) = registry_with(tmp_path, "a")
        reg.promote(id1)

        clock = FakeClock()
        loader = ChampionLoader(tmp_path / "registry", ttl_seconds=10.0, clock=clock)
        loader.get()
        for _ in range(5):
            clock.advance(11)
            loader.get()

        assert loader.state.reloads == 1

    def test_force_refresh_ignores_the_ttl(self, tmp_path: Path) -> None:
        # For an operator who has just rolled back and does not want to wait out
        # the interval.
        reg, (id1, id2) = registry_with(tmp_path, "a", "b")
        reg.promote(id1)

        loader = ChampionLoader(tmp_path / "registry", ttl_seconds=3600.0, clock=FakeClock())
        assert loader.get().meta.id == id1

        reg.promote(id2)
        loader.force_refresh()
        assert loader.get().meta.id == id2


class TestAFailedReloadDoesNotEvict:
    def test_a_broken_new_artifact_leaves_the_old_one_serving(self, tmp_path: Path) -> None:
        """Serving a slightly stale model beats serving nothing.

        If a newly promoted artifact will not load, the process keeps the one it
        already has. The alternative — evicting on failure — turns a bad
        promotion into an outage.
        """
        reg, (id1, id2) = registry_with(tmp_path, "a", "b")
        reg.promote(id1)

        clock = FakeClock()
        loader = ChampionLoader(tmp_path / "registry", ttl_seconds=10.0, clock=clock)
        assert loader.get().meta.id == id1

        # Corrupt the incoming artifact, then promote it.
        (reg.artifacts / id2 / "model.pkl").write_bytes(b"not a pickle")
        reg.promote(id2)

        clock.advance(11)
        assert loader.get().meta.id == id1, "the working model must keep serving"
        assert loader.state.failed_reloads == 1
        assert loader.state.last_error

    def test_a_failed_reload_is_visible_rather_than_silent(self, tmp_path: Path) -> None:
        # Keeping the old model is right; hiding that it happened is not. An
        # operator must be able to see the discrepancy in /health.
        reg, (id1, id2) = registry_with(tmp_path, "a", "b")
        reg.promote(id1)
        clock = FakeClock()
        loader = ChampionLoader(tmp_path / "registry", ttl_seconds=10.0, clock=clock)
        loader.get()

        (reg.artifacts / id2 / "model.pkl").write_bytes(b"broken")
        reg.promote(id2)
        clock.advance(11)
        loader.get()

        assert loader.state.artifact_id == id1
        assert reg.resolve(CHAMPION) == id2, "the registry and the process disagree"
        assert loader.state.last_error is not None

    def test_a_feature_mismatch_also_does_not_evict(self, tmp_path: Path) -> None:
        # The most likely real cause: someone edits to_text and redeploys before
        # refitting. The artifact refuses to load, and the running model stays.
        reg, (id1, id2) = registry_with(tmp_path, "a", "b")
        reg.promote(id1)
        clock = FakeClock()
        loader = ChampionLoader(tmp_path / "registry", ttl_seconds=10.0, clock=clock)
        loader.get()

        meta_path = reg.artifacts / id2 / "meta.json"
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        raw["feature_hash"] = "0" * 64
        meta_path.write_text(json.dumps(raw), encoding="utf-8")
        reg.promote(id2)

        clock.advance(11)
        assert loader.get().meta.id == id1
        assert "FeatureMismatch" in (loader.state.last_error or "")

    def test_recovery_happens_on_the_next_poll(self, tmp_path: Path) -> None:
        reg, (id1, id2) = registry_with(tmp_path, "a", "b")
        reg.promote(id1)
        clock = FakeClock()
        loader = ChampionLoader(tmp_path / "registry", ttl_seconds=10.0, clock=clock)
        loader.get()

        good = (reg.artifacts / id2 / "model.pkl").read_bytes()
        (reg.artifacts / id2 / "model.pkl").write_bytes(b"broken")
        reg.promote(id2)
        clock.advance(11)
        assert loader.get().meta.id == id1

        # Someone re-uploads a working artifact; no restart needed.
        (reg.artifacts / id2 / "model.pkl").write_bytes(good)
        clock.advance(11)
        assert loader.get().meta.id == id2
        assert loader.state.last_error is None


class TestConcurrency:
    def test_only_one_reload_runs_at_a_time(self, tmp_path: Path) -> None:
        """Without the lock, every request arriving after the TTL expires would
        unpickle its own copy of the same artifact."""
        reg, (id1, id2) = registry_with(tmp_path, "a", "b")
        reg.promote(id1)

        clock = FakeClock()
        loader = ChampionLoader(tmp_path / "registry", ttl_seconds=10.0, clock=clock)
        loader.get()

        reg.promote(id2)
        clock.advance(11)

        seen: list[str] = []
        barrier = threading.Barrier(8)

        def hammer() -> None:
            barrier.wait()
            seen.append(loader.get().meta.id)

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(seen) == 8
        assert loader.state.reloads == 2, "one reload, not eight"

    def test_concurrent_first_requests_all_get_the_model(self, tmp_path: Path) -> None:
        """The bug this test was written after finding.

        `_refresh` stamps `_checked_at` before the unpickle finishes. A second
        thread arriving during that first load therefore saw "not stale",
        skipped the refresh, and found no artifact — returning 503 while a
        champion existed. In production it would have hit the first few
        concurrent requests after every boot, intermittently, which is the worst
        kind of bug to be handed in an incident.

        The fix is that `get()` refreshes whenever there is no artifact, not only
        when the TTL has expired.
        """
        reg, (id1,) = registry_with(tmp_path, "a")
        reg.promote(id1)

        loader = ChampionLoader(tmp_path / "registry", ttl_seconds=60.0, clock=FakeClock())

        barrier = threading.Barrier(8)
        outcomes: list[str] = []
        lock = threading.Lock()

        def first_request() -> None:
            barrier.wait()  # all eight arrive together, before anything is loaded
            try:
                got = loader.get().meta.id
            except NoChampion as err:
                got = f"FAILED: {err}"
            with lock:
                outcomes.append(got)

        threads = [threading.Thread(target=first_request) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert outcomes == [id1] * 8, f"some first requests were refused: {outcomes}"
        assert loader.state.reloads == 1

    def test_every_request_sees_a_complete_model(self, tmp_path: Path) -> None:
        """A request in flight during a swap must see one model or the other.

        Bounded on purpose. An earlier version span `while not stop.is_set()` in
        four threads while the main thread promoted in a loop, which made the
        runtime depend on how fast Windows completes an atomic rename — and on
        this machine that was slow enough to look like a hang. An unbounded spin
        loop in a test is a hazard whatever it is testing.

        Each worker now does a fixed number of iterations and the promotions
        happen alongside them, so the test exercises the same interleaving in a
        knowable amount of work.
        """
        reg, (id1, id2) = registry_with(tmp_path, "a", "b")
        reg.promote(id1)
        clock = FakeClock()
        loader = ChampionLoader(tmp_path / "registry", ttl_seconds=1.0, clock=clock)

        ITERATIONS = 25
        results: list[str] = []
        lock = threading.Lock()
        start = threading.Barrier(5)  # four workers plus the promoter

        def predict_n() -> None:
            start.wait()
            local: list[str] = []
            for _ in range(ITERATIONS):
                artifact = loader.get()
                out = artifact.predict(["Harga emas naik"])
                # A half-built model would show up as a missing label or a
                # prediction count that does not match the input.
                local.append(out[0].label if len(out) == 1 else "TORN")
            with lock:
                results.extend(local)

        def promote_n() -> None:
            start.wait()
            for i in range(ITERATIONS):
                clock.advance(2)
                reg.promote(id2 if i % 2 else id1)

        threads = [threading.Thread(target=predict_n) for _ in range(4)]
        threads.append(threading.Thread(target=promote_n))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 4 * ITERATIONS
        assert "TORN" not in results, "a request saw a partially swapped model"
        assert all(r in {"politik", "ekonomi", "olahraga"} for r in results)


class TestNoChampionYet:
    def test_raises_a_message_that_says_what_to_do(self, tmp_path: Path) -> None:
        loader = ChampionLoader(tmp_path / "nothing", ttl_seconds=0.0)
        with pytest.raises(NoChampion, match="nothing to serve"):
            loader.get()

    def test_a_missing_alias_is_not_recorded_as_an_error(self, tmp_path: Path) -> None:
        # Never having promoted is a normal state, not a failure. Counting it as
        # one would make /health report degraded on a fresh deployment.
        loader = ChampionLoader(tmp_path / "nothing", ttl_seconds=0.0)
        with pytest.raises(NoChampion):
            loader.get()
        assert loader.state.failed_reloads == 0
        assert loader.state.last_error is None

    def test_it_starts_serving_once_something_is_promoted(self, tmp_path: Path) -> None:
        reg, (id1,) = registry_with(tmp_path, "a")
        loader = ChampionLoader(tmp_path / "registry", ttl_seconds=0.0)
        with pytest.raises(NoChampion):
            loader.get()

        reg.promote(id1)
        assert loader.get().meta.id == id1
