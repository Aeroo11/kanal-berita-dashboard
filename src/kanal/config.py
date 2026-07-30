"""Runtime configuration.

Everything that a deployment might reasonably want to change lives here and
reads from the environment, so the same code runs on a laptop, in CI, and in a
scheduled Action without edits.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_root() -> Path:
    """The project root, found by walking up from this file."""
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KANAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: Path = _repo_root() / "data"
    """Root of the landing zone. Partitioned JSONL lands under `raw/`."""

    # ── Polling policy ───────────────────────────────────────────────────
    # These are a courtesy contract with the publishers, not tuning knobs.
    # See DATA.md. Feeds are published for syndication; the way to stay
    # welcome is to behave like a well-mannered consumer.
    user_agent: str = (
        "kanal-research/0.1 (+https://github.com/Aeroo11/kanal-berita-dashboard) "
        "Indonesian news classification research; contact via GitHub issues"
    )
    request_timeout_s: float = 20.0
    """Per-request timeout. Feeds that hang must not stall the whole cycle."""

    inter_request_delay_s: float = 1.0
    """Pause between requests. One feed at a time, never parallel fan-out."""

    max_retries: int = 3
    backoff_base_s: float = 2.0

    # ── Environment-specific expectations ────────────────────────────────
    expect_unreachable: str = ""
    """Comma-separated sources known to be unreachable from *this* environment.

    Declared at the deployment boundary rather than in the source registry,
    because reachability is a property of where the code runs, not of the
    publisher. CNN and Tempo sit behind Cloudflare and answer 403 to GitHub's
    datacentre IPs while accepting any request — even one with an empty
    User-Agent — from an Indonesian address.

    A source listed here is still polled and still reported; it just does not
    make the cycle unhealthy. That distinction matters: a run left permanently
    red for a known, understood, unfixable reason stops being a signal, and the
    next real failure arrives to an audience that has learnt to ignore it.

    Deliberately empty by default, so a local run treats a CNN failure as the
    problem it would be there.
    """

    @property
    def unreachable_sources(self) -> frozenset[str]:
        return frozenset(s.strip() for s in self.expect_unreachable.split(",") if s.strip())

    # ── Landing zone ─────────────────────────────────────────────────────
    landing_lookback_days: int = 3
    """How many previous day-partitions to scan when deduplicating.

    Partitions are per UTC day, so scanning only today's would make every
    article still sitting in a feed look unseen at midnight — measured at 39.1%
    redundant lines before this existed. A feed advertises its last 25–100
    items, so three days covers everything a publisher is still offering while
    keeping the scan bounded.
    """

    # ── Circuit breaker ──────────────────────────────────────────────────
    # A source that is failing should be dropped for a while rather than
    # retried every cycle. Detik and Tempo are known to be flaky; the point
    # is that one bad source never costs us the others.
    breaker_failure_threshold: int = 3
    breaker_cooldown_s: float = 3600.0

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def state_path(self) -> Path:
        """Per-feed HTTP caching state (ETag / Last-Modified) and breaker state."""
        return self.data_dir / "state" / "feeds.json"


settings = Settings()
