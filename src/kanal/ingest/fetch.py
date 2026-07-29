"""Polite, resilient feed fetching.

Three behaviours matter more than speed here:

**Conditional requests.** A feed polled hourly is unchanged most of the time.
Sending `If-None-Match` / `If-Modified-Since` lets the publisher answer `304`
with no body — cheaper for them, faster for us, and the single clearest signal
that we are a well-behaved consumer.

**Failing one source must not cost the others.** Detik resets connections and
Tempo returns 403 to unknown user agents. A cycle that dies on the first bad
feed loses the good ones too, and RSS is a sliding window — what is missed is
gone. So every feed is isolated, and a persistently failing source is tripped
out for a cooldown rather than retried every cycle.

**No parallel fan-out.** Requests are sequential with a delay between them. The
whole cycle takes under a minute and there is no deadline to beat; hammering a
publisher to save thirty seconds is how a research project gets blocked.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from kanal.config import settings
from kanal.ingest.sources import Feed

log = logging.getLogger(__name__)

# Retried: transient by nature. 403/404 are not here on purpose — they are
# verdicts about who we are or what we asked for, and retrying is just noise.
_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass
class FeedState:
    """What we remember about a feed between runs."""

    etag: str | None = None
    last_modified: str | None = None
    consecutive_failures: int = 0
    breaker_open_until: float = 0.0
    last_success_ts: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "etag": self.etag,
            "last_modified": self.last_modified,
            "consecutive_failures": self.consecutive_failures,
            "breaker_open_until": self.breaker_open_until,
            "last_success_ts": self.last_success_ts,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FeedState:
        return cls(
            etag=raw.get("etag"),
            last_modified=raw.get("last_modified"),
            consecutive_failures=int(raw.get("consecutive_failures", 0)),
            breaker_open_until=float(raw.get("breaker_open_until", 0.0)),
            last_success_ts=raw.get("last_success_ts"),
        )


class StateStore:
    """Feed state, persisted as one small JSON file.

    A database would be overkill for a few dozen rows that only this process
    touches, and a plain file is inspectable in the repo when something looks
    wrong.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.state_path
        self._states: dict[str, FeedState] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Corrupt state is recoverable: we lose conditional-request headers
            # and re-download once. Losing the run would be worse.
            log.warning("feed state at %s unreadable; starting fresh", self.path)
            return
        self._states = {k: FeedState.from_dict(v) for k, v in raw.items()}

    def get(self, feed_id: str) -> FeedState:
        return self._states.setdefault(feed_id, FeedState())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v.to_dict() for k, v in self._states.items()}
        # Write-then-replace: a crash mid-write must not leave a truncated file.
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


@dataclass
class FetchResult:
    """Outcome of polling one feed."""

    feed: Feed
    status: str
    """`ok` | `not_modified` | `skipped_breaker` | `failed`"""

    body: bytes | None = None
    http_status: int | None = None
    error: str | None = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def has_content(self) -> bool:
        return self.status == "ok" and bool(self.body)


class FeedFetcher:
    """Sequential, conditional, circuit-broken feed fetching."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        state: StateStore | None = None,
        *,
        sleep: bool = True,
    ) -> None:
        self._owns_client = client is None
        self.client = client or httpx.Client(
            headers={
                "User-Agent": settings.user_agent,
                "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=settings.request_timeout_s,
            follow_redirects=True,
        )
        self.state = state or StateStore()
        self._sleep = sleep

    def __enter__(self) -> FeedFetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.state.save()
        if self._owns_client:
            self.client.close()

    def fetch(self, feed: Feed) -> FetchResult:
        st = self.state.get(feed.feed_id)
        now = time.time()

        if now < st.breaker_open_until:
            remaining = int(st.breaker_open_until - now)
            log.info("breaker open for %s, %ss remaining", feed.feed_id, remaining)
            return FetchResult(feed, "skipped_breaker", error=f"cooldown {remaining}s")

        headers: dict[str, str] = {}
        if st.etag:
            headers["If-None-Match"] = st.etag
        if st.last_modified:
            headers["If-Modified-Since"] = st.last_modified

        last_error: str | None = None
        for attempt in range(settings.max_retries):
            try:
                resp = self.client.get(feed.url, headers=headers)
            except httpx.HTTPError as exc:
                # Connection reset, timeout, DNS — Detik's signature failure.
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("%s attempt %d: %s", feed.feed_id, attempt + 1, last_error)
                self._backoff(attempt)
                continue

            if resp.status_code == 304:
                self._record_success(st)
                return FetchResult(feed, "not_modified", http_status=304)

            if resp.status_code in _RETRY_STATUS:
                last_error = f"HTTP {resp.status_code}"
                # Honour Retry-After when the server bothers to send one.
                self._backoff(attempt, retry_after=resp.headers.get("Retry-After"))
                continue

            if resp.is_success:
                st.etag = resp.headers.get("ETag")
                st.last_modified = resp.headers.get("Last-Modified")
                self._record_success(st)
                return FetchResult(feed, "ok", body=resp.content, http_status=resp.status_code)

            # 403 (Tempo), 404, anything else: a verdict, not a hiccup.
            last_error = f"HTTP {resp.status_code}"
            break

        self._record_failure(st, feed)
        return FetchResult(feed, "failed", error=last_error)

    def fetch_all(self, feeds: tuple[Feed, ...]) -> list[FetchResult]:
        results: list[FetchResult] = []
        for i, feed in enumerate(feeds):
            results.append(self.fetch(feed))
            if self._sleep and i < len(feeds) - 1:
                time.sleep(settings.inter_request_delay_s)
        return results

    # ── internals ────────────────────────────────────────────────────────

    def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        if not self._sleep:
            return
        delay = settings.backoff_base_s * (2**attempt)
        if retry_after:
            # Retry-After may be seconds or an HTTP-date. Only the numeric form
            # is worth parsing; for the date form the exponential delay is a
            # good enough approximation and not worth a dependency.
            with suppress(ValueError):
                delay = max(delay, float(retry_after))
        time.sleep(min(delay, 60.0))

    @staticmethod
    def _record_success(st: FeedState) -> None:
        st.consecutive_failures = 0
        st.breaker_open_until = 0.0
        st.last_success_ts = time.time()

    @staticmethod
    def _record_failure(st: FeedState, feed: Feed) -> None:
        st.consecutive_failures += 1
        if st.consecutive_failures >= settings.breaker_failure_threshold:
            st.breaker_open_until = time.time() + settings.breaker_cooldown_s
            log.warning(
                "breaker tripped for %s after %d failures; cooling down %.0fs",
                feed.feed_id,
                st.consecutive_failures,
                settings.breaker_cooldown_s,
            )
