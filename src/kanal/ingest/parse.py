"""Turn feed bytes into article records.

The raw layer keeps the publisher's own fields verbatim alongside the derived
ones. That is deliberate: feeds change shape without warning — Liputan6 emits
`<category>`, ANTARA does not; date formats differ; a publisher may start
sending `media:content` next month. If parsing only kept the fields we
currently understand, the day a feed changed would be a day of data silently
thrown away.

So: store everything, derive what we can, record a `_schema_version`, and let
the warehouse layer decide what to make of it. A parse failure on one item is
recorded and skipped, never fatal for the batch.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import feedparser

from kanal.ingest.normalize import article_key, canonical_url, clean_text, title_fingerprint
from kanal.ingest.sources import Feed

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(slots=True)
class Article:
    """One article as it lands in the raw zone."""

    # ── identity ─────────────────────────────────────────────────────────
    article_key: str
    canonical_url: str
    title_fingerprint: str

    # ── content (the only fields a model may ever see) ───────────────────
    title: str
    summary: str

    # ── label and provenance ─────────────────────────────────────────────
    # `kanal` is the label. `source`, `channel` and `raw_link` identify where
    # it came from — useful for analysis and auditing, and forbidden as
    # features (see tests/test_leakage.py).
    kanal: str
    source: str
    channel: str
    feed_id: str
    raw_link: str

    # ── timing ───────────────────────────────────────────────────────────
    published_at: str | None
    fetched_at: str

    # ── forward compatibility ────────────────────────────────────────────
    schema_version: int = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)
    """Whatever else the publisher sent. Never dropped, never yet trusted."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParseReport:
    """What a parse produced, including what it could not."""

    articles: list[Article]
    total_entries: int
    failures: int
    failure_samples: list[str] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failures / self.total_entries if self.total_entries else 0.0


# Publisher fields we never carry forward into `extra`, because they either
# duplicate a first-class column or leak the label.
_EXCLUDED_KEYS = frozenset(
    {
        "title",
        "title_detail",
        "summary",
        "summary_detail",
        "link",
        "links",
        "id",
        "guidislink",
        "published",
        "published_parsed",
        "updated",
        "updated_parsed",
        # Label leakage: an item-level category *is* the answer.
        "tags",
        "category",
    }
)


def _published_at(entry: Any) -> str | None:
    """Best-effort publication timestamp, normalised to UTC ISO-8601.

    feedparser already handles RFC-822 (`+0700`, as ANTARA sends) and ISO
    variants. When it cannot, we return None rather than guessing — a wrong
    timestamp would corrupt the temporal split, which is the one thing the
    evaluation depends on being right.
    """
    for key in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, key, None)
        if not parsed:
            continue
        try:
            # feedparser normalises to UTC and hands back a 9-tuple. Indexing
            # the six fields explicitly, rather than unpacking with `*`, keeps
            # the arity visible — an unpack here silently spills the trailing
            # fields into `tzinfo` if the tuple is ever a different shape.
            return datetime(
                year=int(parsed[0]),
                month=int(parsed[1]),
                day=int(parsed[2]),
                hour=int(parsed[3]),
                minute=int(parsed[4]),
                second=int(parsed[5]),
                tzinfo=UTC,
            ).isoformat()
        except (TypeError, ValueError, IndexError):
            continue
    return None


def parse_feed(feed: Feed, body: bytes, fetched_at: datetime) -> ParseReport:
    """Parse one feed's bytes into articles."""
    parsed = feedparser.parse(body)

    if parsed.bozo and not parsed.entries:
        # Malformed *and* empty — nothing salvageable.
        reason = str(getattr(parsed, "bozo_exception", "unknown"))
        log.warning("%s: unparseable feed (%s)", feed.feed_id, reason)
        return ParseReport([], 0, 1, [reason[:200]])

    articles: list[Article] = []
    failures = 0
    samples: list[str] = []
    seen: set[str] = set()
    stamp = fetched_at.isoformat()

    for entry in parsed.entries:
        try:
            link = (getattr(entry, "link", "") or "").strip()
            title = clean_text(getattr(entry, "title", ""))
            if not link or not title:
                raise ValueError("entry missing link or title")

            key = article_key(link)
            # A feed occasionally repeats an item within one response.
            if key in seen:
                continue
            seen.add(key)

            extra = {
                k: v
                for k, v in entry.items()
                if k not in _EXCLUDED_KEYS and isinstance(v, str | int | float | bool)
            }

            articles.append(
                Article(
                    article_key=key,
                    canonical_url=canonical_url(link),
                    title_fingerprint=title_fingerprint(title),
                    title=title,
                    summary=clean_text(getattr(entry, "summary", "")),
                    kanal=str(feed.kanal),
                    source=feed.source,
                    channel=feed.channel,
                    feed_id=feed.feed_id,
                    raw_link=link,
                    published_at=_published_at(entry),
                    fetched_at=stamp,
                    extra=extra,
                )
            )
        except (ValueError, AttributeError, TypeError) as exc:
            failures += 1
            if len(samples) < 3:
                samples.append(f"{type(exc).__name__}: {exc}"[:200])

    if failures:
        log.info("%s: %d/%d entries failed to parse", feed.feed_id, failures, len(parsed.entries))

    return ParseReport(articles, len(parsed.entries), failures, samples)
