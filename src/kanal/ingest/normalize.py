"""URL canonicalisation and the natural key.

Ingestion runs hourly against feeds that mostly repeat themselves — a 50-item
feed polled every hour yields the same 45 articles it did last time. Re-running
a poll, or backfilling a partition, must therefore be a no-op rather than a
duplicate. That requires a stable identity for an article, and the only
identifier every feed reliably carries is its link.

Links are not stable as published. The same article arrives as

    https://www.antaranews.com/berita/123/judul?utm_source=rss&utm_medium=feed
    https://www.antaranews.com/berita/123/judul/
    http://antaranews.com/berita/123/judul#section

so the key is a hash of the *canonicalised* URL, not the raw one. Every rule
below exists because it changes what counts as the same article; none of them
are cosmetic.
"""

from __future__ import annotations

import hashlib
import html
import re
from html.parser import HTMLParser
from io import StringIO
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Tracking parameters carry no identity — the same article is the same article
# whether it was reached from RSS, a newsletter, or a share button.
_TRACKING_PREFIXES = ("utm_", "pk_", "mtm_", "_ga")
_TRACKING_EXACT = frozenset(
    {
        "fbclid",
        "gclid",
        "igshid",
        "mc_cid",
        "mc_eid",
        "ref",
        "source",
        "medium",
        "campaign",
        "amp",
        "spec",  # Liputan6 appends ?spec=... on syndicated links
    }
)

_MULTI_SLASH = re.compile(r"/{2,}")


def _is_tracking(param: str) -> bool:
    p = param.lower()
    return p in _TRACKING_EXACT or p.startswith(_TRACKING_PREFIXES)


def canonical_url(raw: str) -> str:
    """Reduce a URL to a stable identity.

    Applied rules, in order:

    - scheme forced to https (publishers redirect http → https anyway, and the
      two forms would otherwise be different articles)
    - host lowercased, leading `www.` and any `:443` / `:80` removed
    - tracking query parameters dropped, survivors sorted so ordering cannot
      change the key
    - fragment dropped — it addresses a position within a page, not a page
    - duplicate slashes collapsed and a single trailing slash removed, but the
      path is otherwise left alone, including its case: some CMSes do serve
      case-sensitive slugs and lowercasing them would merge distinct articles

    Raises `ValueError` on input that is not a usable absolute http(s) URL,
    because silently returning a mangled string would create a key collision.
    """
    if not raw or not raw.strip():
        raise ValueError("empty URL")

    parts = urlsplit(raw.strip())
    if parts.scheme not in ("http", "https", ""):
        raise ValueError(f"unsupported scheme {parts.scheme!r} in {raw!r}")

    host = parts.hostname
    if not host:
        raise ValueError(f"URL has no host: {raw!r}")
    host = host.lower().removeprefix("www.")

    # Keep a non-default port; drop the defaults, which carry no identity.
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"

    path = _MULTI_SLASH.sub("/", parts.path) or "/"
    if len(path) > 1:
        path = path.rstrip("/")

    kept = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k)
    ]
    query = urlencode(sorted(kept))

    return urlunsplit(("https", host, path, query, ""))


def article_key(url: str) -> str:
    """Stable primary key for an article: sha256 of its canonical URL.

    Hex rather than raw bytes so it survives JSONL, Parquet, and a SQL `VARCHAR`
    without encoding decisions, and sorts sensibly in a UI.
    """
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()


# Publisher boilerplate that prefixes summaries. It identifies the publisher,
# and publishers have different section mixes — so leaving it in hands the model
# a partial label. Stripped at ingest so the raw store is already clean of it,
# and asserted again in the feature layer.
_BOILERPLATE = (
    re.compile(r"^Liputan6\.com,\s*[^-–—]{0,40}[-–—]\s*", re.IGNORECASE),
    re.compile(r"^KOMPAS\.com\s*[-–—]\s*", re.IGNORECASE),
    re.compile(r"^CNN\s*Indonesia\s*[-–—]\s*", re.IGNORECASE),
    re.compile(r"^TEMPO\.CO,\s*[^-–—]{0,40}[-–—]\s*", re.IGNORECASE),
    re.compile(r"^detik\w*\s*[-–—]\s*", re.IGNORECASE),
    re.compile(r"^ANTARA\s*[-–—]\s*", re.IGNORECASE),
    re.compile(r"^Jakarta\s*[-–—]\s*", re.IGNORECASE),
)

_TRAILING_NOISE = re.compile(
    r"\s*(Baca juga\s*:.*|Simak selengkapnya.*|\(\s*Baca\s*:.*\)|Artikel ini telah tayang.*)$",
    re.IGNORECASE | re.DOTALL,
)

_WHITESPACE = re.compile(r"\s+")


class _TextExtractor(HTMLParser):
    """Collect the text of an HTML fragment, discarding markup entirely.

    ANTARA embeds a thumbnail in every summary::

        <img align="left" border="0" src="https://cdn.antaranews.com/cache/
        800x533/2026/07/16/iran.jpg"/>Actual summary text follows…

    That markup is not content. Worse, the image path carries a date and a
    filename derived from the story, so leaving it in hands a model a feature
    that has nothing to do with the headline and everything to do with which
    CDN bucket the publisher used.

    Uses the standard library's parser rather than a regex: RSS summaries carry
    unclosed tags, stray angle brackets in quoted text, and entity references,
    and a regex gets all three wrong in different ways.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out = StringIO()

    def handle_data(self, data: str) -> None:
        self._out.write(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Block-ish tags become a space so words either side do not fuse.
        if tag in ("br", "p", "div", "li", "tr"):
            self._out.write(" ")

    @property
    def text(self) -> str:
        return self._out.getvalue()


def strip_html(value: str) -> str:
    """Return the visible text of an HTML fragment."""
    if "<" not in value and "&" not in value:
        return value  # the common case: nothing to do

    parser = _TextExtractor()
    try:
        parser.feed(value)
        parser.close()
        return parser.text
    except Exception:
        # Malformed beyond recovery. Fall back to a blunt tag strip rather
        # than dropping the summary, which would lose a real article.
        return html.unescape(re.sub(r"<[^>]*>", " ", value))


def clean_text(value: str | None) -> str:
    """Normalise feed text: strip markup, publisher boilerplate, and cross-promotion."""
    if not value:
        return ""

    text = _WHITESPACE.sub(" ", strip_html(value)).strip()
    # Some feeds stack two prefixes ("Liputan6.com, Jakarta - Jakarta - ").
    for _ in range(2):
        before = text
        for pattern in _BOILERPLATE:
            text = pattern.sub("", text, count=1).strip()
        if text == before:
            break

    text = _TRAILING_NOISE.sub("", text).strip()
    return text


_TITLE_NOISE = re.compile(r"[^\w\s]", re.UNICODE)


def title_fingerprint(title: str) -> str:
    """Aggressively normalised title, for near-duplicate detection.

    A wire story republished by three outlets keeps its headline almost intact
    but picks up punctuation and casing differences. Lowercasing and dropping
    punctuation makes those collapse. This is the *cheap* exact-match layer; the
    MinHash clustering in `kanal.data.dedup` catches the rest.
    """
    folded = _WHITESPACE.sub(" ", _TITLE_NOISE.sub(" ", title.lower())).strip()
    return hashlib.sha256(folded.encode("utf-8")).hexdigest()
