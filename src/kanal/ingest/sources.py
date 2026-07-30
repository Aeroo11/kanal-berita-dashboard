"""The feed registry.

Each entry pairs a URL with the *canonical kanal* that feed represents. That
pairing is the entire labelling strategy: the label of an article is the
section its publisher filed it under. No annotation, no cost, and new labels
arrive with every poll.

Two things are deliberately explicit here rather than inferred:

1.  **The taxonomy mapping.** Publishers disagree about how to slice the news.
    ANTARA files football under both `sepakbola` and `olahraga`; Liputan6 puts
    crime under `news`; CNN has `internasional` where ANTARA has `dunia`.
    Collapsing those onto eight canonical classes is a judgement call, and the
    `notes` on each entry record the reasoning so it can be argued with.

2.  **Leakage properties.** CNN puts the section in the article URL, Liputan6
    puts it in the URL *and* an item-level `<category>`, ANTARA does neither.
    A model shown any of those scores ~100% and has learnt nothing. The
    feature layer must never see them — see `kanal.features.text` and
    `tests/test_leakage.py`. The contrast between these three sources is what
    makes the leakage demonstration possible, so it is a property worth
    preserving rather than an inconvenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Kanal(StrEnum):
    """The canonical taxonomy. Eight classes, fixed."""

    POLITIK = "politik"
    EKONOMI = "ekonomi"
    OLAHRAGA = "olahraga"
    TEKNOLOGI = "teknologi"
    HIBURAN = "hiburan"
    INTERNASIONAL = "internasional"
    HUKUM_KRIMINAL = "hukum-kriminal"
    GAYA_HIDUP_KESEHATAN = "gaya-hidup-kesehatan"


@dataclass(frozen=True, slots=True)
class Feed:
    """One pollable feed, and what it means."""

    source: str
    """Publisher key. Also the landing-zone partition key."""

    channel: str
    """The publisher's own name for this section, kept verbatim for provenance."""

    url: str

    kanal: Kanal
    """Canonical label for articles arriving on this feed."""

    notes: str = ""
    """Why this channel maps to this kanal. Judgement calls get recorded."""

    @property
    def feed_id(self) -> str:
        return f"{self.source}:{self.channel}"


@dataclass(frozen=True, slots=True)
class Source:
    """A publisher, and the quirks that come with it."""

    name: str
    homepage: str
    section_in_url: bool
    """True if the article URL leaks the section. Documented, never used as a feature."""

    item_level_category: bool
    """True if items carry their own <category> element."""

    notes: str = ""
    feeds: tuple[Feed, ...] = field(default_factory=tuple)


# ── ANTARA ───────────────────────────────────────────────────────────────
# The state news agency. Leaks nothing: article URLs are /berita/{id}/{slug}.
# That makes it the clean control group for the leakage experiment, and the
# only source we could train on in isolation without contamination.
_ANTARA = "antara"
ANTARA = Source(
    name=_ANTARA,
    homepage="https://www.antaranews.com",
    section_in_url=False,
    item_level_category=True,  # channel-level only, not per item
    notes=(
        "Wire agency: its stories are syndicated near-verbatim by other outlets "
        "hours later, which is the origin of the cross-source near-duplicate "
        "problem the dedup clustering exists to solve."
    ),
    feeds=(
        Feed(_ANTARA, "politik", "https://www.antaranews.com/rss/politik.xml", Kanal.POLITIK),
        Feed(_ANTARA, "hukum", "https://www.antaranews.com/rss/hukum.xml", Kanal.HUKUM_KRIMINAL),
        Feed(_ANTARA, "ekonomi", "https://www.antaranews.com/rss/ekonomi.xml", Kanal.EKONOMI),
        Feed(
            _ANTARA,
            "metro",
            "https://www.antaranews.com/rss/metro.xml",
            Kanal.HUKUM_KRIMINAL,
            notes=(
                "Judgement call: ANTARA's 'metro' is Jakarta city news, which is "
                "dominated by crime and civic incidents. Mapped to hukum-kriminal "
                "rather than inventing a 'daerah' class. Revisit if the confusion "
                "matrix shows it bleeding into politik."
            ),
        ),
        Feed(
            _ANTARA,
            "sepakbola",
            "https://www.antaranews.com/rss/sepakbola.xml",
            Kanal.OLAHRAGA,
            notes=(
                "ANTARA's only live sport feed. Its /rss/olahraga.xml was in this "
                "registry until the feed-health contract measured it: 100% "
                "evergreen with a freshest item 198 days old, so abandoned "
                "rather than quiet — and redundant, since this feed covers the "
                "same class and is current. Removed on that evidence."
            ),
        ),
        Feed(_ANTARA, "tekno", "https://www.antaranews.com/rss/tekno.xml", Kanal.TEKNOLOGI),
        Feed(_ANTARA, "hiburan", "https://www.antaranews.com/rss/hiburan.xml", Kanal.HIBURAN),
        Feed(
            _ANTARA,
            "dunia",
            "https://www.antaranews.com/rss/dunia.xml",
            Kanal.INTERNASIONAL,
            notes="ANTARA says 'dunia' where CNN says 'internasional'. Same class.",
        ),
        Feed(
            _ANTARA,
            "humaniora",
            "https://www.antaranews.com/rss/humaniora.xml",
            Kanal.GAYA_HIDUP_KESEHATAN,
            notes=(
                "Weakest mapping in the registry. 'humaniora' spans health, "
                "education and social affairs; it is the class most likely to be "
                "responsible for label noise. Flagged for the noise-ceiling study."
            ),
        ),
        Feed(
            _ANTARA,
            "lifestyle",
            "https://www.antaranews.com/rss/lifestyle.xml",
            Kanal.GAYA_HIDUP_KESEHATAN,
        ),
    ),
)

# ── CNN Indonesia ────────────────────────────────────────────────────────
# Section appears in the article URL (/nasional/2026...). Strong leak.
_CNN = "cnn"
CNN = Source(
    name=_CNN,
    homepage="https://www.cnnindonesia.com",
    section_in_url=True,
    item_level_category=False,
    notes="Section is the first path segment of every article URL — a total giveaway.",
    feeds=(
        Feed(
            _CNN,
            "nasional",
            "https://www.cnnindonesia.com/nasional/rss",
            Kanal.POLITIK,
            notes=(
                "Judgement call: CNN's 'nasional' is predominantly domestic "
                "politics and government. Mapped to politik, accepting that some "
                "crime stories land here — a known source of label noise."
            ),
        ),
        Feed(_CNN, "ekonomi", "https://www.cnnindonesia.com/ekonomi/rss", Kanal.EKONOMI),
        Feed(_CNN, "olahraga", "https://www.cnnindonesia.com/olahraga/rss", Kanal.OLAHRAGA),
        Feed(_CNN, "teknologi", "https://www.cnnindonesia.com/teknologi/rss", Kanal.TEKNOLOGI),
        Feed(_CNN, "hiburan", "https://www.cnnindonesia.com/hiburan/rss", Kanal.HIBURAN),
        Feed(
            _CNN,
            "internasional",
            "https://www.cnnindonesia.com/internasional/rss",
            Kanal.INTERNASIONAL,
        ),
        Feed(
            _CNN,
            "gaya-hidup",
            "https://www.cnnindonesia.com/gaya-hidup/rss",
            Kanal.GAYA_HIDUP_KESEHATAN,
        ),
    ),
)

# ── Liputan6 ─────────────────────────────────────────────────────────────
# Leaks twice: section in the URL *and* an item-level <category>.
_LIPUTAN6 = "liputan6"
LIPUTAN6 = Source(
    name=_LIPUTAN6,
    homepage="https://www.liputan6.com",
    section_in_url=True,
    item_level_category=True,
    notes=(
        "Leaks the label two ways. Also prefixes summaries with boilerplate "
        "('Liputan6.com, Jakarta - ') that must be stripped, since the prefix "
        "itself identifies the publisher and correlates with its section mix."
    ),
    feeds=(
        Feed(_LIPUTAN6, "bisnis", "https://feed.liputan6.com/rss/bisnis", Kanal.EKONOMI),
        Feed(_LIPUTAN6, "bola", "https://feed.liputan6.com/rss/bola", Kanal.OLAHRAGA),
        Feed(_LIPUTAN6, "tekno", "https://feed.liputan6.com/rss/tekno", Kanal.TEKNOLOGI),
        Feed(
            _LIPUTAN6,
            "showbiz",
            "https://feed.liputan6.com/rss/showbiz",
            Kanal.HIBURAN,
        ),
        Feed(
            _LIPUTAN6,
            "global",
            "https://feed.liputan6.com/rss/global",
            Kanal.INTERNASIONAL,
        ),
        Feed(
            _LIPUTAN6,
            "health",
            "https://feed.liputan6.com/rss/health",
            Kanal.GAYA_HIDUP_KESEHATAN,
        ),
        Feed(
            _LIPUTAN6,
            "lifestyle",
            "https://feed.liputan6.com/rss/lifestyle",
            Kanal.GAYA_HIDUP_KESEHATAN,
        ),
    ),
)

# ── Republika ────────────────────────────────────────────────────────────
# Added to recover a third editorial perspective after CNN turned out to be
# unreachable from CI: CNN and Tempo sit behind Cloudflare and answer 403 to
# datacentre IPs, while ANTARA, Liputan6 and Republika do not.
#
# Republika's RSS estate is *mostly abandoned*, and surveying it before adding
# anything was the difference between a third source and a poisoned dataset.
# Of 21 section feeds probed:
#
#   live (newest item < 48h)  nasional, ekonomi, internasional, khazanah,
#                             otomotif, pendidikan
#   stale by years            kesehatan (15.4y), leisure (8.5y), olahraga (3.9y),
#                             teknologi (3.9y), sepakbola (3.5y), trendtek (3.5y),
#                             dunia-islam (3.5y), jurnalisme-warga (4.2y),
#                             hiburan (1.2y)
#   empty                     bola, islam, gayahidup, sepak-bola, amp
#
# `/rss/kesehatan` is the trap: it maps *cleanly* onto gaya-hidup-kesehatan and
# would have flooded the store with articles from 2011. This is a different risk
# class from ANTARA's evergreen mixing — a wholly abandoned feed, not a live one
# carrying old items — and it is why `mart_feed_health` and the staleness
# contract now exist.
_REPUBLIKA = "republika"
REPUBLIKA = Source(
    name=_REPUBLIKA,
    homepage="https://republika.co.id",
    # Sections live on subdomains rather than path segments, so the leak is
    # partial: ekonomi.republika.co.id gives its section away, while both
    # nasional and internasional are served from news.republika.co.id and do not.
    section_in_url=True,
    item_level_category=True,
    notes=(
        "Islamic-leaning national daily. Prefixes summaries with "
        "'REPUBLIKA.CO.ID, <CITY> -- ' using HTML entities (&nbsp;, &ndash;). "
        "Only the three feeds below are both actively maintained and cleanly "
        "mappable onto the canonical taxonomy — khazanah (Islamic affairs), "
        "pendidikan and otomotif are live but have no honest home among eight "
        "classes, and forcing them in would be label noise by construction."
    ),
    feeds=(
        Feed(
            _REPUBLIKA,
            "nasional",
            "https://republika.co.id/rss/nasional",
            Kanal.POLITIK,
            notes=(
                "Same judgement call as CNN's 'nasional': predominantly domestic "
                "politics and government, with some crime bleeding in. Served "
                "from news.republika.co.id, so the URL does not leak the section."
            ),
        ),
        Feed(
            _REPUBLIKA,
            "ekonomi",
            "https://republika.co.id/rss/ekonomi",
            Kanal.EKONOMI,
            notes="Served from ekonomi.republika.co.id — the subdomain leaks the section.",
        ),
        Feed(
            _REPUBLIKA,
            "internasional",
            "https://republika.co.id/rss/internasional",
            Kanal.INTERNASIONAL,
            notes="Served from news.republika.co.id, so the URL does not leak the section.",
        ),
    ),
)

SOURCES: tuple[Source, ...] = (ANTARA, CNN, LIPUTAN6, REPUBLIKA)

ALL_FEEDS: tuple[Feed, ...] = tuple(f for s in SOURCES for f in s.feeds)

SOURCE_BY_NAME: dict[str, Source] = {s.name: s for s in SOURCES}


def feeds_for(source: str | None = None) -> tuple[Feed, ...]:
    """All feeds, or just one source's. Used by the CLI to poll selectively."""
    if source is None:
        return ALL_FEEDS
    if source not in SOURCE_BY_NAME:
        known = ", ".join(sorted(SOURCE_BY_NAME))
        raise KeyError(f"unknown source {source!r}; known sources: {known}")
    return SOURCE_BY_NAME[source].feeds
