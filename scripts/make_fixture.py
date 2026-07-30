"""Write a small synthetic landing zone.

CI has no network access to the publishers — and should not have. A dbt build
still needs rows to build against, so this generates a landing zone that is
structurally identical to a real one: same schema, same partition layout, and
the same *properties* the models care about.

**Derived from the feed registry, not maintained beside it.** The first version
hard-coded its own list of feeds, and drifted the moment a source changed:
dropping `antara/olahraga` and adding Republika left the fixture emitting a
channel the taxonomy no longer knew, which failed `assert_taxonomy_coverage`.
The contract caught it, but a fixture that can disagree with the registry will
keep finding new ways to. Reading `ALL_FEEDS` makes that impossible.

The deliberate properties are layered on top, because a fixture that is merely
well-formed passes every test while proving nothing:

- **URL leakage**, reproduced per source from `Source.section_in_url`, so CNN and
  Liputan6 give their label away and ANTARA does not,
- **ANTARA's evergreen items** — old articles mixed into live news feeds,
- **a cross-source duplicate** — one wire story republished under a different
  section, which exercises the clustering and the label-disagreement flag,
- **a missing timestamp**, because feedparser cannot always read a date,
- **publisher boilerplate**, so the stripping is exercised end to end.

Usage: `python scripts/make_fixture.py [--out data/raw]`
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kanal.ingest.sources import ALL_FEEDS, SOURCE_BY_NAME, Feed

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)

TITLES = [
    "Pemerintah umumkan kebijakan baru soal subsidi energi",
    "Harga emas menguat di tengah ketidakpastian global",
    "Timnas lolos ke babak berikutnya setelah menang tipis",
    "Peneliti temukan metode baru deteksi dini penyakit",
    "Bank sentral pertahankan suku bunga acuan bulan ini",
    "Startup lokal umumkan pendanaan seri A",
    "Kompetisi liga dimulai akhir pekan ini",
    "Kualitas udara membaik setelah hujan sepekan",
]

# Boilerplate the cleaner is expected to strip, keyed by source.
BOILERPLATE = {
    "liputan6": "Liputan6.com, Jakarta - ",
    "republika": "REPUBLIKA.CO.ID,&nbsp;JAKARTA &ndash;&nbsp;",
}


def article_url(feed: Feed, article_id: int, slug: str) -> str:
    """A URL shaped like the publisher's, including whether it leaks the section.

    Derived from the registry rather than hard-coded, so a source whose leakage
    behaviour is corrected in `sources.py` is reflected here automatically.
    """
    source = SOURCE_BY_NAME[feed.source]
    host = f"{feed.source}.example.co.id"
    if source.section_in_url:
        return f"https://{host}/{feed.channel}/read/{article_id}/{slug}"
    return f"https://{host}/berita/{article_id}/{slug}"


def key_for(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def row(
    *,
    title: str,
    summary: str,
    feed: Feed,
    url: str,
    published: datetime | None,
    fetched: datetime,
) -> dict[str, object]:
    return {
        "article_key": key_for(url),
        "canonical_url": url,
        "title_fingerprint": hashlib.sha256(title.lower().encode("utf-8")).hexdigest(),
        "title": title,
        "summary": summary,
        "kanal": str(feed.kanal),
        "source": feed.source,
        "channel": feed.channel,
        "feed_id": feed.feed_id,
        "raw_link": url,
        "published_at": published.isoformat() if published else None,
        "fetched_at": fetched.isoformat(),
        "schema_version": 1,
        "extra": {"fixture": True},
    }


def build(out: Path) -> int:
    partitions: dict[str, list[dict[str, object]]] = {}
    day = NOW.date().isoformat()

    def add(r: dict[str, object]) -> None:
        partitions.setdefault(str(r["source"]), []).append(r)

    for i, feed in enumerate(ALL_FEEDS):
        for j, base_title in enumerate(TITLES):
            # Unique per *feed*, not per source. Suffixing by source alone made
            # every feed of a publisher emit identical headlines, which clustered
            # into fake duplicate groups and manufactured 102 rows of label
            # disagreement — a fixture inventing the very signal the models look
            # for is worse than no fixture, because the build goes green while
            # testing nothing.
            title = f"{base_title} ({feed.source}/{feed.channel}-{j})"
            slug = base_title.lower().replace(" ", "-")[:40]
            url = article_url(feed, 100_000 + i * 100 + j, slug)

            # ANTARA mixes evergreen explainers into its live news feeds.
            evergreen = feed.source == "antara" and j % 3 == 0
            published = NOW - timedelta(days=120 if evergreen else 0, hours=j + 1)

            prefix = BOILERPLATE.get(feed.source, "")
            add(
                row(
                    title=title,
                    summary=f"{prefix}Ringkasan untuk: {base_title}",
                    feed=feed,
                    url=url,
                    published=published,
                    fetched=NOW,
                )
            )

    # A wire story republished by another publisher under a *different* section.
    # Same normalised headline so it clusters; different label so it exercises
    # the disagreement flag the noise ceiling will be built on.
    wire = "Kesepakatan dagang baru disepakati dua negara"
    pair = _pick_disagreeing_pair()
    for feed in pair:
        add(
            row(
                title=wire,
                summary=f"{BOILERPLATE.get(feed.source, '')}Ringkasan wire.",
                feed=feed,
                url=article_url(feed, 999_000 + hash(feed.feed_id) % 900, "kesepakatan-dagang"),
                published=NOW - timedelta(hours=5),
                fetched=NOW,
            )
        )

    # A feed that emitted a date feedparser could not read.
    undated = ALL_FEEDS[0]
    add(
        row(
            title="Artikel tanpa tanggal terbit yang bisa dibaca",
            summary="Ringkasan tanpa tanggal.",
            feed=undated,
            url=article_url(undated, 999_999, "tanpa-tanggal"),
            published=None,
            fetched=NOW,
        )
    )

    total = 0
    for source, rows in partitions.items():
        directory = out / f"source={source}" / f"dt={day}"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "fixture.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(rows)
    return total


def _pick_disagreeing_pair() -> tuple[Feed, Feed]:
    """Two feeds from different publishers carrying different labels.

    Chosen from the registry so the pair stays valid when sources change, rather
    than naming feeds that may later be dropped.
    """
    for a in ALL_FEEDS:
        for b in ALL_FEEDS:
            if a.source != b.source and a.kanal != b.kanal:
                return a, b
    raise RuntimeError("registry has no two feeds with different sources and labels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    print(f"wrote {build(args.out)} fixture rows to {args.out}")


if __name__ == "__main__":
    main()
