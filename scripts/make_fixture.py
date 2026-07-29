"""Write a small synthetic landing zone.

CI has no network access to the publishers — and should not have. A dbt build
still needs rows to build against, so this generates a landing zone that is
structurally identical to a real one: same schema, same partition layout, and
the same *properties* the models care about.

Those properties are the point. The fixture deliberately reproduces:

- **CNN's URL leakage** — its article paths contain the section, so a build that
  broke `url_leaks_label` would show it,
- **ANTARA's evergreen items** — old articles mixed into the news feeds,
- **a cross-source duplicate** — one wire story republished under two different
  sections, which is what exercises the clustering and the label-disagreement
  flag,
- **a missing timestamp**, because feedparser cannot always read a date.

A fixture that is merely well-formed would pass every test while proving
nothing. Usage: `python scripts/make_fixture.py [--out data/raw]`
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)

# (source, channel, kanal, url_template)
FEEDS = [
    ("antara", "politik", "politik", "https://antaranews.com/berita/{id}/{slug}"),
    ("antara", "ekonomi", "ekonomi", "https://antaranews.com/berita/{id}/{slug}"),
    ("antara", "olahraga", "olahraga", "https://antaranews.com/berita/{id}/{slug}"),
    ("antara", "humaniora", "gaya-hidup-kesehatan", "https://antaranews.com/berita/{id}/{slug}"),
    # CNN puts the section in the path — the leak, reproduced on purpose.
    ("cnn", "nasional", "politik", "https://cnnindonesia.com/nasional/{id}/{slug}"),
    ("cnn", "ekonomi", "ekonomi", "https://cnnindonesia.com/ekonomi/{id}/{slug}"),
    ("cnn", "teknologi", "teknologi", "https://cnnindonesia.com/teknologi/{id}/{slug}"),
    ("liputan6", "bisnis", "ekonomi", "https://liputan6.com/bisnis/read/{id}/{slug}"),
    ("liputan6", "bola", "olahraga", "https://liputan6.com/bola/read/{id}/{slug}"),
    ("liputan6", "health", "gaya-hidup-kesehatan", "https://liputan6.com/health/read/{id}/{slug}"),
]

TITLES = [
    "Pemerintah umumkan kebijakan baru soal subsidi energi",
    "Harga emas menguat di tengah ketidakpastian global",
    "Timnas lolos ke babak berikutnya setelah menang tipis",
    "Peneliti temukan metode baru deteksi dini penyakit",
    "Bank sentral pertahankan suku bunga acuan bulan ini",
    "Startup lokal umumkan pendanaan seri A",
    "Kompetisi liga dimulai akhir pekan ini",
    "Kualitas udara membaik setelah hujan sepekan",
    "Ekspor komoditas naik dibanding kuartal sebelumnya",
    "Aplikasi layanan publik hadirkan fitur baru",
]


def key_for(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def row(
    *,
    title: str,
    source: str,
    channel: str,
    kanal: str,
    url: str,
    published: datetime | None,
    fetched: datetime,
) -> dict[str, object]:
    return {
        "article_key": key_for(url),
        "canonical_url": url,
        "title_fingerprint": hashlib.sha256(title.lower().encode("utf-8")).hexdigest(),
        "title": title,
        "summary": f"Ringkasan untuk: {title}",
        "kanal": kanal,
        "source": source,
        "channel": channel,
        "feed_id": f"{source}:{channel}",
        "raw_link": url,
        "published_at": published.isoformat() if published else None,
        "fetched_at": fetched.isoformat(),
        "schema_version": 1,
        "extra": {"fixture": True},
    }


def build(out: Path) -> int:
    partitions: dict[tuple[str, str], list[dict[str, object]]] = {}
    day = NOW.date().isoformat()

    def add(r: dict[str, object]) -> None:
        partitions.setdefault((str(r["source"]), day), []).append(r)

    for i, (source, channel, kanal, template) in enumerate(FEEDS):
        for j, title in enumerate(TITLES):
            # The suffix must be unique per *feed*, not per source. Using the
            # source alone made all four ANTARA feeds emit identical headlines,
            # which clustered into fake duplicate groups of four and produced
            # 102 rows of "label disagreement" that were pure artefact. A
            # fixture that manufactures the very signal the models look for is
            # worse than no fixture: the build goes green while testing nothing.
            unique_title = f"{title} ({source}/{channel}-{j})"
            slug = title.lower().replace(" ", "-")[:40]
            url = template.format(id=100000 + i * 100 + j, slug=slug)

            # ANTARA's evergreen problem: a third of its items are months old.
            evergreen = source == "antara" and j % 3 == 0
            published = NOW - timedelta(days=120 if evergreen else 0, hours=j + 1)

            add(
                row(
                    title=unique_title,
                    source=source,
                    channel=channel,
                    kanal=kanal,
                    url=url,
                    published=published,
                    fetched=NOW,
                )
            )

    # A wire story republished by another outlet under a *different* section.
    # Same normalised headline, so it clusters; different label, so it exercises
    # the label-disagreement flag that the noise ceiling will be built on.
    wire = "Kesepakatan dagang baru disepakati dua negara"
    add(
        row(
            title=wire,
            source="antara",
            channel="ekonomi",
            kanal="ekonomi",
            url="https://antaranews.com/berita/999001/kesepakatan-dagang",
            published=NOW - timedelta(hours=6),
            fetched=NOW,
        )
    )
    add(
        row(
            title=wire,
            source="cnn",
            channel="nasional",
            kanal="politik",
            url="https://cnnindonesia.com/nasional/999002/kesepakatan-dagang",
            published=NOW - timedelta(hours=4),
            fetched=NOW,
        )
    )

    # A feed that emitted an unreadable date.
    add(
        row(
            title="Artikel tanpa tanggal terbit yang bisa dibaca",
            source="liputan6",
            channel="bisnis",
            kanal="ekonomi",
            url="https://liputan6.com/bisnis/read/999003/tanpa-tanggal",
            published=None,
            fetched=NOW,
        )
    )

    total = 0
    for (source, dt), rows in partitions.items():
        directory = out / f"source={source}" / f"dt={dt}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "fixture.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(rows)

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    written = build(args.out)
    print(f"wrote {written} fixture rows to {args.out}")


if __name__ == "__main__":
    main()
