"""Export the modelled dataset to Parquet.

What gets published is `fct_articles`, not the landing zone. The raw JSONL is
faithful to what each poll returned — including the same article landed in two
day partitions — while the fact table is the deduplicated, typed, contract-checked
view. Publishing the raw layer would hand people the redundancy and none of the
guarantees.

Provenance columns (`canonical_url`, `source`, `channel`, `feed_id`) are included
deliberately, even though a model must never see them. Anyone using this dataset
needs to be able to reproduce the leakage measurement themselves rather than take
it on trust — and the columns are what make the ANTARA-only versus all-sources
experiment possible for someone who did not build the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kanal.warehouse.duck import reader
from kanal.warehouse.loader import default_db_path


@dataclass
class ExportReport:
    parquet_path: Path
    stats_path: Path
    articles: int
    sources: int
    kanal_classes: int
    oldest: str | None
    newest: str | None

    def summary(self) -> str:
        return (
            f"exported {self.articles:,} articles "
            f"({self.sources} sources, {self.kanal_classes} classes) "
            f"→ {self.parquet_path.name}"
        )


# The published column set. Explicit rather than `SELECT *`, so adding an
# internal column to the fact table cannot silently start publishing it.
PUBLISHED_COLUMNS = """
    article_key,
    title,
    summary,
    kanal,

    -- Provenance: never features, but necessary to reproduce the leakage work.
    source,
    channel,
    canonical_url,

    published_at,
    fetched_at,

    -- Properties a user needs in order to split the data honestly.
    is_evergreen,
    url_leaks_label,
    cluster_id,
    cluster_size,
    is_cross_source_duplicate,
    has_label_disagreement,
    label_is_judgement_call,

    title_words
"""


def export(out_dir: Path | None = None, db_path: Path | None = None) -> ExportReport:
    """Write `fct_articles` to Parquet, plus a stats sidecar."""
    out = out_dir or (Path("data") / "export")
    out.mkdir(parents=True, exist_ok=True)

    parquet_path = out / "articles.parquet"
    stats_path = out / "stats.json"

    with reader(db_path or default_db_path()) as conn:
        # ZSTD over the default snappy: the corpus is short Indonesian text with
        # heavy vocabulary overlap, so it compresses well and the dataset is
        # something people download.
        conn.execute(
            f"""
            COPY (SELECT {PUBLISHED_COLUMNS} FROM fct_articles ORDER BY published_at)
            TO '{parquet_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        row = conn.execute(
            """
            SELECT count(*), count(DISTINCT source), count(DISTINCT kanal),
                   min(published_at)::VARCHAR, max(published_at)::VARCHAR,
                   count(*) FILTER (WHERE NOT url_leaks_label),
                   count(DISTINCT cluster_id)
            FROM fct_articles
            """
        ).fetchone()
        if row is None:
            raise RuntimeError("fct_articles is empty — run `kanal load` and `dbt build` first")

        by_kanal = dict(
            conn.execute(
                "SELECT kanal, count(*) FROM fct_articles GROUP BY 1 ORDER BY 2 DESC"
            ).fetchall()
        )
        by_source = dict(
            conn.execute(
                "SELECT source, count(*) FROM fct_articles GROUP BY 1 ORDER BY 1"
            ).fetchall()
        )
        leakage = {
            str(r[0]): round(float(r[1]), 4)
            for r in conn.execute(
                """
                SELECT source, avg(CASE WHEN url_leaks_label THEN 1.0 ELSE 0.0 END)
                FROM fct_articles GROUP BY 1 ORDER BY 1
                """
            ).fetchall()
        }
        evergreen = {
            str(r[0]): round(float(r[1]), 4)
            for r in conn.execute(
                """
                SELECT source, avg(CASE WHEN is_evergreen THEN 1.0 ELSE 0.0 END)
                FROM fct_articles GROUP BY 1 ORDER BY 1
                """
            ).fetchall()
        }

    stats = {
        "generated_at": datetime.now(UTC).isoformat(),
        "articles": int(row[0]),
        "sources": int(row[1]),
        "kanal_classes": int(row[2]),
        "published_range": {"oldest": row[3], "newest": row[4]},
        "articles_by_kanal": {str(k): int(v) for k, v in by_kanal.items()},
        "articles_by_source": {str(k): int(v) for k, v in by_source.items()},
        # The clean subset is what a user actually trains the control on, and it
        # is a per-row property — the per-source averages below hide the fact
        # that Republika has both a leaking feed and two clean ones.
        "rows_without_url_leak": int(row[5]),
        "distinct_stories": int(row[6]),
        # Published as numbers rather than described in prose, so a user can
        # check the claim instead of believing it.
        "url_leak_rate_by_source": leakage,
        "evergreen_rate_by_source": evergreen,
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    return ExportReport(
        parquet_path=parquet_path,
        stats_path=stats_path,
        articles=int(row[0]),
        sources=int(row[1]),
        kanal_classes=int(row[2]),
        oldest=row[3],
        newest=row[4],
    )
