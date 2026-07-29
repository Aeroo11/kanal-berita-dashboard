"""Load the landing zone into DuckDB.

The one writer. Everything else in the project reads.

Loading is deliberately dumb: read JSONL, anti-join on `article_key`, insert
what is new, record which files were consumed. No cleaning, no reshaping, no
judgement — that all belongs in dbt where it is version-controlled, tested, and
visible as lineage. A loader that quietly transforms is a loader whose output
nobody can explain.

Two properties it must have, both tested:

- **Idempotent.** Loading the same files twice adds nothing the second time.
- **Incremental.** A cycle only reads files it has not already consumed, so the
  load stays cheap as the landing zone grows past a hundred thousand rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from kanal.config import settings
from kanal.warehouse.duck import writer
from kanal.warehouse.schema import DDL

log = logging.getLogger(__name__)


@dataclass
class LoadReport:
    files_seen: int
    files_loaded: int
    rows_read: int
    rows_added: int

    @property
    def rows_skipped(self) -> int:
        return self.rows_read - self.rows_added

    def summary(self) -> str:
        return (
            f"loaded {self.files_loaded}/{self.files_seen} new file(s): "
            f"{self.rows_added:,} rows added, {self.rows_skipped:,} already present"
        )


def default_db_path() -> Path:
    return settings.data_dir / "kanal.duckdb"


def load(
    raw_dir: Path | None = None,
    db_path: Path | None = None,
    *,
    force: bool = False,
) -> LoadReport:
    """Bring the warehouse up to date with the landing zone.

    `force` re-reads files already in `_load_log`. The anti-join still prevents
    duplicate rows, so this is a repair tool for a corrupted log rather than a
    way to double-count.
    """
    raw = raw_dir or settings.raw_dir
    db = db_path or default_db_path()

    files = sorted(raw.rglob("*.jsonl")) if raw.exists() else []
    if not files:
        log.info("landing zone at %s is empty", raw)
        return LoadReport(0, 0, 0, 0)

    with writer(db) as conn:
        conn.execute(DDL)

        already: set[str] = set()
        if not force:
            already = {row[0] for row in conn.execute("SELECT file_path FROM _load_log").fetchall()}

        pending = [f for f in files if _relative(f, raw) not in already]
        if not pending:
            log.info("warehouse already current: %d file(s), nothing new", len(files))
            return LoadReport(len(files), 0, 0, 0)

        total_read = 0
        total_added = 0

        for path in pending:
            rel = _relative(path, raw)

            # read_json_auto handles the newline-delimited form and infers the
            # schema. `union_by_name` means a file written by a newer ingest
            # version — one that added a column — still loads, with the missing
            # columns null rather than the whole file rejected.
            #
            # RETURNING gives the inserted count directly. Bracketing the insert
            # with two COUNT(*) queries would give the same number today and the
            # wrong one the moment anything else writes concurrently.
            inserted = conn.execute(
                """
                INSERT INTO raw_articles (
                    article_key, canonical_url, title_fingerprint, title, summary,
                    kanal, source, channel, feed_id, raw_link,
                    published_at, fetched_at, schema_version, extra, _ingest_file
                )
                SELECT
                    j.article_key,
                    j.canonical_url,
                    j.title_fingerprint,
                    j.title,
                    j.summary,
                    j.kanal,
                    j.source,
                    j.channel,
                    j.feed_id,
                    j.raw_link,
                    try_cast(j.published_at AS TIMESTAMP),
                    try_cast(j.fetched_at   AS TIMESTAMP),
                    j.schema_version,
                    to_json(j.extra),
                    $rel
                FROM read_json_auto($path, format='newline_delimited',
                                    union_by_name=true) AS j
                -- Anti-join: only genuinely new articles. A feed repeating an
                -- item across cycles, or the same wire story landing in two
                -- partitions, resolves to one row.
                WHERE NOT EXISTS (
                    SELECT 1 FROM raw_articles r WHERE r.article_key = j.article_key
                )
                RETURNING article_key
                """,
                {"path": str(path), "rel": rel},
            ).fetchall()
            added = len(inserted)

            read = _scalar(
                conn.execute(
                    "SELECT count(*) FROM read_json_auto($path, format='newline_delimited')",
                    {"path": str(path)},
                ).fetchone()
            )

            conn.execute(
                """
                INSERT INTO _load_log (file_path, rows_read, rows_added)
                VALUES ($rel, $read, $added)
                ON CONFLICT (file_path) DO UPDATE
                    SET rows_read = excluded.rows_read,
                        rows_added = excluded.rows_added,
                        loaded_at = now()
                """,
                {"rel": rel, "read": read, "added": added},
            )

            total_read += read
            total_added += added

        report = LoadReport(len(files), len(pending), total_read, total_added)
        log.info("%s", report.summary())
        return report


def _scalar(row: tuple[object, ...] | None) -> int:
    """First column of a single-row result, as an int.

    `fetchone()` is typed as optional because a query *might* return no rows.
    An aggregate always returns exactly one, so a None here would mean the
    query is not what it looks like — worth failing loudly rather than
    defaulting to zero and quietly under-reporting.
    """
    if row is None:
        raise RuntimeError("aggregate query returned no rows")
    value = row[0]
    if not isinstance(value, int | float):
        raise TypeError(f"expected a numeric scalar, got {type(value).__name__}")
    return int(value)


def _relative(path: Path, root: Path) -> str:
    """Landing-zone-relative path, with forward slashes.

    Stored rather than the absolute path so the log survives the repository
    being cloned somewhere else — and so a load performed on a CI runner and one
    performed on a laptop agree about which files they have already seen.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
