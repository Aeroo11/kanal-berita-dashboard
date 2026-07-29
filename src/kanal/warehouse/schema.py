"""Warehouse DDL.

`raw_articles` is a landing table, not a model: it mirrors what ingestion wrote,
including the `extra` blob and `schema_version`, and applies no interpretation.
Everything downstream is a dbt model built from it.

Keeping the raw table faithful matters because the interesting failures are
interpretive. When a feed changes shape, the useful question is "what did the
publisher actually send?", and that is only answerable if the answer was stored
before anyone tried to make sense of it.
"""

from __future__ import annotations

DDL = """
CREATE TABLE IF NOT EXISTS raw_articles (
    article_key        VARCHAR NOT NULL,
    canonical_url      VARCHAR NOT NULL,
    title_fingerprint  VARCHAR NOT NULL,
    title              VARCHAR NOT NULL,
    summary            VARCHAR,
    kanal              VARCHAR NOT NULL,
    source             VARCHAR NOT NULL,
    channel            VARCHAR NOT NULL,
    feed_id            VARCHAR NOT NULL,
    raw_link           VARCHAR NOT NULL,
    published_at       TIMESTAMP,
    fetched_at         TIMESTAMP NOT NULL,
    schema_version     INTEGER  NOT NULL,
    extra              JSON,
    -- Which landing-zone file this row came from. Makes any row traceable back
    -- to the exact poll that produced it, which is what turns "the numbers look
    -- wrong" into a question with an answer.
    _ingest_file       VARCHAR NOT NULL,
    _loaded_at         TIMESTAMP NOT NULL DEFAULT now()
);

-- The natural key. Loading is an anti-join against this, so re-running a load
-- over the same files is a no-op rather than a duplication.
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_articles_key
    ON raw_articles (article_key);

-- Every query is time-bounded or ordered by publication.
CREATE INDEX IF NOT EXISTS idx_raw_articles_published
    ON raw_articles (published_at);

-- Records which files have been consumed, so a load can skip them without
-- reading and parsing their contents first.
CREATE TABLE IF NOT EXISTS _load_log (
    file_path    VARCHAR PRIMARY KEY,
    rows_read    INTEGER NOT NULL,
    rows_added   INTEGER NOT NULL,
    loaded_at    TIMESTAMP NOT NULL DEFAULT now()
);
"""
