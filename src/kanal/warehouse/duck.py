"""DuckDB connection management.

Two lessons are carried over verbatim from an earlier project of mine
(TokenWatch), both learned the expensive way:

**Every DuckDB default is a *host* default.** `temp_directory` is the relative
path `.tmp`, resolved against the working directory; `memory_limit` is ~80% of
whatever RAM the machine reports; `threads` is the host core count. On a laptop
that reads as 25 GiB and 20 threads. Inside a CI runner or a function container
the same defaults mean "spill into a read-only directory, over-commit memory you
do not have, and spawn ten times more threads than you have cores". Pinning them
is not tuning — it is the difference between running and not.

**One connection is not a connection pool.** A DuckDB connection executes one
statement at a time. Sharing a single connection across concurrent readers
serialises everything behind the slowest query, and under an orchestrator it
deadlocks outright. So: exactly one writer, and readers get their own
short-lived connections.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import duckdb

log = logging.getLogger(__name__)


def duckdb_settings() -> dict[str, str]:
    """Settings pinned for portability. Every one is overridable by environment."""
    temp_dir = os.environ.get("KANAL_DUCKDB_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "kanal-duckdb"
    )
    return {
        # tempfile.gettempdir() is /tmp on Linux runners — the one reliably
        # writable path — and the platform temp directory everywhere else.
        "temp_directory": temp_dir,
        "memory_limit": os.environ.get("KANAL_DUCKDB_MEMORY_LIMIT", "1GB"),
        "threads": os.environ.get("KANAL_DUCKDB_THREADS", "2"),
        # Nothing here uses an extension. Left on, DuckDB would try to download
        # and write them under $HOME at query time — a network dependency and a
        # write to a directory that may be read-only.
        "autoinstall_known_extensions": "false",
        "autoload_known_extensions": "false",
    }


def connect(
    database: str | Path = ":memory:", *, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """Open a connection with the pinned settings applied."""
    import duckdb  # imported lazily: a ~50 MB native dependency

    config: dict[str, Any] = dict(duckdb_settings())
    Path(config["temp_directory"]).mkdir(parents=True, exist_ok=True)

    if database != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)

    return duckdb.connect(str(database), read_only=read_only, config=config)


@contextmanager
def reader(database: str | Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """A short-lived read-only connection.

    Read-only is not a formality: it lets several readers share the database
    file while the writer is idle, and it makes an accidental write from a
    query path fail loudly instead of silently mutating the warehouse.
    """
    conn = connect(database, read_only=True)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def writer(database: str | Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """The single writer. Nothing else may hold a writable connection.

    DuckDB enforces one writer per database file, so a second caller does not
    corrupt anything — it fails to open. This context manager exists to make
    that constraint visible in the code rather than discovered at runtime.
    """
    conn = connect(database, read_only=False)
    try:
        yield conn
    finally:
        conn.close()
