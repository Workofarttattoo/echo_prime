"""Engine helpers for Postgres and DuckDB connections."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def create_postgres_engine(url: str, *, echo: bool = False) -> Engine:
    """Return a SQLAlchemy engine for Postgres."""
    return create_engine(url, echo=echo, future=True)


def create_duckdb_connection(path: Optional[Path] = None) -> duckdb.DuckDBPyConnection:
    """
    Return a DuckDB connection bound to the supplied path.

    If no path is provided, an in-memory database is used.
    """
    if path is None:
        return duckdb.connect(database=":memory:")
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(database=str(path))
