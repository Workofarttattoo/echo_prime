"""Functions to initialise and mirror the measurement schema."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from qulab_infinite.database.base import Base


def ensure_schema(engine: Engine) -> None:
    """Create the SQL schema in the connected Postgres database if needed."""
    Base.metadata.create_all(engine, checkfirst=True)


def sync_duckdb_mirror(
    postgres_engine: Engine,
    duckdb_path: Path,
    *,
    table_names: Iterable[str] = ("measurements",),
) -> None:
    """
    Mirror requested tables from Postgres into a DuckDB file for fast analytics.
    """
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    duck_conn = duckdb.connect(str(duckdb_path))

    with postgres_engine.connect() as connection:
        inspector = inspect(connection)
        available = set(inspector.get_table_names())

        for name in table_names:
            if name not in available:
                continue

            frame = pd.read_sql_table(name, connection)
            duck_conn.register("mirror_source_df", frame)
            duck_conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM mirror_source_df")
            duck_conn.unregister("mirror_source_df")

    duck_conn.close()
