"""Prefect flows orchestrating ETL for measurement datasets."""

from __future__ import annotations

from pathlib import Path

from prefect import flow

from qulab_infinite.etl.tasks import (
    deduplicate,
    detect_outliers,
    load_raw_payload,
    parse_measurements,
    persist_measurements,
)


@flow(name="measurement-etl")
def run_measurement_etl(
    raw_path: str,
    postgres_url: str,
    *,
    allow_empty: bool = False,
) -> int:
    """
    Deterministically ingest a measurement dataset and persist it to Postgres.

    Returns:
        Number of new records inserted.
    """
    payload = load_raw_payload(Path(raw_path))
    records = parse_measurements(payload)
    unique_records = deduplicate(records)
    curated = detect_outliers(unique_records)

    if not curated and not allow_empty:
        raise ValueError("No records available after QA gating.")

    inserted = persist_measurements(curated, postgres_url)
    return inserted
