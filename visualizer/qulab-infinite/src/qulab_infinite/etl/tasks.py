"""Prefect tasks for the deterministic ETL pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Iterable, List, Sequence

import numpy as np
from prefect import task

from qulab_infinite.database.engines import create_postgres_engine
from qulab_infinite.database.repository import upsert_measurements
from qulab_infinite.database.schema import ensure_schema
from qulab_infinite.validation.measurement import MeasurementRecord


@task
def load_raw_payload(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@task
def parse_measurements(items: Sequence[dict]) -> List[MeasurementRecord]:
    return [MeasurementRecord(**item) for item in items]


@task
def deduplicate(records: Sequence[MeasurementRecord]) -> List[MeasurementRecord]:
    seen = set()
    output = []
    for record in records:
        if record.provenance_hash in seen:
            continue
        seen.add(record.provenance_hash)
        output.append(record)
    return output


@task
def detect_outliers(records: Sequence[MeasurementRecord], z_threshold: float = 3.5) -> List[MeasurementRecord]:
    if not records:
        return []

    values = np.array([record.value_si for record in records], dtype=float)
    med = median(values.tolist())
    mad = np.median(np.abs(values - med)) or 1e-12
    modified_z = 0.6745 * (values - med) / mad

    filtered: List[MeasurementRecord] = []
    for record, z_score in zip(records, modified_z, strict=False):
        if abs(z_score) <= z_threshold:
            filtered.append(record)
    return filtered


@task
def persist_measurements(records: Sequence[MeasurementRecord], postgres_url: str) -> int:
    engine = create_postgres_engine(postgres_url)
    ensure_schema(engine)
    return upsert_measurements(engine, list(records))
