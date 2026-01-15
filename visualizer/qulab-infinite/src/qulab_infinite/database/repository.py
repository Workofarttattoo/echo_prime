"""Persistence helpers for measurements."""

from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from qulab_infinite.database.models import Measurement
from qulab_infinite.validation.measurement import MeasurementRecord


def upsert_measurements(engine: Engine, records: Sequence[MeasurementRecord]) -> int:
    """
    Insert or update measurements based on provenance hash.

    Returns the number of rows inserted.
    """
    inserted = 0
    if not records:
        return 0

    with Session(engine) as session:
        hashes = [r.provenance_hash for r in records]
        existing_hashes = {
            row[0]
            for row in session.execute(
                select(Measurement.provenance_hash).where(Measurement.provenance_hash.in_(hashes))
            )
        }

        for record in records:
            if record.provenance_hash in existing_hashes:
                continue
            session.add(Measurement(**record.as_orm_dict()))
            inserted += 1

        session.commit()
    return inserted
