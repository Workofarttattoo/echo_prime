"""ORM models for persistent measurements."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.sqltypes import Float, Integer, String, Text

from qulab_infinite.database.base import Base, TimestampMixin


class Measurement(Base, TimestampMixin):
    """Captured measurement datum with canonical SI values."""

    __tablename__ = "measurements"
    __table_args__ = (
        UniqueConstraint("provenance_hash", name="uq_measurements_provenance_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    quantity: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    datum_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    value_raw: Mapped[float] = mapped_column(Float, nullable=False)
    value_si: Mapped[float] = mapped_column(Float, nullable=False)
    u_minus: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    u_plus: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sigma: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    units_input: Mapped[str] = mapped_column(String(64), nullable=False)
    units_si: Mapped[str] = mapped_column(String(64), nullable=False)
    temperature_k: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pressure_pa: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    strain_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    purity: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sample_prep: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    method: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    n: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    context_tags: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    citation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    doi: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    license: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    provenance_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    date_ingested: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the measurement."""
        return {
            "id": self.id,
            "quantity": self.quantity,
            "dataset_id": self.dataset_id,
            "datum_index": self.datum_index,
            "value_raw": self.value_raw,
            "value_si": self.value_si,
            "u_minus": self.u_minus,
            "u_plus": self.u_plus,
            "sigma": self.sigma,
            "units_input": self.units_input,
            "units_si": self.units_si,
            "temperature_k": self.temperature_k,
            "pressure_pa": self.pressure_pa,
            "strain_rate": self.strain_rate,
            "purity": self.purity,
            "sample_prep": self.sample_prep,
            "method": self.method,
            "n": self.n,
            "context_tags": self.context_tags,
            "citation": self.citation,
            "doi": self.doi,
            "license": self.license,
            "provenance_hash": self.provenance_hash,
            "date_ingested": self.date_ingested.isoformat(),
        }
