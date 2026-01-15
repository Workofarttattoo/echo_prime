"""Pydantic models for validating raw measurement payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from qulab_infinite.units.registry import canonical_unit, get_unit_registry
from qulab_infinite.utils.provenance import deterministic_hash


class MeasurementRecord(BaseModel):
    quantity: str
    dataset_id: Optional[str] = Field(default=None, description="Source dataset identifier.")
    datum_index: Optional[int] = Field(default=None, description="Row index within the source dataset.")
    value: float = Field(..., description="Raw measurement value.")
    unit: str = Field(..., description="Unit string accompanying the raw value.")
    u_minus: Optional[float] = Field(default=None, ge=0)
    u_plus: Optional[float] = Field(default=None, ge=0)
    sigma: Optional[float] = Field(default=None, ge=0)
    temperature: Optional[float] = None
    temperature_unit: Optional[str] = "kelvin"
    pressure: Optional[float] = None
    pressure_unit: Optional[str] = "pascal"
    strain_rate: Optional[float] = None
    strain_rate_unit: Optional[str] = "1/second"
    purity: Optional[str] = None
    sample_prep: Optional[str] = None
    method: Optional[str] = None
    n: Optional[int] = Field(default=None, ge=1)
    citation: Optional[str] = None
    doi: Optional[str] = None
    license: Optional[str] = None
    context_tags: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: Optional[str] = None
    date_ingested: datetime = Field(default_factory=datetime.utcnow)

    value_si: float = Field(default=0.0)
    unit_si: str = Field(default="")
    temperature_k: Optional[float] = Field(default=None)
    pressure_pa: Optional[float] = Field(default=None)
    strain_rate_si: Optional[float] = Field(default=None)

    @field_validator("unit", "temperature_unit", "pressure_unit", "strain_rate_unit")
    @classmethod
    def ensure_unit_is_known(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        canonical_unit(v)
        return v

    @model_validator(mode="after")
    def compute_si_fields(self) -> "MeasurementRecord":
        registry = get_unit_registry()

        # Value conversion
        quantity = registry.Quantity(self.value, self.unit)
        base = quantity.to_base_units()
        self.value_si = base.magnitude
        self.unit_si = f"{base.units:~P}"

        # Aux state conversions
        if self.temperature is not None:
            temp_quantity = registry.Quantity(self.temperature, self.temperature_unit or "kelvin")
            self.temperature_k = temp_quantity.to("kelvin").magnitude
        if self.pressure is not None:
            pressure_quantity = registry.Quantity(self.pressure, self.pressure_unit or "pascal")
            self.pressure_pa = pressure_quantity.to("pascal").magnitude
        if self.strain_rate is not None:
            rate = registry.Quantity(self.strain_rate, self.strain_rate_unit or "1/second")
            self.strain_rate_si = rate.to("1/second").magnitude

        # Provenance hashing
        if not self.provenance_hash:
            payload = {
                "dataset_id": self.dataset_id,
                "datum_index": self.datum_index,
                "quantity": self.quantity,
                "value_si": self.value_si,
                "unit_si": self.unit_si,
                "temperature_k": self.temperature_k,
                "pressure_pa": self.pressure_pa,
                "strain_rate_si": self.strain_rate_si,
                "context_tags": self.context_tags,
            }
            self.provenance_hash = deterministic_hash(payload)

        return self

    def as_orm_dict(self) -> Dict[str, Any]:
        """Return a dict compatible with the SQLAlchemy Measurement model."""
        return {
            "quantity": self.quantity,
            "dataset_id": self.dataset_id,
            "datum_index": self.datum_index,
            "value_raw": self.value,
            "value_si": self.value_si,
            "u_minus": self.u_minus,
            "u_plus": self.u_plus,
            "sigma": self.sigma,
            "units_input": canonical_unit(self.unit),
            "units_si": self.unit_si,
            "temperature_k": self.temperature_k,
            "pressure_pa": self.pressure_pa,
            "strain_rate": self.strain_rate_si,
            "purity": self.purity,
            "sample_prep": self.sample_prep,
            "method": self.method,
            "n": self.n,
            "context_tags": self.context_tags,
            "citation": self.citation,
            "doi": self.doi,
            "license": self.license,
            "provenance_hash": self.provenance_hash,
            "date_ingested": self.date_ingested,
        }

    def raw_payload(self) -> Dict[str, Any]:
        """Return the raw payload suitable for JSON storage."""
        payload = {
            "dataset_id": self.dataset_id,
            "datum_index": self.datum_index,
            "quantity": self.quantity,
            "value": self.value,
            "unit": self.unit,
            "u_minus": self.u_minus,
            "u_plus": self.u_plus,
            "sigma": self.sigma,
            "temperature": self.temperature,
            "temperature_unit": self.temperature_unit,
            "pressure": self.pressure,
            "pressure_unit": self.pressure_unit,
            "strain_rate": self.strain_rate,
            "strain_rate_unit": self.strain_rate_unit,
            "purity": self.purity,
            "sample_prep": self.sample_prep,
            "method": self.method,
            "n": self.n,
            "citation": self.citation,
            "doi": self.doi,
            "license": self.license,
            "context_tags": self.context_tags,
            "provenance_hash": self.provenance_hash,
            "date_ingested": self.date_ingested.isoformat(),
        }
        return {k: v for k, v in payload.items() if v is not None}
