"""Utilities for loading and configuring the shared Pint unit registry."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

from pint import UnitRegistry


def _config_path() -> Path:
    """Return the path to the shared units configuration."""
    package_root = Path(__file__).resolve().parents[4]
    config_path = package_root / "config" / "units.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing units configuration at {config_path}")
    return config_path


def _load_unit_aliases() -> Dict[str, Dict[str, Iterable[str]]]:
    with _config_path().open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def get_unit_registry() -> UnitRegistry:
    """Return a globally shared UnitRegistry configured from units.json."""
    registry = UnitRegistry()
    for quantity, metadata in _load_unit_aliases().items():
        target = metadata["default"]
        for alias in metadata.get("aliases", []):
            alias_norm = alias.strip()
            if not alias_norm or alias_norm == target:
                continue
            try:
                registry.define(f"{alias_norm} = {target}")
            except Exception as exc:
                raise ValueError(
                    f"Failed to register alias '{alias_norm}' for quantity '{quantity}'"
                ) from exc
    registry.default_format = "~P"
    return registry


def canonical_unit(unit_str: str) -> str:
    """
    Return the canonical unit label for a provided string.

    Raises:
        ValueError: if the unit cannot be parsed.
    """
    registry = get_unit_registry()
    try:
        quantity = registry.Quantity(1, unit_str)
    except Exception as exc:
        raise ValueError(f"Unrecognized unit string: '{unit_str}'") from exc
    return f"{quantity.units:~P}"
