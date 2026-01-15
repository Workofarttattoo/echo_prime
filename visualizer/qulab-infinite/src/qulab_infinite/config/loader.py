"""Helpers for loading repository-scoped configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def repo_root() -> Path:
    """Return the root of the visualizer repository."""
    return Path(__file__).resolve().parents[4]


def load_units() -> Dict[str, Any]:
    """Load the canonical units map shared with Node tooling."""
    units_path = repo_root() / "config" / "units.json"
    with units_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
