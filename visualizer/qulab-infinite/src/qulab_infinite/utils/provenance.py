"""Deterministic hashing helpers for provenance tracking."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def deterministic_hash(payload: Dict[str, Any]) -> str:
    """
    Return a SHA256 hash built from a canonical JSON representation of the payload.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
