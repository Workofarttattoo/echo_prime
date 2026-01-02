"""
Configuration helpers for the cognitive architecture.

Provides centralized dimension presets so the hierarchy can scale from a
resource-heavy research build to a lightweight local build.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Tuple

DimensionSpec = Tuple[int, int, str]

# Presets are defined as (input_dim, output_dim, level_name)
_COGNITIVE_DIMENSION_PRESETS: dict[str, List[DimensionSpec]] = {
    "full": [
        (1_000_000, 100_000, "Sensory"),
        (100_000, 10_000, "Perceptual"),
        (10_000, 1_000, "Associative"),
        (1_000, 100, "Prefrontal"),
        (100, 10, "Meta"),
    ],
    "balanced": [
        (65_536, 16_384, "Sensory"),
        (16_384, 4_096, "Perceptual"),
        (4_096, 1_024, "Associative"),
        (1_024, 256, "Prefrontal"),
        (256, 64, "Meta"),
    ],
    "lite": [
        (16_384, 4_096, "Sensory"),
        (4_096, 1_024, "Perceptual"),
        (1_024, 256, "Associative"),
        (256, 64, "Prefrontal"),
        (64, 16, "Meta"),
    ],
}

_DEFAULT_PROFILE = os.environ.get("ECH0_DIM_PROFILE", "lite").lower()


@lru_cache(maxsize=None)
def get_dimension_profile(profile: str | None = None) -> tuple[List[DimensionSpec], str]:
    """
    Return the configured dimension profile.

    Args:
        profile: Optional override (e.g., \"balanced\" or \"full\").
    """
    requested = (profile or _DEFAULT_PROFILE).lower()
    if requested not in _COGNITIVE_DIMENSION_PRESETS:
        requested = "lite"
    return list(_COGNITIVE_DIMENSION_PRESETS[requested]), requested


def get_available_profiles() -> List[str]:
    """Return the list of valid dimension profiles."""
    return list(_COGNITIVE_DIMENSION_PRESETS.keys())


def get_sensory_dim(profile: str | None = None) -> int:
    """Convenience helper for consumers that only need the Level-0 dimension."""
    dims, _ = get_dimension_profile(profile)
    return dims[0][0]


__all__ = [
    "DimensionSpec",
    "get_dimension_profile",
    "get_available_profiles",
    "get_sensory_dim",
]
