"""
Utility helpers for configuring Matplotlib to use a writable cache directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def ensure_mpl_config_dir(subdir_name: str = ".matplotlib_cache") -> str:
    """
    Ensure Matplotlib can write to its configuration/cache directory.

    Matplotlib defaults to ~/.matplotlib, which may not be writable inside
    sandboxes or constrained environments. This helper picks (or overrides)
    MPLCONFIGDIR so it always points to a writable path inside the project.
    """
    project_root = Path(__file__).resolve().parent
    default_dir = project_root / subdir_name

    configured_dir: Optional[str] = os.environ.get("MPLCONFIGDIR")
    candidate_paths = []

    if configured_dir:
        candidate_paths.append(Path(configured_dir))
    candidate_paths.append(default_dir)

    last_error: Optional[Exception] = None

    for candidate in candidate_paths:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(candidate)
            return str(candidate)
        except OSError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Unable to create a writable Matplotlib config directory "
        f"(last tried {candidate_paths[-1]!s}): {last_error}"
    )


__all__ = ["ensure_mpl_config_dir"]
