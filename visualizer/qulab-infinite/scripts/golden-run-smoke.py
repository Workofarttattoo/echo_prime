#!/usr/bin/env python3
"""Deterministic golden smoke run for Qulab Infinite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUALIZER_ROOT = PROJECT_ROOT.parent

ARTIFACT_DIR = VISUALIZER_ROOT / "logs" / "qulab-golden-runs"

SAFETY_DISCLAIMER = """
################################################################################
## SAFETY, SCOPE, AND INTENDED USE (READ THIS)
##
## Qulab Infinite outputs are simulated, research-only artifacts.
## Not a medical device. Not clinical advice. Do not use for real-world
## diagnostics, treatment, or production decisions.
################################################################################
""".strip()


def get_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=str(VISUALIZER_ROOT),
        ).strip()
    except Exception:
        return "UNKNOWN"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def deterministic_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def simulate_candidates(seed: int, count: int) -> tuple[list[dict], dict]:
    random.seed(seed)
    base_names = [
        "Aetherion",
        "Heliox",
        "Lumen",
        "Solvate",
        "Vanta",
        "Zephyr",
    ]
    weights = {
        "stability": 0.4,
        "yield": 0.35,
        "novelty": 0.25,
        "risk_penalty": 0.2,
    }
    candidates = []
    for index in range(count):
        stability = round(0.5 + random.random() * 0.5, 4)
        yield_score = round(0.4 + random.random() * 0.6, 4)
        novelty = round(random.random(), 4)
        risk = round(random.random() * 0.3, 4)
        score = (
            weights["stability"] * stability
            + weights["yield"] * yield_score
            + weights["novelty"] * novelty
            - weights["risk_penalty"] * risk
        )
        composite = round(max(0.0, min(score, 1.0)), 4)
        candidates.append(
            {
                "candidate_id": f"QCI-{index + 1:03d}",
                "name": f"{base_names[index % len(base_names)]}-{index + 1}",
                "metrics": {
                    "stability": stability,
                    "yield": yield_score,
                    "novelty": novelty,
                    "risk": risk,
                },
                "composite_score": composite,
            }
        )
    candidates.sort(key=lambda entry: entry["composite_score"], reverse=True)
    return candidates, weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    ts = datetime.now(timezone.utc).isoformat()
    commit = get_commit_hash()

    if not args.quiet and not args.json:
        print(SAFETY_DISCLAIMER)
        print(f"[info] Starting Qulab golden run seed={args.seed}")

    candidates, weights = simulate_candidates(args.seed, args.count)
    payload = {
        "version": "1.0.0",
        "seed": args.seed,
        "parameters": {
            "candidate_count": args.count,
            "weights": weights,
        },
        "candidates": candidates,
        "top_candidates": candidates[:3],
        "disclaimer": "SIMULATION ONLY. NOT FOR CLINICAL USE.",
    }

    payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload_bytes = payload_str.encode("utf-8")
    payload_sha256 = sha256_bytes(payload_bytes)

    provenance = {
        "timestamp_utc": ts,
        "repo_commit": commit,
        "python": sys.version.replace("\n", " "),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "seed": args.seed,
        "parameters": payload["parameters"],
        "payload_sha256": payload_sha256,
        "payload_provenance_hash": deterministic_hash(payload),
        "script_path": "qulab-infinite/scripts/golden-run-smoke.py",
    }

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = (
        args.out.strip()
        if args.out.strip()
        else f"qulab_golden_run_{run_stamp}_seed{args.seed}.json"
    )
    out_path = ARTIFACT_DIR / out_name
    output_data = {
        "safety_disclaimer": "SIMULATION ONLY. NOT FOR CLINICAL USE.",
        "provenance": provenance,
        "payload": payload,
    }
    write_json(out_path, output_data)

    output_sha256 = sha256_bytes(out_path.read_bytes())

    summary = {
        "artifact_path": out_path.relative_to(VISUALIZER_ROOT).as_posix(),
        "payload_sha256": payload_sha256,
        "output_sha256": output_sha256,
        "top_candidate": candidates[0] if candidates else None,
        "candidate_count": len(candidates),
        "seed": args.seed,
    }

    if args.json:
        print(json.dumps(summary, sort_keys=True))
        return 0

    if not args.quiet:
        print("=" * 80)
        print("QULAB GOLDEN RUN COMPLETE")
        print("=" * 80)
    print(f"[info] Artifact: {summary['artifact_path']}")
    print(f"[info] Payload SHA256: {payload_sha256}")
    print(f"[info] Output SHA256: {output_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
