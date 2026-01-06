#!/usr/bin/env python3
"""
Generate a consolidated "final supremacy" report + verified results JSON.

This script is intentionally offline/local-only. It aggregates:
- Latest AI benchmark suite output (benchmark_results_*.json)
- Latest HLE output (benchmark_results/hle_results_*.json)
- Wisdom processing report (research_drop/processed/wisdom_processing_report.json)
- Skim-ingestion summaries from logs (logs/ingest_local_*.log)
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_glob(pattern: str) -> Optional[Path]:
    matches = [Path(p) for p in glob.glob(str(ROOT / pattern))]
    matches = [p for p in matches if p.exists()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_ingest_summary(log_path: Path) -> Optional[Dict[str, int]]:
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = re.search(r"Processed\s+(\d+)\s+pertinent files,\s+stored\s+(\d+)\s+chunks", text)
    if not m:
        return None
    return {"pertinent_files": int(m.group(1)), "stored_chunks": int(m.group(2))}


def main() -> int:
    out_dir = ROOT
    (ROOT / "benchmark_results").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)

    suite_path = _latest_glob("benchmark_results_*.json")
    hle_path = _latest_glob("benchmark_results/hle_results_*.json")
    wisdom_path = ROOT / "research_drop" / "processed" / "wisdom_processing_report.json"

    ingest_workspace_log = ROOT / "logs" / "ingest_local_workspace.log"
    ingest_home_log = ROOT / "logs" / "ingest_local_home.log"

    suite = _load_json(suite_path) if suite_path else None
    hle = _load_json(hle_path) if hle_path else None
    wisdom = _load_json(wisdom_path) if wisdom_path.exists() else None

    ingest_workspace = _parse_ingest_summary(ingest_workspace_log)
    ingest_home = _parse_ingest_summary(ingest_home_log)

    verified: Dict[str, Any] = {
        "timestamp": int(time.time()),
        "artifacts": {},
        "wisdom_processing": wisdom,
        "ingestion": {
            "workspace": ingest_workspace,
            "home": ingest_home,
        },
        "benchmarks": {
            "HLE_100": hle,
            "AI_Suite": suite,
        },
        "notes": [],
    }

    if suite_path:
        verified["artifacts"]["ai_suite"] = {
            "path": str(suite_path.relative_to(ROOT)),
            "sha256": _sha256_file(suite_path),
        }
    else:
        verified["notes"].append("AI suite results not found (expected benchmark_results_*.json).")

    if hle_path:
        verified["artifacts"]["hle"] = {
            "path": str(hle_path.relative_to(ROOT)),
            "sha256": _sha256_file(hle_path),
        }
    else:
        verified["notes"].append("HLE results not found (expected benchmark_results/hle_results_*.json).")

    if wisdom_path.exists():
        verified["artifacts"]["wisdom_report"] = {
            "path": str(wisdom_path.relative_to(ROOT)),
            "sha256": _sha256_file(wisdom_path),
        }

    # Write verified JSON
    verified_json_path = out_dir / "verified_benchmark_results.json"
    with open(verified_json_path, "w", encoding="utf-8") as f:
        json.dump(verified, f, indent=2)

    # Write concise markdown report
    md_path = out_dir / "FINAL_SUPREMACY_REPORT.md"
    lines = []
    lines.append("## Final Supremacy Report\n")
    lines.append(f"- **timestamp**: `{verified['timestamp']}`")
    if suite and isinstance(suite, dict):
        lines.append(f"- **AI suite overall score**: **{suite.get('overall_score', 'n/a')}%**")
        lines.append(f"- **AI suite total questions**: **{suite.get('total_questions', 'n/a')}**")
        lines.append(f"- **AI suite model**: `{suite.get('model_used', 'n/a')}`")
    if hle and isinstance(hle, dict):
        lines.append(f"- **HLE samples**: **{hle.get('num_samples', 'n/a')}**")
        lines.append(f"- **HLE percent score**: **{hle.get('percent_score', 'n/a')}%**")
    if ingest_workspace:
        lines.append(f"- **skim ingestion (workspace)**: **{ingest_workspace['pertinent_files']} files**, **{ingest_workspace['stored_chunks']} chunks**")
    if ingest_home:
        lines.append(f"- **skim ingestion (home)**: **{ingest_home['pertinent_files']} files**, **{ingest_home['stored_chunks']} chunks**")
    if wisdom and isinstance(wisdom, dict):
        ps = wisdom.get("processing_stats", {})
        ms = wisdom.get("memory_stats", {})
        lines.append(f"- **wisdom processed**: **{ps.get('files_processed', 'n/a')} files** (pdfs={ps.get('pdfs_found','n/a')}, jsons={ps.get('jsons_found','n/a')})")
        lines.append(f"- **wisdom memory**: episodic={ms.get('episodic_memories','n/a')}, semantic={ms.get('semantic_concepts','n/a')}")

    if verified["notes"]:
        lines.append("\n### Notes\n")
        for n in verified["notes"]:
            lines.append(f"- {n}")

    lines.append("\n### Artifact hashes\n")
    for k, v in verified.get("artifacts", {}).items():
        lines.append(f"- **{k}**: `{v['path']}` sha256=`{v['sha256']}`")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {verified_json_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




