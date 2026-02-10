#!/usr/bin/env python3
"""
Community benchmark onboarding packet generator.

This script does not submit to third-party services directly. Instead, it
produces submission-ready artifacts and checklists for the highest-signal
community benchmark destinations (LM Arena, Open LLM Leaderboard, LiveBench,
AlpacaEval 2, and LM Evaluation Harness workflows).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TARGET_ORDER = [
    "lmarena",
    "hf_open_llm_leaderboard",
    "livebench",
    "alpaca_eval_2",
    "lm_eval_harness",
]


TARGET_SPECS: Dict[str, Dict[str, Any]] = {
    "lmarena": {
        "display_name": "LM Arena (LMSYS / LM Arena)",
        "priority": 1,
        "submission_url": "https://lmarena.ai/",
        "why_it_matters": (
            "Highest-visibility community preference benchmark for chat quality."
        ),
        "submission_type": "Manual managed onboarding",
        "required_inputs": [
            "Public model identifier (for example, Hugging Face model ID).",
            "Stable inference endpoint with deterministic defaults.",
            "System prompt and decoding parameter defaults.",
            "Safety and acceptable-use policy links.",
        ],
        "action_items": [
            "Open the LM Arena website and locate the latest model intake form.",
            "Provide model metadata, endpoint details, and evaluation defaults.",
            "Respond to organizer follow-up for validation and launch window.",
            "Track listing status until the model appears on the public board.",
        ],
    },
    "hf_open_llm_leaderboard": {
        "display_name": "Hugging Face Open LLM Leaderboard",
        "priority": 2,
        "submission_url": (
            "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"
        ),
        "why_it_matters": (
            "Widely referenced open benchmark hub with transparent evaluation settings."
        ),
        "submission_type": "Self-serve submission",
        "required_inputs": [
            "Public Hugging Face model repository.",
            "Model card with license, architecture, and usage constraints.",
            "Tokenizer and generation configuration.",
            "Optional: precomputed local evals for sanity checking.",
        ],
        "action_items": [
            "Ensure model repo and model card are complete and public.",
            "Submit model ID through the leaderboard Space workflow.",
            "Monitor evaluation queue and verify results after completion.",
            "Record final leaderboard URL in internal status tracker.",
        ],
    },
    "livebench": {
        "display_name": "LiveBench",
        "priority": 3,
        "submission_url": "https://livebench.ai/",
        "why_it_matters": (
            "Community benchmark focused on freshness and contamination resistance."
        ),
        "submission_type": "Project workflow + manual publication",
        "required_inputs": [
            "Reproducible evaluation command and environment details.",
            "Model endpoint or local runner configuration.",
            "Machine-readable result artifact and run logs.",
        ],
        "action_items": [
            "Run the official LiveBench evaluation workflow.",
            "Archive raw outputs, scoring report, and runtime metadata.",
            "Follow LiveBench publication instructions for result inclusion.",
            "Link published results in internal benchmark tracker.",
        ],
    },
    "alpaca_eval_2": {
        "display_name": "AlpacaEval 2.0",
        "priority": 4,
        "submission_url": "https://tatsu-lab.github.io/alpaca_eval/",
        "why_it_matters": (
            "Standardized pairwise preference benchmark used in model release reporting."
        ),
        "submission_type": "Self-run evaluation + public result sharing",
        "required_inputs": [
            "Model outputs on AlpacaEval prompts.",
            "Evaluation config (judge model and protocol).",
            "Win-rate summary and confidence intervals.",
        ],
        "action_items": [
            "Generate model outputs on the AlpacaEval prompt set.",
            "Run AlpacaEval 2 evaluation pipeline and capture reports.",
            "Publish results alongside full methodology details.",
            "Reference the report in release notes and benchmark tracker.",
        ],
    },
    "lm_eval_harness": {
        "display_name": "EleutherAI LM Evaluation Harness track",
        "priority": 5,
        "submission_url": "https://github.com/EleutherAI/lm-evaluation-harness",
        "why_it_matters": (
            "Most-used open LLM benchmarking framework and baseline compatibility layer."
        ),
        "submission_type": "Code-based reproducible evaluation",
        "required_inputs": [
            "Model endpoint adapter (hf/vllm/local) with pinned settings.",
            "Task list and few-shot configuration.",
            "JSON result dump with task-level metrics.",
        ],
        "action_items": [
            "Run lm-evaluation-harness on agreed community task set.",
            "Export raw results JSON and command invocation metadata.",
            "Attach results to model card and benchmark packet.",
            "Reuse these artifacts when submitting to other boards.",
        ],
    },
}


LEGACY_LEADERBOARD_TO_TARGET = {
    "all": TARGET_ORDER,
    "huggingface": ["hf_open_llm_leaderboard"],
    "papers_with_code": ["livebench"],
    "eleuther_ai": ["lm_eval_harness"],
    "custom": ["lmarena"],
}


DEFAULT_MODEL_ID = "ech0prime/ech0-prime-csa"
DEFAULT_MODEL_NAME = "ECH0-PRIME"


@dataclass
class NormalizedResults:
    source_file: Optional[str]
    metrics_percent: Dict[str, float]
    sample_counts: Dict[str, int]
    overall_score_percent: Optional[float]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "metrics_percent": self.metrics_percent,
            "sample_counts": self.sample_counts,
            "overall_score_percent": self.overall_score_percent,
            "notes": self.notes,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_percent(value: Any) -> Optional[float]:
    if not _is_number(value):
        return None
    numeric = float(value)
    if 0.0 <= numeric <= 1.0:
        numeric *= 100.0
    return round(numeric, 4)


def _safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "__").replace(" ", "_")


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    if text.startswith("version https://git-lfs.github.com/spec/v1"):
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return payload
    return None


def find_latest_results_file() -> Optional[Tuple[Path, Dict[str, Any]]]:
    patterns = [
        "full_benchmark_results_*.json",
        "benchmark_results_*.json",
        "benchmark_execution_summary.json",
        "benchmark_results/*.json",
    ]

    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(Path(".").glob(pattern))

    if not candidates:
        return None

    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        payload = _read_json_file(path)
        if payload is not None:
            return path, payload
    return None


def _extract_from_full_runner(payload: Dict[str, Any], notes: List[str]) -> Tuple[Dict[str, float], Dict[str, int], Optional[float]]:
    metrics: Dict[str, float] = {}
    sample_counts: Dict[str, int] = {}
    overall: Optional[float] = None

    comp_report = payload.get("comprehensive_report", {})
    if isinstance(comp_report, dict):
        overall = _to_percent(comp_report.get("overall_performance", {}).get("overall_accuracy"))

    individual_results = payload.get("individual_results")
    if not isinstance(individual_results, dict):
        individual_results = comp_report.get("dataset_breakdown", {})

    if not isinstance(individual_results, dict):
        return metrics, sample_counts, overall

    for dataset, result in individual_results.items():
        if not isinstance(result, dict):
            continue

        accuracy = _to_percent(result.get("accuracy"))
        if accuracy is not None:
            metrics[dataset] = accuracy

        total = result.get("total_samples")
        errors = result.get("ech0_errors", 0)
        if _is_number(total):
            evaluated = int(float(total) - float(errors if _is_number(errors) else 0))
            sample_counts[dataset] = max(evaluated, 0)

    if metrics:
        notes.append("Parsed full_benchmark_runner output schema.")
    return metrics, sample_counts, overall


def _extract_from_ai_suite(payload: Dict[str, Any], notes: List[str]) -> Tuple[Dict[str, float], Dict[str, int], Optional[float]]:
    metrics: Dict[str, float] = {}
    sample_counts: Dict[str, int] = {}
    overall: Optional[float] = _to_percent(payload.get("overall_score"))

    results = payload.get("results")
    if not isinstance(results, dict):
        return metrics, sample_counts, overall

    for benchmark, result in results.items():
        if not isinstance(result, dict):
            continue
        score = _to_percent(result.get("score"))
        if score is not None:
            metrics[benchmark] = score
        total_questions = result.get("total_questions")
        if _is_number(total_questions):
            sample_counts[benchmark] = int(total_questions)

    if metrics:
        notes.append("Parsed ai_benchmark_suite output schema.")
    return metrics, sample_counts, overall


def _extract_from_execution_summary(payload: Dict[str, Any], notes: List[str]) -> Tuple[Dict[str, float], Dict[str, int], Optional[float]]:
    metrics: Dict[str, float] = {}
    sample_counts: Dict[str, int] = {}

    perf = payload.get("performance_metrics")
    if not isinstance(perf, dict):
        return metrics, sample_counts, None

    for key, value in perf.items():
        if not key.endswith("_accuracy"):
            continue
        benchmark = key[: -len("_accuracy")]
        score = _to_percent(value)
        if score is not None:
            metrics[benchmark] = score

    overall = None
    if metrics:
        overall = round(sum(metrics.values()) / len(metrics), 4)
        notes.append("Parsed benchmark_execution_summary schema.")

    return metrics, sample_counts, overall


def normalize_results(source_file: Optional[Path], payload: Optional[Dict[str, Any]]) -> NormalizedResults:
    if not source_file or not payload:
        return NormalizedResults(
            source_file=None,
            metrics_percent={},
            sample_counts={},
            overall_score_percent=None,
            notes=["No result file supplied; created packet with placeholders."],
        )

    notes: List[str] = []
    metrics: Dict[str, float] = {}
    sample_counts: Dict[str, int] = {}
    overall_score: Optional[float] = None

    for extractor in (
        _extract_from_full_runner,
        _extract_from_ai_suite,
        _extract_from_execution_summary,
    ):
        extracted_metrics, extracted_counts, extracted_overall = extractor(payload, notes)
        metrics.update(extracted_metrics)
        sample_counts.update(extracted_counts)
        if overall_score is None and extracted_overall is not None:
            overall_score = extracted_overall

    if not metrics:
        notes.append(
            "Result schema not recognized; generated packet without benchmark metrics."
        )

    return NormalizedResults(
        source_file=str(source_file),
        metrics_percent=metrics,
        sample_counts=sample_counts,
        overall_score_percent=overall_score,
        notes=notes,
    )


def resolve_targets(explicit_targets: List[str], legacy_leaderboard: Optional[str]) -> List[str]:
    if explicit_targets and explicit_targets != ["all"]:
        requested = explicit_targets
    elif legacy_leaderboard:
        requested = LEGACY_LEADERBOARD_TO_TARGET.get(legacy_leaderboard, TARGET_ORDER)
    else:
        requested = TARGET_ORDER

    if "all" in requested:
        return list(TARGET_ORDER)

    invalid = [target for target in requested if target not in TARGET_SPECS]
    if invalid:
        valid = ", ".join(["all"] + TARGET_ORDER)
        raise ValueError(f"Unknown target(s): {', '.join(invalid)}. Valid values: {valid}")

    deduped: List[str] = []
    for target in requested:
        if target not in deduped:
            deduped.append(target)
    return deduped


def _benchmark_table_lines(normalized: NormalizedResults) -> List[str]:
    if not normalized.metrics_percent:
        return ["No normalized benchmark metrics available yet."]

    lines = ["benchmark,score_percent,samples"]
    for benchmark in sorted(normalized.metrics_percent):
        score = normalized.metrics_percent[benchmark]
        samples = normalized.sample_counts.get(benchmark, "")
        lines.append(f"{benchmark},{score:.4f},{samples}")
    return lines


def write_target_packet(
    target_dir: Path,
    target_key: str,
    model_context: Dict[str, Any],
    normalized: NormalizedResults,
) -> None:
    target = TARGET_SPECS[target_key]
    target_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": utc_now_iso(),
        "target_key": target_key,
        "target_name": target["display_name"],
        "target_priority": target["priority"],
        "submission_url": target["submission_url"],
        "model": model_context,
        "normalized_results": normalized.to_dict(),
        "required_inputs": target["required_inputs"],
    }
    (target_dir / "submission_payload.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    checklist_lines = [
        f"# {target['display_name']} submission checklist",
        "",
        f"Submission URL: {target['submission_url']}",
        f"Submission type: {target['submission_type']}",
        "",
        "Why this benchmark matters:",
        f"- {target['why_it_matters']}",
        "",
        "Required inputs:",
    ]
    checklist_lines.extend(f"- [ ] {item}" for item in target["required_inputs"])
    checklist_lines.append("")
    checklist_lines.append("Action items:")
    checklist_lines.extend(f"{idx}. {item}" for idx, item in enumerate(target["action_items"], start=1))
    checklist_lines.append("")
    checklist_lines.append("Normalized benchmark snapshot:")
    checklist_lines.extend(f"- `{line}`" for line in _benchmark_table_lines(normalized))
    checklist_lines.append("")

    (target_dir / "README.md").write_text(
        "\n".join(checklist_lines), encoding="utf-8"
    )


def write_manifest(
    packet_dir: Path,
    selected_targets: List[str],
    model_context: Dict[str, Any],
    normalized: NormalizedResults,
    benchmark_name: Optional[str],
) -> None:
    manifest = {
        "generated_at_utc": utc_now_iso(),
        "benchmark_name": benchmark_name or "",
        "model": model_context,
        "selected_targets": selected_targets,
        "target_specs": {key: TARGET_SPECS[key] for key in selected_targets},
        "normalized_results": normalized.to_dict(),
        "next_steps": [
            "Fill any missing model metadata in this packet.",
            "Complete each target checklist under targets/<target>/README.md.",
            "Submit to targets in priority order and track confirmation links.",
        ],
    }
    (packet_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def write_packet_readme(
    packet_dir: Path,
    selected_targets: List[str],
    normalized: NormalizedResults,
) -> None:
    lines = [
        "# Community benchmark onboarding packet",
        "",
        f"Generated at (UTC): {utc_now_iso()}",
        "",
        "Priority targets:",
    ]
    for target in selected_targets:
        spec = TARGET_SPECS[target]
        lines.append(
            f"{spec['priority']}. {spec['display_name']} ({spec['submission_url']})"
        )

    lines.append("")
    lines.append("Normalized benchmark snapshot:")
    for row in _benchmark_table_lines(normalized):
        lines.append(f"- `{row}`")

    lines.append("")
    lines.append("Packet contents:")
    lines.append("- submission_manifest.json")
    lines.append("- targets/<target>/submission_payload.json")
    lines.append("- targets/<target>/README.md")

    (packet_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_announcement_template(
    packet_dir: Path,
    model_name: str,
    selected_targets: List[str],
    normalized: NormalizedResults,
) -> None:
    top_metrics = sorted(
        normalized.metrics_percent.items(), key=lambda item: item[1], reverse=True
    )[:5]

    lines = [
        f"# {model_name} community benchmark launch draft",
        "",
        "Planned benchmark submissions:",
    ]
    lines.extend(
        f"- {TARGET_SPECS[target]['display_name']}" for target in selected_targets
    )
    lines.append("")
    lines.append("Current internal benchmark snapshot:")
    if top_metrics:
        lines.extend(f"- {name}: {score:.2f}%" for name, score in top_metrics)
    else:
        lines.append("- Metrics pending; run benchmark suite first.")
    lines.append("")
    lines.append(
        "We are submitting this model to independent community benchmarks for transparent external validation."
    )

    (packet_dir / "announcement_template.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def build_packet(
    output_root: Path,
    model_context: Dict[str, Any],
    normalized: NormalizedResults,
    selected_targets: List[str],
    benchmark_name: Optional[str],
    include_announcement: bool,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_fragment = _safe_model_dir(model_context["model_id"])
    packet_dir = output_root / f"{timestamp}_{model_fragment}"
    targets_root = packet_dir / "targets"
    targets_root.mkdir(parents=True, exist_ok=True)

    write_manifest(packet_dir, selected_targets, model_context, normalized, benchmark_name)
    write_packet_readme(packet_dir, selected_targets, normalized)

    for target in selected_targets:
        write_target_packet(targets_root / target, target, model_context, normalized)

    if include_announcement:
        write_announcement_template(
            packet_dir=packet_dir,
            model_name=model_context["model_name"],
            selected_targets=selected_targets,
            normalized=normalized,
        )

    return packet_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate community benchmark onboarding artifacts."
    )
    parser.add_argument(
        "--target",
        nargs="+",
        default=["all"],
        help=(
            "Target community benchmark(s). "
            "Values: all, " + ", ".join(TARGET_ORDER)
        ),
    )
    parser.add_argument(
        "--leaderboard",
        choices=list(LEGACY_LEADERBOARD_TO_TARGET),
        help=(
            "Deprecated alias used by older scripts. "
            "Mapped to modern targets."
        ),
    )
    parser.add_argument(
        "--benchmark",
        help=(
            "Optional benchmark label for packet metadata. "
            "Retained for backwards compatibility."
        ),
    )
    parser.add_argument("--results-file", help="Path to benchmark results JSON.")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/community_submissions",
        help="Directory where submission packet will be written.",
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("MODEL_ID", DEFAULT_MODEL_ID),
        help="Public model identifier (for example, org/model-name).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Human-readable model name.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.getenv("MODEL_ENDPOINT_URL", ""),
        help="Public inference endpoint URL, if available.",
    )
    parser.add_argument(
        "--contact-email",
        default=os.getenv("BENCHMARK_CONTACT_EMAIL", ""),
        help="Contact email for benchmark organizers.",
    )
    parser.add_argument(
        "--license",
        dest="license_name",
        default="Proprietary",
        help="Model license name.",
    )
    parser.add_argument(
        "--announce",
        action="store_true",
        help="Generate announcement_template.md in the packet.",
    )
    return parser.parse_args()


def load_results_from_args(args: argparse.Namespace) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    if args.results_file:
        candidate = Path(args.results_file)
        payload = _read_json_file(candidate)
        return candidate if payload is not None else None, payload

    discovered = find_latest_results_file()
    if discovered:
        return discovered
    return None, None


def main() -> int:
    args = parse_args()

    try:
        selected_targets = resolve_targets(args.target, args.leaderboard)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2

    result_path, payload = load_results_from_args(args)
    normalized = normalize_results(result_path, payload)

    model_context = {
        "model_id": args.model_id,
        "model_name": args.model_name,
        "endpoint_url": args.endpoint_url,
        "contact_email": args.contact_email,
        "license": args.license_name,
    }

    output_root = Path(args.output_dir)
    packet_dir = build_packet(
        output_root=output_root,
        model_context=model_context,
        normalized=normalized,
        selected_targets=selected_targets,
        benchmark_name=args.benchmark,
        include_announcement=args.announce,
    )

    print("Community benchmark onboarding packet generated.")
    print(f"Packet directory: {packet_dir}")
    print(f"Targets: {', '.join(selected_targets)}")
    if normalized.source_file:
        print(f"Results source: {normalized.source_file}")
    else:
        print("Results source: none (placeholder packet)")
    if normalized.overall_score_percent is not None:
        print(f"Overall score: {normalized.overall_score_percent:.2f}%")
    if normalized.notes:
        print("Notes:")
        for note in normalized.notes:
            print(f"- {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
