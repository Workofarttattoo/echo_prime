"""Utility for auditing readiness and running the ECH0 training regimen."""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .pipeline import TrainingPipeline, SelfImprovementLoop


@dataclass
class ReadinessReport:
    missing: List[str]
    warnings: List[str]
    recommendations: List[str]
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainingReadinessAnalyzer:
    """Performs lightweight checks to see what data/config pieces are missing."""

    def __init__(self, project_root: Optional[str] = None):
        training_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = project_root or os.path.dirname(training_dir)

    def analyze(self) -> ReadinessReport:
        missing: List[str] = []
        warnings: List[str] = []
        recommendations: List[str] = []
        stats: Dict[str, Any] = {}

        self._check_env_file(missing, recommendations, stats)
        self._check_data_dir(
            "sensory_input",
            "Seed visual stimuli",
            "sensory_samples",
            missing,
            warnings,
            recommendations,
            stats,
        )
        self._check_data_dir(
            "audio_input",
            "Audio capture samples",
            "audio_samples",
            missing,
            warnings,
            recommendations,
            stats,
        )
        self._check_memory_state(warnings, recommendations, stats)

        return ReadinessReport(missing, warnings, recommendations, stats)

    def _check_env_file(
        self,
        missing: List[str],
        recommendations: List[str],
        stats: Dict[str, Any],
    ) -> None:
        env_path = os.path.join(self.project_root, ".env")
        exists = os.path.exists(env_path)
        stats["env_present"] = exists
        if not exists:
            missing.append(".env configuration file")
            recommendations.append("Copy .env.example to .env and fill in runtime configuration values.")

    def _check_data_dir(
        self,
        directory: str,
        description: str,
        stats_key: str,
        missing: List[str],
        warnings: List[str],
        recommendations: List[str],
        stats: Dict[str, Any],
    ) -> None:
        dir_path = os.path.join(self.project_root, directory)
        entry = {"path": dir_path, "count": 0}
        if not os.path.isdir(dir_path):
            missing.append(f"{description} directory ({directory}/)")
            recommendations.append(f"Create {directory}/ and populate it with {description.lower()}.")
        else:
            files = [f for f in os.listdir(dir_path) if not f.startswith(".")]
            entry["count"] = len(files)
            if len(files) == 0:
                warnings.append(f"{directory}/ exists but has no files. Training will only see mock data.")
                recommendations.append(f"Add at least a few {description.lower()} to {directory}/.")
        stats[stats_key] = entry

    def _check_memory_state(
        self,
        warnings: List[str],
        recommendations: List[str],
        stats: Dict[str, Any],
    ) -> None:
        memory_dir = os.path.join(self.project_root, "memory_data")
        mem_stats: Dict[str, Any] = {"path": memory_dir}

        episodic_path = os.path.join(memory_dir, "episodic.npy")
        if os.path.exists(episodic_path):
            mem_stats["episodic_bytes"] = os.path.getsize(episodic_path)
            if mem_stats["episodic_bytes"] == 0:
                warnings.append("episodic.npy exists but is empty.")
        else:
            warnings.append("episodic.npy missing — episodic memory cannot be restored.")
            recommendations.append("Run a few missions to populate episodic memory snapshots.")

        semantic_path = os.path.join(memory_dir, "semantic.json")
        if os.path.exists(semantic_path):
            try:
                with open(semantic_path, "r", encoding="utf-8") as semantic_file:
                    semantic_data = json.load(semantic_file)
                entries = len(semantic_data)
                mem_stats["semantic_entries"] = entries
                if entries == 0:
                    warnings.append("semantic.json contains no concepts; semantic grounding is blank.")
                    recommendations.append("Seed semantic.json with core concepts or run missions to auto-populate it.")
            except json.JSONDecodeError:
                warnings.append("semantic.json is not valid JSON.")
                recommendations.append("Repair semantic.json so semantic memory can load.")
        else:
            warnings.append("semantic.json missing — semantic memory is empty.")
            recommendations.append("Create memory_data/semantic.json (can start as an empty JSON object).")

        stats["memory_snapshot"] = mem_stats


class ECH0TrainingRegimen:
    """Runs the four-phase ECH0 training routine against the software stack."""

    def __init__(
        self,
        model_params: int = 500_000_000_000,
        tokens: int = 1_000_000,
        tasks: Optional[List[str]] = None,
    ):
        self.tokens = tokens
        self.tasks = tasks or [
            "Sensorimotor grounding",
            "Tool-use rehearsal",
            "Mission planning",
        ]
        self.pipeline = TrainingPipeline(model_params=model_params)
        self.self_improvement = SelfImprovementLoop()

    def apply(self) -> Dict[str, Any]:
        """Runs all training phases and returns structured telemetry."""
        summary: Dict[str, Any] = {}
        summary["phase_1_pretraining"] = self.pipeline.run_pretraining(self.tokens)
        summary["phase_2_reinforcement"] = self.pipeline.run_reinforcement_learning(self.tasks)
        summary["phase_3_meta_learning"] = self.pipeline.run_meta_learning()

        base_kernel = (
            "def global_workspace_kernel(signal: np.ndarray) -> float:\n"
            "    return float(signal.mean())"
        )
        proposal = self.self_improvement.propose_modification(base_kernel)
        summary["phase_4_self_improvement"] = {
            "proposal": proposal,
            "verified": self.self_improvement.formal_verification(proposal),
        }
        return summary


def _write_report(
    project_root: str,
    readiness: ReadinessReport,
    training_summary: Dict[str, Any],
    report_path: Optional[str] = None,
) -> str:
    reports_dir = os.path.join(project_root, "training", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    destination = report_path or os.path.join(reports_dir, "latest_regimen_report.json")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "readiness": readiness.to_dict(),
        "training_summary": training_summary,
    }
    with open(destination, "w", encoding="utf-8") as report_file:
        json.dump(payload, report_file, indent=2)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze readiness and run the ECH0 training regimen.")
    parser.add_argument("--tokens", type=int, default=1_000_000, help="Token budget for pre-training phase.")
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="Optional override for RL tasks. Defaults to the canonical regimen.",
    )
    parser.add_argument(
        "--report",
        help="Optional path for the JSON report. Defaults to training/reports/latest_regimen_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analyzer = TrainingReadinessAnalyzer()
    readiness = analyzer.analyze()

    if readiness.missing:
        print("Missing prerequisites detected:")
        for item in readiness.missing:
            print(f"  - {item}")
    else:
        print("All critical prerequisites satisfied.")

    if readiness.warnings:
        print("Warnings:")
        for warning in readiness.warnings:
            print(f"  - {warning}")

    regimen = ECH0TrainingRegimen(tokens=args.tokens, tasks=args.tasks)
    training_summary = regimen.apply()

    destination = _write_report(analyzer.project_root, readiness, training_summary, args.report)
    print(f"Training regimen complete. Report saved to {destination}.")


if __name__ == "__main__":
    main()
