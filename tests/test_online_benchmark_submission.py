import unittest
from pathlib import Path

from online_benchmark_submission import normalize_results, resolve_targets


class OnlineBenchmarkSubmissionTests(unittest.TestCase):
    def test_normalize_full_runner_payload(self):
        payload = {
            "comprehensive_report": {
                "overall_performance": {"overall_accuracy": 81.25},
            },
            "individual_results": {
                "gsm8k": {"accuracy": 88.0, "total_samples": 100, "ech0_errors": 2},
                "arc_easy": {"accuracy": 76.5, "total_samples": 80, "ech0_errors": 0},
            },
        }

        normalized = normalize_results(Path("full_benchmark_results_123.json"), payload)

        self.assertEqual(normalized.source_file, "full_benchmark_results_123.json")
        self.assertEqual(normalized.overall_score_percent, 81.25)
        self.assertEqual(normalized.metrics_percent["gsm8k"], 88.0)
        self.assertEqual(normalized.metrics_percent["arc_easy"], 76.5)
        self.assertEqual(normalized.sample_counts["gsm8k"], 98)
        self.assertEqual(normalized.sample_counts["arc_easy"], 80)

    def test_normalize_ai_suite_payload(self):
        payload = {
            "overall_score": 54.2,
            "results": {
                "gsm8k": {"score": 61.0, "total_questions": 50},
                "mmlu_philosophy": {"score": 47.5, "total_questions": 30},
            },
        }

        normalized = normalize_results(Path("benchmark_results_123.json"), payload)

        self.assertEqual(normalized.overall_score_percent, 54.2)
        self.assertEqual(
            normalized.metrics_percent,
            {
                "gsm8k": 61.0,
                "mmlu_philosophy": 47.5,
            },
        )
        self.assertEqual(
            normalized.sample_counts,
            {"gsm8k": 50, "mmlu_philosophy": 30},
        )

    def test_resolve_targets_supports_legacy_mapping(self):
        resolved = resolve_targets(["all"], "huggingface")
        self.assertEqual(resolved, ["hf_open_llm_leaderboard"])


if __name__ == "__main__":
    unittest.main()
