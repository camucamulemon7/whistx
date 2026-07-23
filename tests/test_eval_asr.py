from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("eval_asr", ROOT / "scripts" / "eval_asr.py")
assert SPEC and SPEC.loader
eval_asr = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = eval_asr
SPEC.loader.exec_module(eval_asr)


class AsrEvaluationTests(unittest.TestCase):
    def test_report_combines_accuracy_latency_cost_and_failures(self) -> None:
        cases = eval_asr.discover_cases(ROOT / "tests" / "fixtures" / "asr_eval")
        results = [eval_asr.evaluate_case(case) for case in cases]
        summary = eval_asr.build_summary(results)

        self.assertEqual(summary["sampleCount"], 1)
        self.assertIn("namedEntityAccuracy", summary)
        self.assertEqual(summary["speakerErrorRate"], 0.0)
        self.assertEqual(summary["apiRequests"], 1)
        self.assertEqual(summary["latencyMs"], 840.0)
        self.assertAlmostEqual(summary["estimatedCostUsd"], 0.0012)

    def test_failure_classifier_covers_common_regressions(self) -> None:
        failures = eval_asr.classify_failures(
            reference="正常な文章です",
            hypothesis="壊れた\ufffd文章文章文章",
            cer=0.7,
            missing_rate=0.0,
            duplicate_rate=0.5,
            boundary_fragment_rate=0.5,
            metadata={"timestampMonotonic": False},
        )
        self.assertEqual(
            failures,
            ["duplicate", "boundary_fragment", "encoding", "high_error", "timestamp_inconsistent"],
        )


if __name__ == "__main__":
    unittest.main()
