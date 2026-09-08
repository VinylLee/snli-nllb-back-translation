import unittest

from scripts.final_semantic_metric_bakeoff import (
    combination_decision,
    combination_metrics,
    expert_threshold_metrics,
    metric_distribution,
    percentile_rank,
    score_bleurt,
    select_final_disagreement,
    threshold_sweep,
)


class FakeRawScorer:
    revision = "fake-revision"

    def predict_raw(self, pairs):
        return [-0.75, 0.25][:len(pairs)]


class FinalBakeoffHelpersTest(unittest.TestCase):
    def test_expert_metrics_exclude_borderline_from_pass_fail_precision(self):
        rows = [
            {"expert_semantic_verdict": "FAIL", "secondary_sts_score": "0.70"},
            {"expert_semantic_verdict": "PASS", "secondary_sts_score": "0.90"},
            {"expert_semantic_verdict": "BORDERLINE", "secondary_sts_score": "0.70"},
        ]
        result = expert_threshold_metrics(rows, "secondary_sts_score", 0.75)
        self.assertEqual(result["fail_rejected"], 1)
        self.assertEqual(result["pass_falsely_rejected"], 0)
        self.assertEqual(result["borderline_rejected"], 1)
        self.assertEqual(result["rejection_precision_pass_fail"], 1.0)

    def test_bleurt_raw_scores_are_not_sigmoided(self):
        rows = [
            {"original_sentence": "a", "back_translated_sentence": "b"},
            {"original_sentence": "c", "back_translated_sentence": "d"},
        ]
        elapsed = score_bleurt(FakeRawScorer(), rows)
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(rows[0]["secondary_bleurt_score"], -0.75)
        self.assertEqual(rows[1]["secondary_bleurt_score"], 0.25)

    def test_threshold_sweep_and_percentile(self):
        rows = [
            {"expert_semantic_verdict": "FAIL", "secondary_sts_large_score": "0.60", "accepted": "True"},
            {"expert_semantic_verdict": "PASS", "secondary_sts_large_score": "0.90", "accepted": "True"},
            {"expert_semantic_verdict": "BORDERLINE", "secondary_sts_large_score": "0.70", "accepted": "False"},
        ]
        result = threshold_sweep(rows, "secondary_sts_large_score", (0.75,))[0]
        self.assertEqual(result["fail_rejected"], 1)
        self.assertEqual(result["pass_falsely_rejected"], 0)
        self.assertEqual(result["borderline_rejected"], 1)
        accepted_population = [
            {"accepted": "True", "secondary_sts_large_score": "0.90"},
            {"accepted": "True", "secondary_sts_large_score": "0.70"},
            {"accepted": "True", "secondary_sts_large_score": "0.50"},
        ]
        result = threshold_sweep(rows, "secondary_sts_large_score", (0.75,), accepted_population)[0]
        self.assertEqual(result["retained"], 1)
        self.assertEqual(result["newly_rejected"], 2)
        self.assertAlmostEqual(percentile_rank(0.75, [0.5, 0.75, 1.0]), 66.66666666666667)

    def test_combination_modes(self):
        row = {"secondary_sts_score": "0.70", "secondary_bleurt_score": "0.90"}
        # either = conservative OR-risk: reject if either metric fails.
        self.assertFalse(combination_decision(row, 0.75, 0.80, "either"))
        # both = permissive AND-risk: reject only if both metrics fail.
        self.assertTrue(combination_decision(row, 0.75, 0.80, "both"))
        metrics = combination_metrics(
            [{**row, "expert_semantic_verdict": "FAIL", "accepted": "True"}],
            0.75, 0.80, "either",
        )
        self.assertEqual(metrics["fail_rejected"], 1)

    def test_disagreement_selection_is_analysis_only_and_capped_by_caller(self):
        rows = [
            {"candidate_id": "a", "source_index": "517858", "accepted": "True",
             "secondary_sts_score": "0.90", "secondary_sts_large_score": "0.90",
             "secondary_bleurt_score": "0.1"},
            {"candidate_id": "b", "source_index": "1", "accepted": "False",
             "secondary_sts_score": "0.1", "secondary_sts_large_score": "0.1",
             "secondary_bleurt_score": "0.1"},
        ]
        selected = select_final_disagreement(rows, 0.2)
        self.assertEqual([row["candidate_id"] for row in selected], ["a"])
        self.assertIn("known_risk_source", selected[0]["final_disagreement_reasons"])

    def test_distribution_has_requested_quantiles(self):
        stats = metric_distribution([{"secondary_sts_score": str(x)} for x in range(4)], "sts_base")
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["median"], 1.5)
        self.assertIn("p05", stats)
        self.assertIn("p90", stats)


if __name__ == "__main__":
    unittest.main()
