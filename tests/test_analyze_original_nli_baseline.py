import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_original_nli_baseline import (
    build_review_package,
    classify_transition,
    join_audit,
    number_change_type,
)


class OriginalBaselineAnalysisTest(unittest.TestCase):
    def test_transition_classification(self):
        self.assertEqual(classify_transition("entailment", "entailment", "entailment"), "STABLE_GOLD")
        self.assertEqual(classify_transition("entailment", "neutral", "entailment"), "TRUE_FLIP")
        self.assertEqual(classify_transition("neutral", "neutral", "entailment"), "VERIFIER_GOLD_DISAGREEMENT_STABLE")
        self.assertEqual(classify_transition("neutral", "entailment", "entailment"), "RECOVERED_TO_GOLD")
        self.assertEqual(classify_transition("neutral", "contradiction", "entailment"), "NON_GOLD_TRANSITION")

    def test_join_adds_confidence_delta_and_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.csv"
            fields = ["source_index", "candidate_id", "augmented_field", "gold_label", "nli_predicted_label",
                      "nli_gold_probability", "accepted", "reasons", "hard_cue_changes", "soft_cue_changes"]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerow({"source_index": 0, "candidate_id": "0:premise:fra_Latn", "augmented_field": "premise",
                                 "gold_label": "entailment", "nli_predicted_label": "neutral",
                                 "nli_gold_probability": "0.70", "accepted": "False", "reasons": "[]",
                                 "hard_cue_changes": "[]", "soft_cue_changes": "[]"})
            baseline = [{"calibration_index": 0, "source_index": 123, "original_nli_predicted_label": "entailment",
                         "original_nli_gold_probability": 0.95, "original_entailment_probability": 0.95,
                         "original_neutral_probability": 0.03, "original_contradiction_probability": 0.02}]
            row = join_audit(path, baseline)[0]
            self.assertEqual(row["original_source_index"], 123)
            self.assertEqual(row["nli_transition"], "TRUE_FLIP")
            self.assertAlmostEqual(row["nli_gold_probability_delta"], -0.25)
            self.assertTrue(row["large_confidence_drop"])

    def test_number_change_types_and_master_deduplication(self):
        self.assertEqual(number_change_type('{"5":1}', '{"five":1}'), "LEXICAL_NORMALIZATION")
        self.assertEqual(number_change_type('{"one":1}', '{}'), "NUMBER_REMOVED")
        self.assertEqual(number_change_type('{}', '{"two":1}'), "NUMBER_ADDED")
        self.assertEqual(number_change_type('{"11":1}', '{"12":1}'), "POSSIBLE_VALUE_CHANGE")
        row = {
            "source_index": "0", "original_source_index": "99", "candidate_id": "0:premise:fra_Latn",
            "augmented_field": "premise", "gold_label": "entailment", "accepted": "False",
            "reasons": json.dumps(["label_flip", "hard_cue_changed"]), "nli_transition": "TRUE_FLIP",
            "_hard_groups": {"number"}, "_soft_groups": set(), "_number_before": '{"5":1}',
            "_number_after": '{"five":1}', "semantic_forward_entailment": "0.9",
            "semantic_backward_entailment": "0.9", "semantic_min": "0.9",
            "candidate_nli_gold_probability": "0.1", "original_nli_predicted_label": "entailment",
            "candidate_nli_predicted_label": "neutral", "original_nli_gold_probability": "0.9",
            "nli_gold_probability_delta": "-0.8", "hard_cue_changes": "[]", "soft_cue_changes": "[]",
            "change_ratio": "0.3", "length_ratio": "1.0",
        }
        with tempfile.TemporaryDirectory() as directory:
            counts = build_review_package([row], Path(directory), 42)
            self.assertEqual(counts["number_cue_audit.csv"], 1)
            master = list(csv.DictReader((Path(directory) / "MASTER_REVIEW.csv").open(encoding="utf-8")))
            self.assertEqual(len(master), 1)
            self.assertIn("NUMBER_CUE", master[0]["review_categories"])
            self.assertIn("TRUE_FLIP", master[0]["review_categories"])


if __name__ == "__main__":
    unittest.main()
