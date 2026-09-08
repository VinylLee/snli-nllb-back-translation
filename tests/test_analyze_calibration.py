import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_calibration import CSV_COLUMNS, analyze


class AnalyzeCalibrationTest(unittest.TestCase):
    def test_analyze_writes_full_and_review_artifacts(self):
        provenance = {
            "translation_model": "fake-nllb", "translation_model_revision": "translation-rev",
            "verifier_model": "fake-verifier", "verifier_model_revision": "verifier-rev",
            "generation": {"num_beams": 2, "max_input_tokens": 128, "max_new_tokens": 64},
        }
        accepted = {
            "source_index": 4, "candidate_id": "4:premise:fra_Latn", "label": 0, "gold_label": 0,
            "original_premise": "A man runs", "original_hypothesis": "Someone moves",
            "premise": "A man jogs", "hypothesis": "Someone moves", "augmented_field": "premise",
            "pivot_lang": "fra_Latn", "was_truncated": False, "truncation": {},
            "quality": {"accepted": True, "reasons": [], "semantic_premise_forward_entailment": 0.96,
                        "semantic_premise_backward_entailment": 0.95, "nli_gold_probability": 0.97,
                        "nli_predicted_label": "entailment", "hard_cue_changes": [], "soft_cue_changes": [],
                        "premise_change_ratio": 0.35, "premise_length_ratio": 1.0},
            "provenance": provenance,
        }
        rejected = {
            **accepted, "source_index": 5, "candidate_id": "5:hypothesis:fra_Latn",
            "augmented_field": "hypothesis", "hypothesis": "Nobody moves",
            "quality": {"accepted": False, "reasons": ["semantic_drift", "label_flip"],
                        "semantic_hypothesis_forward_entailment": 0.2,
                        "semantic_hypothesis_backward_entailment": 0.3, "nli_gold_probability": 0.1,
                        "nli_predicted_label": "contradiction", "hard_cue_changes": [], "soft_cue_changes": [],
                        "hypothesis_change_ratio": 0.8, "hypothesis_length_ratio": 1.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted_path = root / "accepted.jsonl"
            rejected_path = root / "rejected.jsonl"
            source_path = root / "sources.jsonl"
            accepted_path.write_text(json.dumps(accepted) + "\n", encoding="utf-8")
            rejected_path.write_text(json.dumps(rejected) + "\n", encoding="utf-8")
            source_path.write_text(json.dumps({"source_index": 4, "premise": "A man runs", "hypothesis": "Someone moves", "label": 0}) + "\n", encoding="utf-8")
            result = analyze(accepted_path, rejected_path, source_path, root, "source-rev", "pipeline-rev")
            self.assertEqual(result["candidate_count"], 2)
            self.assertEqual(result["accepted_count"], 1)
            self.assertTrue((root / "snli_calibration_800.audit.csv").exists())
            self.assertTrue((root / "CALIBRATION_REPORT.md").exists())
            with (root / "snli_calibration_800.audit.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(set(CSV_COLUMNS).issubset(rows[0]))
            self.assertEqual(rows[0]["human_semantic_equivalent"], "")
            self.assertTrue((root / "accepted_high_confidence.csv").exists())
            self.assertTrue((root / "rejected_semantic_drift.csv").exists())


if __name__ == "__main__":
    unittest.main()
