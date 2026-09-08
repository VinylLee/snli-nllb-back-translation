import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.analyze_precision_audit import (
    bootstrap_ci,
    error_counts,
    merge_review_rows,
    optimistic_quality,
    strict_quality,
    weighted_rate,
)
from scripts.build_precision_audit import (
    BLIND_COLUMNS,
    construct_rows,
    build_package,
)


BASE = Path("data/nli/calibration")
AUDIT = BASE / "snli_calibration_800_v2.audit.csv"
SOURCES = BASE / "snli_calibration_800_sources.jsonl"
SCORED = BASE / "snli_calibration_800_v2.final_semantic_scores.csv"


class PrecisionAuditBuildTest(unittest.TestCase):
    def test_deterministic_stratified_sampling_and_blind_schema(self):
        first, first_key, sizes = construct_rows(AUDIT, SOURCES, scored_path=SCORED)
        second, second_key, _ = construct_rows(AUDIT, SOURCES, scored_path=SCORED)
        self.assertEqual([r["candidate_id"] for r in first_key], [r["candidate_id"] for r in second_key])
        counts = Counter((r["gold_label_name"], r["augmented_field"]) for r in first)
        self.assertEqual(set(counts.values()), {40})
        self.assertEqual(len(first), 240)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["audit_id"], "A0001")
        self.assertEqual(len({r["audit_id"] for r in first}), 240)
        self.assertEqual(set(BLIND_COLUMNS), set(first[0]))
        forbidden = {"source_index", "input_position", "candidate_id", "semantic_min",
                     "candidate_nli_gold_probability", "reasons", "hard_cue_changes",
                     "soft_cue_changes", "secondary_sts_score", "secondary_sts_large_score",
                     "secondary_bleurt_score", "secondary_roberta_min_entailment"}
        self.assertFalse(forbidden.intersection(first[0]))
        for row in first:
            self.assertTrue(all(row[field] == "" for field in (
                "review_semantic_equivalent", "review_label_preserved",
                "review_useful_augmentation", "review_severity",
                "review_error_types", "review_notes")))
        self.assertEqual(sizes[("entailment", "premise")], 133)

    def test_batches_are_contiguous_slices_and_key_ids_match(self):
        with tempfile.TemporaryDirectory() as directory:
            result = build_package(AUDIT, SOURCES, Path(directory), scored_path=SCORED)
            master_ids = [r["audit_id"] for r in result["blind"]]
            key_ids = [r["audit_id"] for r in result["key"]]
            self.assertEqual(master_ids, key_ids)
            for index in range(1, 7):
                path = Path(directory) / f"accepted_precision_audit_blind_{index:02d}.csv"
                with path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 40)
                self.assertEqual([r["audit_id"] for r in rows], master_ids[(index - 1) * 40:index * 40])


class PrecisionAuditAnalysisTest(unittest.TestCase):
    def _rows(self):
        return [
            {"audit_id": "A0001", "gold_label_name": "entailment", "augmented_field": "premise",
             "stratum_population_size": "100", "review_semantic_equivalent": "PASS",
             "review_label_preserved": "PASS", "review_useful_augmentation": "PASS",
             "review_severity": "NONE", "review_error_types": ""},
            {"audit_id": "A0002", "gold_label_name": "neutral", "augmented_field": "hypothesis",
             "stratum_population_size": "300", "review_semantic_equivalent": "FAIL",
             "review_label_preserved": "FAIL", "review_useful_augmentation": "FAIL",
             "review_severity": "MAJOR", "review_error_types": "TENSE_ASPECT|EVENT_PREDICATE"},
        ]

    def test_strict_optimistic_and_population_weighting(self):
        rows = self._rows()
        self.assertTrue(strict_quality(rows[0]))
        self.assertTrue(optimistic_quality(rows[0]))
        self.assertFalse(strict_quality(rows[1]))
        self.assertEqual(weighted_rate(rows, strict_quality), 0.25)
        self.assertEqual(weighted_rate(rows, optimistic_quality), 0.25)
        self.assertEqual(error_counts(rows)["TENSE_ASPECT"], 1)
        self.assertEqual(error_counts(rows)["EVENT_PREDICATE"], 1)

    def test_bootstrap_is_reproducible(self):
        rows = self._rows()
        first = bootstrap_ci(rows, strict_quality, iterations=200, seed=20260908)
        second = bootstrap_ci(rows, strict_quality, iterations=200, seed=20260908)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_future_review_merge_requires_matching_audit_ids(self):
        review = [{"audit_id": "A0001", "gold_label_name": "entailment", "augmented_field": "premise",
                   "review_semantic_equivalent": "PASS"}]
        key = [{"audit_id": "A0001", "candidate_id": "1:premise:fra_Latn",
                "stratum_population_size": "100"}]
        merged = merge_review_rows(review, key)
        self.assertEqual(merged[0]["key_candidate_id"], "1:premise:fra_Latn")
        with self.assertRaises(ValueError):
            merge_review_rows(review, key + [{"audit_id": "A0002"}])


if __name__ == "__main__":
    unittest.main()
