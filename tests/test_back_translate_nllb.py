import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.back_translate_nllb import (
    Candidate,
    candidate_fields,
    completed_candidate_ids,
    make_candidates,
    output_record,
    read_records,
)
from scripts.quality_filter import QualityFilter, validate_record


HIGH = {"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02}


class FakeVerifier:
    def __init__(self, semantic=None, pair=None):
        self.semantic = semantic or [HIGH, HIGH]
        self.pair = pair or HIGH
        self.calls = 0

    def predict_proba(self, pairs):
        self.calls += 1
        if self.calls == 1:
            return self.semantic[: len(pairs)]
        return [self.pair for _ in pairs]


def quality(verifier=None):
    return QualityFilter(verifier or FakeVerifier(), 0.80, 0.80, 0.03)


class QualityFilterTest(unittest.TestCase):
    def test_semantic_drift(self):
        low = {"entailment": 0.20, "neutral": 0.70, "contradiction": 0.10}
        decision = quality(FakeVerifier([low, HIGH])).evaluate(
            "A woman hugs a child", "Someone is present", "A woman kisses a child",
            "Someone is present", 1, "premise"
        )
        self.assertIn("semantic_drift", decision.reasons)
        self.assertFalse(decision.accepted)

    def test_label_flip(self):
        pair = {"entailment": 0.05, "neutral": 0.05, "contradiction": 0.90}
        decision = quality(FakeVerifier(pair=pair)).evaluate(
            "A man runs", "A man runs", "A man runs", "A man does not run", 0, "hypothesis"
        )
        self.assertIn("label_flip", decision.reasons)

    def test_low_nli_confidence(self):
        pair = {"entailment": 0.60, "neutral": 0.20, "contradiction": 0.20}
        decision = quality(FakeVerifier(pair=pair)).evaluate(
            "A man runs", "A man runs", "A man runs quickly", "A man runs", 0, "premise"
        )
        self.assertIn("low_nli_confidence", decision.reasons)

    def test_logical_cue_changed(self):
        decision = quality().evaluate(
            "A person is outside", "A person is visible", "A person is inside",
            "A person is visible", 1, "premise"
        )
        self.assertIn("logical_cue_changed", decision.reasons)

    def test_truncation(self):
        decision = quality().evaluate(
            "A man runs", "A man runs", "A man runs quickly", "A man runs", 0,
            "premise", was_truncated=True
        )
        self.assertIn("input_too_long", decision.reasons)

    def test_invalid_label_and_field(self):
        self.assertEqual(validate_record({"premise": "x", "hypothesis": "y", "label": -1}), ["invalid_label"])
        self.assertIn("invalid_field", validate_record({"premise": None, "hypothesis": "y", "label": 0}))


class PipelineHelpersTest(unittest.TestCase):
    def test_separate_mode_creates_two_candidates(self):
        record = {"premise": "old p", "hypothesis": "old h", "label": 0}
        candidates = make_candidates(12, record, {"premise": "new p", "hypothesis": "new h"}, {}, "separate")
        self.assertEqual(candidate_fields("separate"), ("premise", "hypothesis"))
        self.assertEqual([c.candidate_id for c in candidates], ["12:premise", "12:hypothesis"])
        self.assertEqual(candidates[0].hypothesis, "old h")
        self.assertEqual(candidates[1].premise, "old p")

    def test_metadata_contains_quality_and_provenance(self):
        args = Namespace(
            no_metadata=False, pivot_lang="fra_Latn", model_name="nllb", verifier_model="verifier",
            verifier_revision=None, quality_filter="on", num_beams=4, max_input_tokens=256,
            max_new_tokens=256, semantic_threshold=0.8, nli_threshold=0.8, min_change_ratio=0.03,
        )
        candidate = Candidate(3, {"premise": "p", "hypothesis": "h", "label": 0}, "new p", "h", "premise", False)
        record = output_record(candidate, quality().evaluate("p", "h", "new p", "h", 0, "premise"), args, None)
        self.assertIn("quality", record)
        self.assertIn("provenance", record)
        self.assertEqual(record["candidate_id"], "3:premise")

    def test_resume_counts_accepted_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            accepted = Path(directory) / "accepted.jsonl"
            rejected = Path(directory) / "rejected.jsonl"
            accepted.write_text(json.dumps({"candidate_id": "1:premise"}) + "\n", encoding="utf-8")
            rejected.write_text(json.dumps({"source_index": 1, "augmented_field": "hypothesis"}) + "\n", encoding="utf-8")
            self.assertEqual(completed_candidate_ids((accepted, rejected)), {"1:premise", "1:hypothesis"})

    def test_jsonl_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('{"premise":"p","hypothesis":"h","label":1}\n\n', encoding="utf-8")
            self.assertEqual(list(read_records(path))[0]["label"], 1)


if __name__ == "__main__":
    unittest.main()
