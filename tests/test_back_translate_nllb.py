import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.back_translate_nllb import (
    TRUNCATION_KEYS,
    completed_candidate_ids,
    make_candidates,
    parse_args,
    run_pipeline,
)
from scripts.quality_filter import QualityFilter, corruption_reasons, cue_snapshot, logical_cue_changes, validate_record


HIGH = {"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02}
FLIP = {"entailment": 0.05, "neutral": 0.05, "contradiction": 0.90}


class FakeVerifier:
    def __init__(self, pair_outputs=None, semantic=None):
        self.pair_outputs = pair_outputs or [HIGH]
        self.semantic = semantic or HIGH
        self.batch_sizes = []
        self.calls = 0

    def predict_proba(self, pairs):
        self.batch_sizes.append(len(pairs))
        self.calls += 1
        if self.calls == 1:
            return [self.semantic for _ in pairs]
        return [self.pair_outputs[min(i, len(self.pair_outputs) - 1)] for i in range(len(pairs))]


class FakeTranslator:
    model_revision = "fake-translation-revision"

    def __init__(self, truncated=False, intermediate=False):
        self.truncated = truncated
        self.intermediate = intermediate
        self.calls = []

    def token_length(self, text):
        return 999 if text == "TOO_LONG" else len(text.split())

    def translate_with_metadata(self, texts, batch_size, allow_truncation):
        self.calls.append((list(texts), batch_size, allow_truncation))
        return (
            [f"{text} paraphrase" for text in texts],
            [{
                "source_to_pivot_over_limit": self.truncated,
                "source_to_pivot_truncated": self.truncated and allow_truncation,
                "pivot_to_source_over_limit": self.intermediate,
                "pivot_to_source_truncated": self.intermediate and allow_truncation,
                "was_truncated": (self.truncated or self.intermediate) and allow_truncation,
                "intermediate_input_too_long": self.intermediate and not allow_truncation,
            } for _ in texts],
        )


def quality_args(directory, mode="separate", resume=False, no_metadata=False):
    accepted = Path(directory) / "accepted.jsonl"
    rejected = Path(directory) / "rejected.jsonl"
    return Namespace(
        input=Path(directory) / "input.jsonl", accepted_path=accepted, rejected_path=rejected,
        accepted_output=accepted, rejected_output=rejected, output=None, resume=resume,
        overwrite=not resume, max_samples=None, chunk_size=8, batch_size=4,
        max_input_tokens=256, max_new_tokens=64, num_beams=2, pivot_lang="fra_Latn",
        model_name="fake-nllb", verifier_model="fake-verifier", verifier_revision=None,
        quality_filter="on", semantic_threshold=0.8, nli_threshold=0.8,
        min_change_ratio=0.03, augmentation_mode=mode, no_metadata=no_metadata, allow_truncation=False,
    )


class QualityFilterTest(unittest.TestCase):
    def test_contraction_snapshot_terminates_and_detects_negation(self):
        for contraction in ("isn't", "doesn't", "wasn't", "can't", "weren't"):
            snapshot = cue_snapshot(f"The man {contraction} running.")
            self.assertEqual(snapshot["negation"].get("n't"), 1)

    def test_semantic_drift(self):
        low = {"entailment": 0.20, "neutral": 0.70, "contradiction": 0.10}
        decision = QualityFilter(FakeVerifier(semantic=low)).evaluate(
            "A woman hugs a child", "Someone is present", "A woman kisses a child",
            "Someone is present", 1, "premise")
        self.assertIn("semantic_drift", decision.reasons)

    def test_corrupt_food_switch_is_rejected(self):
        self.assertEqual(corruption_reasons("A boy is turning on a hamburger."), ["corrupt_output"])

    def test_batch_evaluation_uses_two_batch_calls(self):
        verifier = FakeVerifier()
        quality = QualityFilter(verifier)
        items = [{
            "original_premise": "A man runs", "original_hypothesis": "Someone moves",
            "candidate_premise": "A man jogs", "candidate_hypothesis": "Someone moves",
            "gold_label": 0, "augmented_field": "premise", "was_truncated": False,
        } for _ in range(3)]
        decisions = quality.evaluate_batch(items)
        self.assertEqual(len(decisions), 3)
        self.assertEqual(verifier.batch_sizes, [6, 3])

    def test_label_flip_and_low_confidence(self):
        decision = QualityFilter(FakeVerifier(pair_outputs=[FLIP])).evaluate(
            "A man runs", "A man runs", "A man runs quickly", "A man does not run", 0, "hypothesis")
        self.assertIn("label_flip", decision.reasons)
        low = {"entailment": 0.60, "neutral": 0.20, "contradiction": 0.20}
        decision = QualityFilter(FakeVerifier(pair_outputs=[low])).evaluate(
            "A man runs", "A man runs", "A man runs quickly", "A man runs", 0, "premise")
        self.assertIn("low_nli_confidence", decision.reasons)

    def test_hard_and_soft_cue_behavior(self):
        hard = QualityFilter(FakeVerifier()).evaluate(
            "A person is running", "Someone is visible", "A person is not running",
            "Someone is visible", 1, "premise")
        self.assertIn("hard_cue_changed", hard.reasons)
        soft = QualityFilter(FakeVerifier()).evaluate(
            "A person is outside", "Someone is visible", "A person is inside",
            "Someone is visible", 0, "premise")
        self.assertNotIn("hard_cue_changed", soft.reasons)
        self.assertIn("soft_cue_changed", soft.flags)
        self.assertTrue(soft.accepted)

    def test_number_cue_supports_arbitrary_literals_and_decimals(self):
        for original, candidate in (("11 people", "12 people"), ("20 people", "30 people"),
                                    ("100 people", "200 people"), ("3.5 people", "4.5 people")):
            changes = logical_cue_changes(original, candidate)
            self.assertIn("number", {change["group"] for change in changes})
            decision = QualityFilter(FakeVerifier()).evaluate(
                original, "Someone is present", candidate, "Someone is present", 0, "premise")
            self.assertIn("hard_cue_changed", decision.reasons)
        unchanged = logical_cue_changes("11 people", "There are 11 people")
        self.assertNotIn("number", {change["group"] for change in unchanged})

    def test_truncation_is_flag_not_automatic_rejection(self):
        truncation = {"source_to_pivot_over_limit": False, "source_to_pivot_truncated": False,
                      "pivot_to_source_over_limit": True, "pivot_to_source_truncated": True,
                      "was_truncated": True}
        decision = QualityFilter(FakeVerifier()).evaluate(
            "A man runs", "Someone moves", "A man jogs", "Someone moves", 0,
            "premise", was_truncated=True, truncation=truncation)
        self.assertTrue(decision.accepted)
        self.assertIn("was_truncated", decision.flags)
        self.assertTrue(decision.scores["truncation"]["pivot_to_source_truncated"])

    def test_invalid_label_and_field(self):
        self.assertIn("invalid_label", validate_record({"premise": "x", "hypothesis": "y", "label": -1}))
        self.assertIn("invalid_field", validate_record({"premise": None, "hypothesis": "y", "label": 0}))


class PipelineTest(unittest.TestCase):
    def test_separate_pivot_ids_and_field_specific_truncation(self):
        record = {"premise": "old p", "hypothesis": "old h", "label": 0}
        fra = make_candidates(12, record, {"premise": "new p", "hypothesis": "new h"}, {
            "premise": {"source_to_pivot_truncated": True, "pivot_to_source_truncated": False, "was_truncated": True},
            "hypothesis": {"source_to_pivot_truncated": False, "pivot_to_source_truncated": True, "was_truncated": True},
        }, "separate", "fra_Latn")
        deu = make_candidates(12, record, {"premise": "new p", "hypothesis": "new h"}, {}, "separate", "deu_Latn")
        self.assertEqual([candidate.candidate_id for candidate in fra], ["12:premise:fra_Latn", "12:hypothesis:fra_Latn"])
        self.assertNotEqual(fra[0].candidate_id, deu[0].candidate_id)
        self.assertTrue(fra[0].truncation["source_to_pivot_truncated"])
        self.assertFalse(fra[0].truncation["pivot_to_source_truncated"])
        self.assertFalse(fra[1].truncation["source_to_pivot_truncated"])
        self.assertTrue(fra[1].truncation["pivot_to_source_truncated"])

    def test_both_mode_candidate_and_truncation_merge(self):
        record = {"premise": "old p", "hypothesis": "old h", "label": 0}
        candidates = make_candidates(12, record, {"premise": "new p", "hypothesis": "new h"}, {
            "premise": {"source_to_pivot_over_limit": True, "source_to_pivot_truncated": False},
            "hypothesis": {"pivot_to_source_over_limit": True, "pivot_to_source_truncated": True},
        }, "both", "fra_Latn")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].candidate_id, "12:both:fra_Latn")
        info = candidates[0].truncation
        self.assertTrue(info["source_to_pivot_over_limit"])
        self.assertFalse(info["source_to_pivot_truncated"])
        self.assertTrue(info["pivot_to_source_over_limit"])
        self.assertTrue(info["pivot_to_source_truncated"])
        self.assertTrue(info["was_truncated"])

    def test_end_to_end_accept_reject_batch_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            args = quality_args(directory)
            Path(args.input).write_text(json.dumps({"premise": "A man runs", "hypothesis": "Someone moves", "label": 0}) + "\n", encoding="utf-8")
            verifier = FakeVerifier(pair_outputs=[HIGH, FLIP])
            stats = run_pipeline(args, FakeTranslator(), QualityFilter(verifier))
            self.assertEqual((stats["accepted"], stats["rejected"]), (1, 1))
            self.assertEqual(verifier.batch_sizes, [4, 2])
            self.assertEqual(len(args.accepted_path.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(len(args.rejected_path.read_text(encoding="utf-8").splitlines()), 1)
            args.resume = True
            args.overwrite = False
            second = run_pipeline(args, FakeTranslator(), QualityFilter(FakeVerifier()))
            self.assertEqual((second["accepted"], second["rejected"]), (0, 0))
            self.assertEqual(len(args.accepted_path.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(len(args.rejected_path.read_text(encoding="utf-8").splitlines()), 1)

    def test_both_mode_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            args = quality_args(directory, mode="both")
            Path(args.input).write_text("\n".join(json.dumps({"premise": f"A man runs {i}", "hypothesis": "Someone moves", "label": 0}) for i in range(2)) + "\n", encoding="utf-8")
            stats = run_pipeline(args, FakeTranslator(), QualityFilter(FakeVerifier(pair_outputs=[HIGH, FLIP])))
            self.assertEqual((stats["accepted"], stats["rejected"]), (1, 1))
            records = [json.loads(line) for line in args.accepted_path.read_text(encoding="utf-8").splitlines()]
            records += [json.loads(line) for line in args.rejected_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({record["candidate_id"] for record in records}, {"0:both:fra_Latn", "1:both:fra_Latn"})

    def test_truncation_disabled_rejects_source_and_records_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            args = quality_args(directory, mode="premise")
            Path(args.input).write_text(json.dumps({"premise": "TOO_LONG", "hypothesis": "Someone moves", "label": 0}) + "\n", encoding="utf-8")
            stats = run_pipeline(args, FakeTranslator(), None)
            self.assertEqual(stats["rejected"], 1)
            self.assertEqual(stats["reasons"], {"input_too_long": 1})
            record = json.loads(args.rejected_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(record["truncation"]["source_to_pivot_over_limit"])
            self.assertFalse(record["truncation"]["source_to_pivot_truncated"])

    def test_truncation_enabled_records_actual_source_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = quality_args(directory, mode="premise")
            args.allow_truncation = True
            Path(args.input).write_text(json.dumps({"premise": "TOO_LONG", "hypothesis": "Someone moves", "label": 0}) + "\n", encoding="utf-8")
            translator = FakeTranslator(truncated=True)
            stats = run_pipeline(args, translator, QualityFilter(FakeVerifier()))
            self.assertEqual(stats["accepted"], 1)
            self.assertEqual(translator.calls[0][2], True)
            record = json.loads(args.accepted_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(record["truncation"]["source_to_pivot_over_limit"])
            self.assertTrue(record["truncation"]["source_to_pivot_truncated"])
            self.assertFalse(record["truncation"]["pivot_to_source_over_limit"])

    def test_intermediate_overflow_rejects_without_truncation_and_allows_with_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            args = quality_args(directory, mode="premise")
            Path(args.input).write_text(json.dumps({"premise": "A man runs", "hypothesis": "Someone moves", "label": 0}) + "\n", encoding="utf-8")
            stats = run_pipeline(args, FakeTranslator(intermediate=True), None)
            self.assertEqual(stats["reasons"], {"intermediate_input_too_long": 1})
            rejected = json.loads(args.rejected_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(rejected["truncation"]["pivot_to_source_over_limit"])
            self.assertFalse(rejected["truncation"]["pivot_to_source_truncated"])
            args.allow_truncation = True
            args.overwrite = True
            stats = run_pipeline(args, FakeTranslator(intermediate=True), QualityFilter(FakeVerifier()))
            self.assertEqual(stats["accepted"], 1)
            accepted = json.loads(args.accepted_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(accepted["truncation"]["pivot_to_source_over_limit"])
            self.assertTrue(accepted["truncation"]["pivot_to_source_truncated"])

    def test_no_metadata_resume_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--input", "in.jsonl", "--output", "out.jsonl", "--no-metadata", "--resume"])

    def test_completed_ids_include_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            accepted, rejected = Path(directory) / "a.jsonl", Path(directory) / "r.jsonl"
            accepted.write_text(json.dumps({"candidate_id": "1:premise:fra_Latn"}) + "\n", encoding="utf-8")
            rejected.write_text(json.dumps({"source_index": 1, "augmented_field": "hypothesis", "pivot_lang": "fra_Latn"}) + "\n", encoding="utf-8")
            self.assertEqual(completed_candidate_ids((accepted, rejected)), {"1:premise:fra_Latn", "1:hypothesis:fra_Latn"})


if __name__ == "__main__":
    unittest.main()
