import json
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace

from scripts.back_translate_nllb import generation_was_truncated, indexed_chunks, make_candidates, run_pipeline
from scripts.policy_v2 import classify_source_nli
from scripts.quality_filter import QualityFilter, numeric_value_conflicts
from scripts.simulate_policy_v2 import simulate, v2_decision


class GateVerifier:
    def __init__(self, probability):
        self.probability = probability
        self.calls = []

    def predict_proba(self, pairs):
        self.calls.append(list(pairs))
        return [self.probability for _ in pairs]


class NoCallTranslator:
    def __init__(self):
        self.calls = 0

    def token_length(self, text):
        return len(text.split())

    def translate_with_metadata(self, texts, batch_size, allow_truncation):
        self.calls += 1
        return [text + " paraphrase" for text in texts], [{} for _ in texts]


class PolicyV2Test(unittest.TestCase):
    def test_source_gate_statuses(self):
        reliable = classify_source_nli({"entailment": .9, "neutral": .05, "contradiction": .05}, 0, .8)
        low = classify_source_nli({"entailment": .7, "neutral": .2, "contradiction": .1}, 0, .8)
        disagreement = classify_source_nli({"entailment": .1, "neutral": .8, "contradiction": .1}, 0, .8)
        self.assertEqual(reliable["source_nli_status"], "RELIABLE_GOLD")
        self.assertEqual(low["source_nli_status"], "GOLD_LOW_CONFIDENCE")
        self.assertEqual(disagreement["source_nli_status"], "GOLD_DISAGREEMENT")

    def test_number_normalization_and_explicit_conflicts(self):
        for left, right in (("5 people", "five people"), ("2 people", "two people"),
                            ("two people", "both people"), ("a couple of people", "two people")):
            self.assertFalse(numeric_value_conflicts(left, right), (left, right))
        for left, right in (("2 people", "3 people"), ("five people", "six people"),
                            ("11 people", "12 people"), ("3.5 people", "4.5 people")):
            self.assertTrue(numeric_value_conflicts(left, right), (left, right))

    def test_pronoun_one_is_not_numeric(self):
        for text in ("no one is here", "the one is red", "one another", "one of them", "another one"):
            self.assertEqual(__import__("scripts.quality_filter", fromlist=["explicit_number_values"]).explicit_number_values(text), [])
        self.assertFalse(numeric_value_conflicts("One man is running", "A man is running"))

    def test_negation_canonicalization_is_not_a_hard_reject(self):
        verifier = GateVerifier({"entailment": .95, "neutral": .03, "contradiction": .02})
        decision = QualityFilter(verifier).evaluate(
            "Nobody has food", "Someone is present", "No one has food", "Someone is present", 0, "premise")
        self.assertNotIn("numeric_value_changed", decision.reasons)
        self.assertNotIn("negation_changed", decision.reasons)

    def test_generation_eos_and_padding(self):
        self.assertFalse(generation_was_truncated([0, 5, 2, 1, 1], 2, 1, 4, 0))
        self.assertFalse(generation_was_truncated([0, 5, 2, 1, 1], 2, 1, 2, 0))
        self.assertTrue(generation_was_truncated([0, 5, 6, 7, 8], 2, 1, 4, 0))
        self.assertFalse(generation_was_truncated([0, 5, 2, 1, 1], 2, 1, 4, 0))

    def test_both_metadata_merges_generation_truncation(self):
        candidates = make_candidates(3, {"premise": "p", "hypothesis": "h", "label": 0},
            {"premise": "p2", "hypothesis": "h2"},
            {"premise": {"source_to_pivot_generation_truncated": True},
             "hypothesis": {"pivot_to_source_generation_truncated": True}}, "both", "fra_Latn")
        self.assertTrue(candidates[0].truncation["source_to_pivot_generation_truncated"])
        self.assertTrue(candidates[0].truncation["pivot_to_source_generation_truncated"])
        self.assertTrue(candidates[0].truncation["generation_truncated"])

    def test_source_gate_writes_placeholders_and_skips_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_path.write_text(json.dumps({"premise": "A man runs", "hypothesis": "Someone moves", "label": 0}) + "\n", encoding="utf-8")
            args = Namespace(input=input_path, accepted_path=root / "accepted.jsonl", rejected_path=root / "rejected.jsonl",
                resume=False, max_samples=None, chunk_size=8, batch_size=2, max_input_tokens=128, max_new_tokens=4,
                num_beams=2, pivot_lang="fra_Latn", augmentation_mode="separate", quality_filter="on", no_metadata=False,
                allow_truncation=False, model_name="fake", verifier_model="fake", verifier_revision=None,
                semantic_threshold=.9, nli_threshold=.8, min_change_ratio=.03, source_nli_threshold=.8, source_nli_gate="on")
            translator = NoCallTranslator()
            stats = run_pipeline(args, translator, None, GateVerifier({"entailment": .1, "neutral": .8, "contradiction": .1}))
            self.assertEqual(stats["rejected"], 2)
            self.assertEqual(translator.calls, 0)
            rows = [json.loads(line) for line in (root / "rejected.jsonl").read_text().splitlines()]
            self.assertEqual({row["candidate_id"] for row in rows}, {"0:premise:fra_Latn", "0:hypothesis:fra_Latn"})
            self.assertTrue(all(row["translation_performed"] is False for row in rows))
            self.assertTrue(all(row["source_nli_status"] == "GOLD_DISAGREEMENT" for row in rows))

    def test_simulation_source_gate_and_semantic_v2(self):
        base = {"source_index": "0", "candidate_id": "0:premise:fra_Latn", "gold_label": "entailment",
                "augmented_field": "premise", "accepted": "True", "reasons": "[]",
                "original_nli_predicted_label": "entailment", "original_nli_gold_probability": "0.9",
                "candidate_nli_predicted_label": "entailment", "candidate_nli_gold_probability": "0.9",
                "semantic_min": "0.85", "original_sentence": "2 people", "back_translated_sentence": "two people"}
        accepted, reasons, _ = v2_decision(base)
        self.assertFalse(accepted)
        self.assertIn("semantic_drift", reasons)
        base["original_nli_predicted_label"] = "neutral"
        accepted, reasons, _ = v2_decision(base)
        self.assertFalse(accepted)
        self.assertEqual(reasons, ["source_label_disagreement"])
        result = simulate([base])
        self.assertEqual(result[0]["source_nli_status"], "GOLD_DISAGREEMENT")


    def test_nllb_real_decoder_start_eos_ids(self):
        normal = [2, 256047, 123, 456, 2, 1, 1]
        cutoff = [2, 256047, 123, 456, 789]
        self.assertFalse(generation_was_truncated(normal, 2, 1, 4, 2))
        self.assertTrue(generation_was_truncated(cutoff, 2, 1, 4, 2))
        self.assertTrue(generation_was_truncated([2, 256047, 123, 456, 789], 2, 1, 4, 2))

    def test_stable_source_index_and_input_position(self):
        rows = list(indexed_chunks([{"source_index": 148356, "premise": "p", "hypothesis": "h", "label": 0}, {"premise": "p2", "hypothesis": "h2", "label": 1}], 8))[0]
        self.assertEqual(rows[0][0], 148356)
        self.assertEqual(rows[0][1]["_input_position"], 0)
        self.assertEqual(rows[1][0], 1)
        self.assertEqual(rows[1][1]["_input_position"], 1)
        self.assertIn("invalid_source_index", list(indexed_chunks([{"source_index": "148356", "premise": "p", "hypothesis": "h", "label": 0}], 8))[0][0][1]["_source_index_reasons"])
        with self.assertRaises(ValueError):
            list(indexed_chunks([{"source_index": 5}, {"source_index": 5}], 8))

    def test_source_gate_placeholder_all_modes(self):
        for mode, expected_fields in (("separate", {"premise", "hypothesis"}), ("premise", {"premise"}), ("hypothesis", {"hypothesis"}), ("both", {"both"})):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                input_path = root / "input.jsonl"
                input_path.write_text(json.dumps({"source_index": 148356, "premise": "A man runs", "hypothesis": "Someone moves", "label": 0}) + "\n", encoding="utf-8")
                args = Namespace(input=input_path, accepted_path=root / "accepted.jsonl", rejected_path=root / "rejected.jsonl", resume=False, max_samples=None, chunk_size=8, batch_size=2, max_input_tokens=128, max_new_tokens=4, num_beams=2, pivot_lang="fra_Latn", augmentation_mode=mode, quality_filter="on", no_metadata=False, allow_truncation=False, model_name="fake", verifier_model="fake", verifier_revision=None, semantic_threshold=.9, nli_threshold=.8, min_change_ratio=.03, source_nli_threshold=.8, source_nli_gate="on")
                translator = NoCallTranslator()
                stats = run_pipeline(args, translator, None, GateVerifier({"entailment": .1, "neutral": .8, "contradiction": .1}))
                rows = [json.loads(line) for line in (root / "rejected.jsonl").read_text().splitlines()]
                self.assertEqual({row["augmented_field"] for row in rows}, expected_fields)
                self.assertEqual({row["candidate_id"] for row in rows}, {f"148356:{field}:fra_Latn" for field in expected_fields})
                self.assertEqual(translator.calls, 0)
                self.assertTrue(all(row["input_position"] == 0 for row in rows))
                self.assertTrue(all(row["translation_performed"] is False for row in rows))


if __name__ == "__main__":
    unittest.main()
