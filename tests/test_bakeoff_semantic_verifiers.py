import unittest

from scripts.bakeoff_semantic_verifiers import (
    anchor_pairwise,
    anchor_specs,
    bidirectional,
    score_pair_rows,
    secondary_disagreement_rows,
    stable_review_rows,
    threshold_rows,
)


class FakeScorer:
    def __init__(self, entailment=None, regression=None):
        self.entailment = entailment
        self.regression = regression

    def predict_entailment(self, pairs):
        return [self.entailment[i % len(self.entailment)] for i in range(len(pairs))]

    def predict_regression(self, pairs):
        return [self.regression[i % len(self.regression)] for i in range(len(pairs))]


class BakeoffHelpersTest(unittest.TestCase):
    def test_bidirectional_min_and_serialization(self):
        nli = FakeScorer(entailment=[0.91, 0.73, 0.61, 0.82])
        forward, backward = bidirectional(nli, [("original", "candidate"), ("a", "b")])
        self.assertEqual(forward, [0.91, 0.73])
        self.assertEqual(backward, [0.61, 0.82])

        rows = [{"original_sentence": "original", "back_translated_sentence": "candidate"}]
        sts = FakeScorer(regression=[0.88])
        runtimes = score_pair_rows(rows, nli, sts, nli)
        self.assertEqual(rows[0]["secondary_sts_score"], 0.88)
        self.assertEqual(rows[0]["secondary_roberta_min_entailment"], 0.73)
        self.assertIn("sts", runtimes)

    def test_anchor_loading_and_pairwise_ranking(self):
        source_rows = []
        for source_index, field, original, candidate in (
            ("148356", "hypothesis", "bad1", "bad1b"),
            ("227619", "hypothesis", "bad2", "bad2b"),
            ("512053", "hypothesis", "bad3", "bad3b"),
            ("355209", "hypothesis", "bad4", "bad4b"),
            ("161674", "premise", "bad7", "bad7b"),
            ("138332", "premise", "bad8", "bad8b"),
            ("90", "hypothesis", "pass1", "pass1b"),
        ):
            source_rows.append({"source_index": source_index, "augmented_field": field,
                                "original_sentence": original, "back_translated_sentence": candidate})
        anchors = anchor_specs(source_rows)
        self.assertEqual(len(anchors), 13)
        self.assertEqual(anchors[0]["expert_verdict"], "FAIL")
        anchors[0]["sts_score"] = 0.2
        anchors[-1]["sts_score"] = 0.9
        self.assertEqual(anchor_pairwise(anchors, "sts_score"), 1.0)

    def test_stable_source_index_join(self):
        audit = [{"source_index": "148356", "input_position": "225",
                  "candidate_id": "148356:hypothesis:fra_Latn",
                  "augmented_field": "hypothesis", "semantic_forward": "0.95",
                  "semantic_backward": "0.94", "semantic_min": "0.94"}]
        mapping = [{"source_index": "148356", "input_position": "225"}]
        review = [{"source_index": "225", "original_source_index": "148356",
                   "candidate_id": "225:hypothesis:fra_Latn", "augmented_field": "hypothesis",
                   "original_sentence": "o", "back_translated_sentence": "b"}]
        joined = stable_review_rows(review, audit, mapping)
        self.assertEqual(joined[0]["candidate_id"], "148356:hypothesis:fra_Latn")
        self.assertEqual(joined[0]["input_position"], "225")
        self.assertEqual(joined[0]["current_semantic_min"], "0.94")
        self.assertEqual(joined[0]["original_candidate_id"], "225:hypothesis:fra_Latn")

    def test_threshold_sweep_and_disagreement_selection(self):
        anchors = [{"expert_verdict": "FAIL", "secondary_sts_score": 0.5},
                   {"expert_verdict": "PASS", "secondary_sts_score": 0.95}]
        accepted = [{"secondary_sts_score": 0.75}, {"secondary_sts_score": 0.9}]
        sweep = threshold_rows(anchors, accepted, "secondary_sts_score", (0.8,))
        self.assertEqual(sweep[0]["retained"], 1)
        self.assertEqual(sweep[0]["newly_rejected"], 1)
        self.assertEqual(sweep[0]["fail_anchors_rejected"], 1)
        rows = [
            {"accepted": "True", "semantic_min": "0.97", "secondary_sts_score": "0.7",
             "secondary_roberta_min_entailment": "0.95"},
            {"accepted": "True", "semantic_min": "0.97", "secondary_sts_score": "0.9",
             "secondary_roberta_min_entailment": "0.7"},
            {"accepted": "False", "semantic_min": "0.99", "secondary_sts_score": "0.1",
             "secondary_roberta_min_entailment": "0.1"},
        ]
        selected = secondary_disagreement_rows(rows)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["secondary_sts_score"], "0.7")


if __name__ == "__main__":
    unittest.main()
