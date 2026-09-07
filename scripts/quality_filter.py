"""Explainable, batchable quality checks for NLI back-translation candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Protocol, Sequence


LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_REPEATED_CHUNK_RE = re.compile(r"\b(\w+(?:\s+\w+){1,5})\s+\1\b", re.IGNORECASE)
_IMPOSSIBLE_SWITCH_RE = re.compile(r"\b(turn|turning|switch|switching)\s+(?:on|off)\s+(?:a|an|the)\s+(?:hamburger|burger|food|omelette|pizza|sandwich)\b", re.IGNORECASE)

# Negation and explicit numbers are high-risk hard cues. Other cues are
# retained as auditable flags because valid paraphrases can change them.
HARD_CUE_TYPES = {"negation", "number"}
SOFT_CUE_TYPES = {"quantifier", "modal", "time", "space"}
CUE_GROUPS: dict[str, set[str]] = {
    "negation": {"not", "no", "never", "nobody", "nothing", "without", "n't"},
    "number": {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"},
    "quantifier": {"all", "some", "any", "every", "few", "several", "many", "most", "couple"},
    "modal": {"may", "might", "can", "could", "must", "should", "will"},
    "time": {"before", "after", "during", "first", "later", "already"},
    "space": {"in", "inside", "outside", "on", "under", "over", "behind", "front", "through", "left", "right", "near", "beside"},
}


class Verifier(Protocol):
    def predict_proba(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]: ...


@dataclass
class Decision:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def cue_snapshot(text: str) -> dict[str, dict[str, int]]:
    # Keep the source list immutable while deriving contraction cues. This is
    # intentionally a snapshot: extending a list while iterating it can loop
    # forever for strings such as "isn't".
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    contractions = ["n't" for token in tokens if token.endswith("n't")]
    tokens.extend(contractions)
    return {
        group: {cue: tokens.count(cue) for cue in cues if tokens.count(cue)}
        for group, cues in CUE_GROUPS.items()
    }


def logical_cue_changes(original: str, candidate: str) -> list[dict[str, Any]]:
    before, after = cue_snapshot(original), cue_snapshot(candidate)
    return [{"group": group, "original": before[group], "candidate": after[group]}
            for group in CUE_GROUPS if before[group] != after[group]]


def normalized_change_ratio(original: str, candidate: str) -> float:
    left, right = " ".join(original.lower().split()), " ".join(candidate.lower().split())
    return 1.0 - SequenceMatcher(None, left, right).ratio()


def corruption_reasons(text: str) -> list[str]:
    value = text.strip()
    if not value or _PUNCT_ONLY_RE.fullmatch(value) or _REPEATED_CHUNK_RE.search(value) or _IMPOSSIBLE_SWITCH_RE.search(value):
        return ["corrupt_output"]
    alphanumeric = sum(character.isalnum() or character.isspace() for character in value)
    return ["corrupt_output"] if alphanumeric / max(len(value), 1) < 0.35 else []


def canonical_label(label: Any) -> int | None:
    try:
        value = int(label)
    except (TypeError, ValueError):
        return None
    return value if value in LABEL_NAMES else None


class QualityFilter:
    def __init__(self, verifier: Verifier, semantic_threshold: float = 0.80,
                 nli_threshold: float = 0.80, min_change_ratio: float = 0.03,
                 hard_cue_types: set[str] | None = None) -> None:
        self.verifier = verifier
        self.semantic_threshold = semantic_threshold
        self.nli_threshold = nli_threshold
        self.min_change_ratio = min_change_ratio
        self.hard_cue_types = hard_cue_types or HARD_CUE_TYPES

    def evaluate_batch(self, items: Sequence[dict[str, Any]]) -> list[Decision]:
        """Evaluate all candidates with one semantic and one pair-NLI call.

        The verifier itself chunks these pair lists according to its configured
        batch size. Basic checks are assembled before inference and scores are
        mapped back to their original candidate positions.
        """
        if not items:
            return []
        decisions = [Decision(True, [], {"logical_cue_changes": [], "hard_cue_changes": [], "soft_cue_changes": []}, []) for _ in items]
        semantic_pairs: list[tuple[str, str]] = []
        semantic_map: list[tuple[int, str]] = []
        nli_pairs: list[tuple[str, str]] = []
        for index, item in enumerate(items):
            fields = ("premise", "hypothesis") if item["augmented_field"] == "both" else (item["augmented_field"],)
            texts = {"premise": (item["original_premise"], item["candidate_premise"]),
                     "hypothesis": (item["original_hypothesis"], item["candidate_hypothesis"])}
            scores = decisions[index].scores
            if item.get("was_truncated"):
                decisions[index].flags.append("was_truncated")
            if item.get("truncation"):
                scores["truncation"] = item["truncation"]
            for field_name in fields:
                original_text, candidate_text = texts[field_name]
                decisions[index].reasons.extend(corruption_reasons(candidate_text))
                change_ratio = normalized_change_ratio(original_text, candidate_text)
                scores[f"{field_name}_change_ratio"] = change_ratio
                scores[f"{field_name}_length_ratio"] = len(candidate_text) / max(len(original_text), 1)
                if change_ratio < self.min_change_ratio:
                    decisions[index].reasons.append("trivial_copy")
                if scores[f"{field_name}_length_ratio"] < 0.5 or scores[f"{field_name}_length_ratio"] > 2.0:
                    decisions[index].flags.append("length_ratio_risk")
                changes = logical_cue_changes(original_text, candidate_text)
                for change in changes:
                    change = {**change, "field": field_name}
                    scores["logical_cue_changes"].append(change)
                    if change["group"] in self.hard_cue_types:
                        scores["hard_cue_changes"].append(change)
                    elif change["group"] in SOFT_CUE_TYPES:
                        scores["soft_cue_changes"].append(change)
                if changes:
                    if any(change["group"] in self.hard_cue_types for change in changes):
                        decisions[index].reasons.append("hard_cue_changed")
                    elif any(change["group"] in SOFT_CUE_TYPES for change in changes):
                        decisions[index].flags.append("soft_cue_changed")
                semantic_pairs.extend(((original_text, candidate_text), (candidate_text, original_text)))
                semantic_map.extend(((index, f"{field_name}_forward"), (index, f"{field_name}_backward")))
            nli_pairs.append((item["candidate_premise"], item["candidate_hypothesis"]))

        semantic_probs = self.verifier.predict_proba(semantic_pairs)
        for probability, (item_index, direction) in zip(semantic_probs, semantic_map):
            score = float(probability.get("entailment", 0.0))
            decisions[item_index].scores[f"semantic_{direction}_entailment"] = score
            if score < self.semantic_threshold:
                decisions[item_index].reasons.append("semantic_drift")
        nli_probs = self.verifier.predict_proba(nli_pairs)
        for item_index, probability in enumerate(nli_probs):
            gold_label = int(items[item_index]["gold_label"])
            predicted = max(LABEL_NAMES, key=lambda label: float(probability.get(LABEL_NAMES[label], 0.0)))
            gold_probability = float(probability.get(LABEL_NAMES[gold_label], 0.0))
            decisions[item_index].scores["nli_predicted_label"] = predicted
            decisions[item_index].scores["nli_gold_probability"] = gold_probability
            if predicted != gold_label:
                decisions[item_index].reasons.append("label_flip")
            elif gold_probability < self.nli_threshold:
                decisions[item_index].reasons.append("low_nli_confidence")
        for decision in decisions:
            decision.reasons = list(dict.fromkeys(decision.reasons))
            decision.flags = list(dict.fromkeys(decision.flags))
            decision.accepted = not decision.reasons
        return decisions

    def evaluate(self, original_premise: str, original_hypothesis: str,
                 candidate_premise: str, candidate_hypothesis: str, gold_label: int,
                 augmented_field: str, was_truncated: bool = False,
                 truncation: dict[str, bool] | None = None) -> Decision:
        """Compatibility wrapper around the batch implementation."""
        item = {"original_premise": original_premise, "original_hypothesis": original_hypothesis,
                "candidate_premise": candidate_premise, "candidate_hypothesis": candidate_hypothesis,
                "gold_label": gold_label, "augmented_field": augmented_field,
                "was_truncated": was_truncated, "truncation": truncation or {}}
        return self.evaluate_batch([item])[0]


def validate_record(record: dict[str, Any]) -> list[str]:
    reasons = [] if canonical_label(record.get("label")) is not None else ["invalid_label"]
    for field_name in ("premise", "hypothesis"):
        if not isinstance(record.get(field_name), str) or not record[field_name].strip():
            reasons.append("invalid_field")
    return reasons


class TransformersNLI:
    """Transformer verifier with label order resolved from model config."""

    def __init__(self, model_name: str, device: str, dtype: str, batch_size: int,
                 max_length: int = 512) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except (ImportError, OSError) as exc:
            raise RuntimeError("Cannot load verifier dependencies; check torch/transformers and libstdc++") from exc
        self.torch, self.device, self.batch_size, self.max_length = torch, device, batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        selected_dtype = dtype if dtype != "auto" else ("float16" if device.startswith("cuda") else "float32")
        torch_dtype = getattr(torch, selected_dtype)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch_dtype)
        except TypeError:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model = self.model.to(device).eval()
        self.model_revision = getattr(self.model.config, "_commit_hash", None)
        id2label = getattr(self.model.config, "id2label", {})
        self.label_ids: dict[str, int] = {}
        for raw_id, raw_name in id2label.items():
            name, label_id = str(raw_name).lower().replace("_", "-"), int(raw_id)
            for canonical in LABEL_NAMES.values():
                if canonical in name:
                    self.label_ids[canonical] = label_id
        if set(self.label_ids) != set(LABEL_NAMES.values()):
            raise ValueError(f"Could not resolve verifier labels from config: {id2label}")

    def predict_proba(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        result: list[dict[str, float]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            encoded = self.tokenizer([left for left, _ in batch], [right for _, right in batch],
                                      return_tensors="pt", padding=True, truncation=True,
                                      max_length=self.max_length).to(self.device)
            with self.torch.inference_mode():
                probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1).cpu()
            result.extend({name: float(row[index]) for name, index in self.label_ids.items()} for row in probabilities)
        return result

