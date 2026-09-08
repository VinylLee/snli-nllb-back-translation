"""Explainable, batchable quality checks for NLI back-translation candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Protocol, Sequence


LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
_NUMERIC_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?\Z")
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_REPEATED_CHUNK_RE = re.compile(r"\b(\w+(?:\s+\w+){1,5})\s+\1\b", re.IGNORECASE)
_IMPOSSIBLE_SWITCH_RE = re.compile(
    r"\b(turn|turning|switch|switching)\s+(?:on|off)\s+(?:a|an|the)\s+"
    r"(?:hamburger|burger|food|omelette|pizza|sandwich)\b", re.IGNORECASE)

# Cue changes are now diagnostic by default. Only an explicit, reliably
# comparable numeric value conflict is an independent hard failure.
HARD_CUE_TYPES = {"numeric_value"}
SOFT_CUE_TYPES = {"negation", "number", "quantifier", "modal", "time", "space"}
CUE_GROUPS: dict[str, set[str]] = {
    "negation": {"not", "no", "never", "nobody", "nothing", "without", "n't"},
    "number": {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty", "hundred", "thousand", "couple", "dozen", "both",
        "single", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    },
    "quantifier": {"all", "some", "any", "every", "few", "several", "many", "most", "couple"},
    "modal": {"may", "might", "can", "could", "must", "should", "will"},
    "time": {"before", "after", "during", "first", "later", "already"},
    "space": {"in", "inside", "outside", "on", "under", "over", "behind", "front", "through", "left", "right", "near", "beside"},
}

NUMBER_WORD_VALUES: dict[str, int | float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "hundred": 100, "thousand": 1000,
    "couple": 2, "both": 2, "single": 1,
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
    """Return raw cue counts; contraction expansion uses an immutable snapshot."""
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    contractions = ["n't" for token in tokens if token.endswith("n't")]
    tokens.extend(contractions)
    snapshot = {
        group: {cue: tokens.count(cue) for cue in cues if tokens.count(cue)}
        for group, cues in CUE_GROUPS.items()
    }
    for token in set(tokens):
        if _NUMERIC_TOKEN_RE.fullmatch(token):
            snapshot["number"][token] = tokens.count(token)
    return snapshot


def _canonical_negation(text: str) -> dict[str, int]:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    concepts: list[str] = []
    for index, token in enumerate(tokens):
        if token == "no" and index + 1 < len(tokens) and tokens[index + 1] == "one":
            concepts.append("nobody")
        elif token in {"not", "n't"} or token.endswith("n't"):
            concepts.append("not")
        elif token in {"nobody", "noone", "nothing", "never", "without", "no"}:
            concepts.append("nobody" if token in {"nobody", "noone"} else token)
    return {concept: concepts.count(concept) for concept in set(concepts)}


def _is_pronoun_one(tokens: list[str], index: int) -> bool:
    token = tokens[index]
    if token != "one":
        return False
    previous = tokens[index - 1] if index else ""
    following = tokens[index + 1] if index + 1 < len(tokens) else ""
    return previous in {"no", "the", "another"} or following in {"another", "of"}


def explicit_number_values(text: str) -> list[int | float]:
    """Extract comparable number values while avoiding pronoun ``one``."""
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    values: list[int | float] = []
    for index, token in enumerate(tokens):
        if _is_pronoun_one(tokens, index):
            continue
        if token in NUMBER_WORD_VALUES:
            values.append(NUMBER_WORD_VALUES[token])
            continue
        if _NUMERIC_TOKEN_RE.fullmatch(token):
            values.append(float(token) if "." in token else int(token))
    return values


def _raw_number_tokens(text: str) -> dict[str, int]:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    return {token: tokens.count(token) for token in set(tokens)
            if token in CUE_GROUPS["number"] or _NUMERIC_TOKEN_RE.fullmatch(token)}


def numeric_value_conflicts(original: str, candidate: str) -> bool:
    """Return true only when both sides expose comparable, different values."""
    before, after = explicit_number_values(original), explicit_number_values(candidate)
    return bool(before and after and before != after)


def _canonical_cue_snapshot(text: str, group: str) -> dict[str, int]:
    if group == "negation":
        return _canonical_negation(text)
    if group == "number":
        values = explicit_number_values(text)
        return {str(value): values.count(value) for value in set(values)}
    raw = cue_snapshot(text)[group]
    return raw


def logical_cue_changes(original: str, candidate: str) -> list[dict[str, Any]]:
    changes = []
    for group in CUE_GROUPS:
        before = _canonical_cue_snapshot(original, group)
        after = _canonical_cue_snapshot(candidate, group)
        if before != after:
            changes.append({"group": group, "original": before, "candidate": after})
    return changes


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
    def __init__(self, verifier: Verifier, semantic_threshold: float = 0.90,
                 nli_threshold: float = 0.80, min_change_ratio: float = 0.03,
                 hard_cue_types: set[str] | None = None) -> None:
        self.verifier = verifier
        self.semantic_threshold = semantic_threshold
        self.nli_threshold = nli_threshold
        self.min_change_ratio = min_change_ratio
        # Kept as an injectable compatibility attribute; generic cue groups do
        # not become vetoes in policy v2.
        self.hard_cue_types = hard_cue_types or HARD_CUE_TYPES

    def evaluate_batch(self, items: Sequence[dict[str, Any]]) -> list[Decision]:
        if not items:
            return []
        decisions = [Decision(True, [], {"logical_cue_changes": [], "hard_cue_changes": [],
                                        "hard_numeric_value_changes": [], "soft_cue_changes": []}, [])
                     for _ in items]
        semantic_pairs: list[tuple[str, str]] = []
        semantic_map: list[tuple[int, str]] = []
        nli_pairs: list[tuple[str, str]] = []
        for index, item in enumerate(items):
            fields = ("premise", "hypothesis") if item["augmented_field"] == "both" else (item["augmented_field"],)
            texts = {"premise": (item["original_premise"], item["candidate_premise"]),
                     "hypothesis": (item["original_hypothesis"], item["candidate_hypothesis"])}
            scores = decisions[index].scores
            decisions[index].reasons.extend(str(reason) for reason in item.get("pre_reasons", []))
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
                    if change["group"] in SOFT_CUE_TYPES:
                        scores["soft_cue_changes"].append(change)
                    if change["group"] == "number" and numeric_value_conflicts(original_text, candidate_text):
                        scores["hard_cue_changes"].append(change)
                        scores["hard_numeric_value_changes"].append(change)
                changed_groups = {change["group"] for change in changes}
                for group in sorted(changed_groups):
                    decisions[index].flags.append(f"{group}_changed")
                if "number" in changed_groups and numeric_value_conflicts(original_text, candidate_text):
                    decisions[index].reasons.append("numeric_value_changed")
                if changes and changed_groups.intersection(SOFT_CUE_TYPES):
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
