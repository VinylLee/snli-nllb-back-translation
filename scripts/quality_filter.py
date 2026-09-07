"""Explainable quality checks for NLI back-translation candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Protocol, Sequence


LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_REPEATED_CHUNK_RE = re.compile(r"\b(\w+(?:\s+\w+){1,5})\s+\1\b", re.IGNORECASE)
CUE_GROUPS: dict[str, set[str]] = {
    "negation": {"not", "no", "never", "nobody", "nothing", "without", "n't"},
    "quantity": {"all", "some", "any", "every", "few", "several", "many", "most", "couple", "one", "two", "three", "four", "five", "1", "2", "3", "4", "5"},
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
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    tokens.extend("n't" for token in tokens if token.endswith("n't"))
    snapshot: dict[str, dict[str, int]] = {}
    for group, cues in CUE_GROUPS.items():
        snapshot[group] = {cue: tokens.count(cue) for cue in cues if tokens.count(cue)}
    return snapshot


def logical_cue_changes(original: str, candidate: str) -> list[dict[str, Any]]:
    before, after = cue_snapshot(original), cue_snapshot(candidate)
    return [{"group": group, "original": before[group], "candidate": after[group]}
            for group in CUE_GROUPS if before[group] != after[group]]


def normalized_change_ratio(original: str, candidate: str) -> float:
    left, right = " ".join(original.lower().split()), " ".join(candidate.lower().split())
    return 1.0 - SequenceMatcher(None, left, right).ratio()


def corruption_reasons(text: str) -> list[str]:
    value = text.strip()
    if not value or _PUNCT_ONLY_RE.fullmatch(value) or _REPEATED_CHUNK_RE.search(value):
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
                 enable_logical_guard: bool = True) -> None:
        self.verifier = verifier
        self.semantic_threshold = semantic_threshold
        self.nli_threshold = nli_threshold
        self.min_change_ratio = min_change_ratio
        self.enable_logical_guard = enable_logical_guard

    def evaluate(self, original_premise: str, original_hypothesis: str,
                 candidate_premise: str, candidate_hypothesis: str, gold_label: int,
                 augmented_field: str, was_truncated: bool = False) -> Decision:
        fields = ("premise", "hypothesis") if augmented_field == "both" else (augmented_field,)
        texts = {"premise": (original_premise, candidate_premise),
                 "hypothesis": (original_hypothesis, candidate_hypothesis)}
        reasons: list[str] = []
        flags: list[str] = []
        scores: dict[str, Any] = {"logical_cue_changes": []}
        if was_truncated:
            reasons.append("input_too_long")
        semantic_pairs: list[tuple[str, str]] = []
        for field_name in fields:
            original_text, candidate_text = texts[field_name]
            reasons.extend(corruption_reasons(candidate_text))
            change_ratio = normalized_change_ratio(original_text, candidate_text)
            length_ratio = len(candidate_text) / max(len(original_text), 1)
            scores[f"{field_name}_change_ratio"] = change_ratio
            scores[f"{field_name}_length_ratio"] = length_ratio
            if change_ratio < self.min_change_ratio:
                reasons.append("trivial_copy")
            if length_ratio < 0.5 or length_ratio > 2.0:
                flags.append("length_ratio_risk")
            if self.enable_logical_guard:
                for change in logical_cue_changes(original_text, candidate_text):
                    change["field"] = field_name
                    scores["logical_cue_changes"].append(change)
                if logical_cue_changes(original_text, candidate_text):
                    reasons.append("logical_cue_changed")
            semantic_pairs.extend(((original_text, candidate_text), (candidate_text, original_text)))
        semantic_probs = self.verifier.predict_proba(semantic_pairs)
        for offset, field_name in enumerate(fields):
            forward = float(semantic_probs[offset * 2].get("entailment", 0.0))
            backward = float(semantic_probs[offset * 2 + 1].get("entailment", 0.0))
            scores[f"semantic_{field_name}_forward_entailment"] = forward
            scores[f"semantic_{field_name}_backward_entailment"] = backward
            if forward < self.semantic_threshold or backward < self.semantic_threshold:
                reasons.append("semantic_drift")
        pair_probs = self.verifier.predict_proba([(candidate_premise, candidate_hypothesis)])[0]
        pred_label = max(LABEL_NAMES, key=lambda label: float(pair_probs.get(LABEL_NAMES[label], 0.0)))
        gold_probability = float(pair_probs.get(LABEL_NAMES[gold_label], 0.0))
        scores["nli_predicted_label"] = pred_label
        scores["nli_gold_probability"] = gold_probability
        if pred_label != gold_label:
            reasons.append("label_flip")
        elif gold_probability < self.nli_threshold:
            reasons.append("low_nli_confidence")
        return Decision(not reasons, list(dict.fromkeys(reasons)), scores, list(dict.fromkeys(flags)))


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
        torch_dtype = getattr(torch, dtype if dtype != "auto" else ("float16" if device.startswith("cuda") else "float32"))
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch_dtype)
        except TypeError:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model = self.model.to(device).eval()
        self.model_name = model_name
        self.model_revision = getattr(self.model.config, "_commit_hash", None)
        id2label = getattr(self.model.config, "id2label", {})
        self.label_ids: dict[str, int] = {}
        for raw_id, raw_name in id2label.items():
            name = str(raw_name).lower().replace("_", "-")
            label_id = int(raw_id)
            for canonical in ("entailment", "neutral", "contradiction"):
                if canonical in name:
                    self.label_ids[canonical] = label_id
        if set(self.label_ids) != {"entailment", "neutral", "contradiction"}:
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
            for row in probabilities:
                result.append({name: float(row[index]) for name, index in self.label_ids.items()})
        return result

