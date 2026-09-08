"""Small, model-free helpers shared by Policy v2 runtime and simulation."""

from __future__ import annotations

from typing import Any

try:
    from scripts.quality_filter import LABEL_NAMES
except ModuleNotFoundError:
    from quality_filter import LABEL_NAMES


SOURCE_STATUSES = ("RELIABLE_GOLD", "GOLD_LOW_CONFIDENCE", "GOLD_DISAGREEMENT")


def predicted_label(probabilities: dict[str, float]) -> int:
    return max(LABEL_NAMES, key=lambda label: float(probabilities.get(LABEL_NAMES[label], 0.0)))


def classify_source_nli(probabilities: dict[str, float], gold_label: int,
                        threshold: float = 0.80) -> dict[str, Any]:
    predicted = predicted_label(probabilities)
    gold_probability = float(probabilities.get(LABEL_NAMES[gold_label], 0.0))
    if predicted != gold_label:
        status = "GOLD_DISAGREEMENT"
    elif gold_probability < threshold:
        status = "GOLD_LOW_CONFIDENCE"
    else:
        status = "RELIABLE_GOLD"
    return {
        "source_nli_status": status,
        "original_nli_predicted_label": LABEL_NAMES[predicted],
        "original_nli_gold_probability": gold_probability,
        "original_entailment_probability": float(probabilities.get("entailment", 0.0)),
        "original_neutral_probability": float(probabilities.get("neutral", 0.0)),
        "original_contradiction_probability": float(probabilities.get("contradiction", 0.0)),
        "source_nli_threshold": threshold,
    }
