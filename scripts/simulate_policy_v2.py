#!/usr/bin/env python3
"""Offline simulation of Quality Policy v2 on a persisted calibration audit."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.quality_filter import LABEL_NAMES, numeric_value_conflicts
    from scripts.policy_v2 import classify_source_nli
except ModuleNotFoundError:  # direct execution: python scripts/simulate_policy_v2.py
    from quality_filter import LABEL_NAMES, numeric_value_conflicts
    from policy_v2 import classify_source_nli


LABELS = tuple(LABEL_NAMES.values())
OUTPUT_COLUMNS = [
    "source_index", "original_source_index", "candidate_id", "gold_label", "augmented_field",
    "old_accepted", "old_reasons", "source_nli_status", "original_nli_predicted_label",
    "original_nli_gold_probability", "semantic_min", "candidate_nli_predicted_label",
    "candidate_nli_gold_probability", "nli_gold_probability_delta", "old_hard_cue_changes", "v2_accepted", "v2_reasons",
    "decision_change", "normalization_category", "original_premise", "original_hypothesis",
    "augmented_premise", "augmented_hypothesis", "original_sentence", "back_translated_sentence",
    "semantic_forward_entailment", "semantic_backward_entailment", "soft_cue_changes", "change_ratio",
]


def _json(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value) if value not in (None, "") else []
    except (TypeError, json.JSONDecodeError):
        return []


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _gold_label(value: Any) -> str:
    if str(value).isdigit():
        return LABEL_NAMES[int(value)]
    return str(value)


def source_status(row: dict[str, Any], threshold: float = 0.80) -> str:
    existing = str(row.get("source_nli_status", "")).strip()
    if existing in {"RELIABLE_GOLD", "GOLD_LOW_CONFIDENCE", "GOLD_DISAGREEMENT"}:
        return existing
    gold = _gold_label(row.get("gold_label"))
    original = str(row.get("original_nli_predicted_label", ""))
    probability = _float(row.get("original_nli_gold_probability"), 0.0) or 0.0
    return classify_source_nli({"entailment": float(row.get("original_entailment_probability", 0.0) or 0.0),
                                "neutral": float(row.get("original_neutral_probability", 0.0) or 0.0),
                                "contradiction": float(row.get("original_contradiction_probability", 0.0) or 0.0)},
                               LABELS.index(gold), threshold)["source_nli_status"] if original == "" else (
        "RELIABLE_GOLD" if original == gold and probability >= threshold else
        "GOLD_LOW_CONFIDENCE" if original == gold else "GOLD_DISAGREEMENT")


def _normalization_category(row: dict[str, Any]) -> str:
    changes = _json(row.get("old_hard_cue_changes", row.get("hard_cue_changes", "[]")))
    groups = {str(change.get("group")) for change in changes if isinstance(change, dict)}
    if "negation" in groups:
        return "negation normalization"
    if "number" in groups:
        return "number normalization"
    return "other cue"


def v2_decision(row: dict[str, Any], source_threshold: float = 0.80,
                semantic_threshold: float = 0.90, nli_threshold: float = 0.80) -> tuple[bool, list[str], str]:
    status = source_status(row, source_threshold)
    reasons: list[str] = []
    if status == "GOLD_DISAGREEMENT":
        reasons.append("source_label_disagreement")
    elif status == "GOLD_LOW_CONFIDENCE":
        reasons.append("source_nli_low_confidence")
    else:
        semantic = _float(row.get("semantic_min"))
        if semantic is None or semantic < semantic_threshold:
            reasons.append("semantic_drift")
        gold = _gold_label(row.get("gold_label"))
        candidate_pred = str(row.get("candidate_nli_predicted_label", ""))
        if not candidate_pred:
            raw = str(row.get("nli_predicted_label", ""))
            candidate_pred = LABEL_NAMES.get(int(raw), raw) if raw.lstrip("-").isdigit() else raw
        candidate_probability = _float(row.get("candidate_nli_gold_probability", row.get("nli_gold_probability")))
        if candidate_pred != gold:
            reasons.append("label_flip")
        elif candidate_probability is None or candidate_probability < nli_threshold:
            reasons.append("low_nli_confidence")
        original_sentence = str(row.get("original_sentence", ""))
        back_sentence = str(row.get("back_translated_sentence", ""))
        if original_sentence and back_sentence and numeric_value_conflicts(original_sentence, back_sentence):
            reasons.append("numeric_value_changed")
        old_reasons = _json(row.get("old_reasons", row.get("reasons", "[]")))
        for reason in old_reasons:
            if reason in {"corrupt_output", "trivial_copy", "input_too_long", "intermediate_input_too_long"}:
                reasons.append(reason)
    reasons = list(dict.fromkeys(reasons))
    accepted = not reasons
    return accepted, reasons, _normalization_category(row)


def _decision_change(old: bool, new: bool) -> str:
    if old and new:
        return "KEEP_ACCEPTED"
    if not old and not new:
        return "KEEP_REJECTED"
    return "NEWLY_ACCEPTED" if new else "NEWLY_REJECTED"


def simulate(rows: list[dict[str, Any]], source_threshold: float = 0.80,
             semantic_threshold: float = 0.90, nli_threshold: float = 0.80) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        old = _bool(row.get("accepted"))
        new, reasons, category = v2_decision(row, source_threshold, semantic_threshold, nli_threshold)
        status = source_status(row, source_threshold)
        result = {key: row.get(key, "") for key in OUTPUT_COLUMNS}
        result.update({
            "gold_label": _gold_label(row.get("gold_label")), "old_accepted": old,
            "old_reasons": json.dumps(_json(row.get("reasons", "[]")), ensure_ascii=False, separators=(",", ":")),
            "source_nli_status": status,
            "original_nli_predicted_label": row.get("original_nli_predicted_label", ""),
            "original_nli_gold_probability": row.get("original_nli_gold_probability", ""),
            "semantic_min": row.get("semantic_min", ""),
            "candidate_nli_predicted_label": row.get("candidate_nli_predicted_label", ""),
            "candidate_nli_gold_probability": row.get("candidate_nli_gold_probability", row.get("nli_gold_probability", "")),
            "nli_gold_probability_delta": row.get("nli_gold_probability_delta", ""),
            "old_hard_cue_changes": row.get("hard_cue_changes", ""), "v2_accepted": new,
            "v2_reasons": json.dumps(reasons, ensure_ascii=False, separators=(",", ":")),
            "decision_change": _decision_change(old, new), "normalization_category": category,
        })
        output.append(result)
    return output


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str] = OUTPUT_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rate(count: int, total: int) -> str:
    return f"{count / total * 100:.2f}%" if total else "n/a"


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(rows: list[dict[str, Any]], output_path: Path, source_commit: str = "unknown") -> dict[str, Any]:
    total = len(rows)
    old_accepted = sum(_bool(row["old_accepted"]) for row in rows)
    new_accepted = sum(_bool(row["v2_accepted"]) for row in rows)
    statuses = Counter(str(row["source_nli_status"]) for row in rows)
    source_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_rows.setdefault(str(row.get("source_index")), row)
    status_source = Counter(str(row["source_nli_status"]) for row in source_rows.values())
    status_v1 = Counter()
    for row in rows:
        if _bool(row["old_accepted"]):
            status_v1[str(row["source_nli_status"])] += 1
    transitions = Counter(str(row["decision_change"]) for row in rows)
    new_rejected = [row for row in rows if row["decision_change"] == "NEWLY_REJECTED"]
    new_accepted_rows = [row for row in rows if row["decision_change"] == "NEWLY_ACCEPTED"]
    v2_accepted = sum(_bool(row["v2_accepted"]) for row in rows)
    new_reject_categories = Counter()
    for row in new_rejected:
        reasons = _json(row["v2_reasons"])
        if "source_label_disagreement" in reasons:
            new_reject_categories["source_label_disagreement"] += 1
        if "source_nli_low_confidence" in reasons:
            new_reject_categories["source_nli_low_confidence"] += 1
        if _bool(row["old_accepted"]) and _float(row.get("semantic_min"), 0.0) is not None and 0.80 <= (_float(row.get("semantic_min"), 0.0) or 0.0) < 0.90:
            new_reject_categories["semantic threshold 0.80→0.90"] += 1
        if "numeric_value_changed" in reasons:
            new_reject_categories["numeric_value_changed"] += 1
    normalization = Counter(row["normalization_category"] for row in new_accepted_rows)
    deltas = [_float(row.get("nli_gold_probability_delta")) for row in rows]
    deltas = [value for value in deltas if value is not None]
    lines = ["# Quality Policy v2 Offline Simulation Report\n", "## Configuration\n",
             _table(["item", "value"], [("source commit", source_commit), ("input", "existing audit_with_original_nli.csv"),
             ("source NLI threshold", 0.80), ("semantic threshold", 0.90), ("candidate NLI threshold", 0.80),
             ("generation truncation", "excluded: historical audit has no EOS/max-new-token metadata")]),
             "\n## Source gate\n",
             _table(["status", "source count", "candidate rows", "V1 accepted rows"],
                    [(status, status_source[status], statuses[status], status_v1[status]) for status in ("RELIABLE_GOLD", "GOLD_LOW_CONFIDENCE", "GOLD_DISAGREEMENT")]),
             "\n## V1 vs V2\n", _table(["metric", "count", "rate"],
                    [("candidate total", total, "100.00%"), ("V1 accepted", old_accepted, _rate(old_accepted, total)),
                     ("V2 accepted", v2_accepted, _rate(v2_accepted, total)), ("V1 rejected", total-old_accepted, _rate(total-old_accepted, total)),
                     ("V2 rejected", total-v2_accepted, _rate(total-v2_accepted, total))]),
             "\nDecision changes: " + json.dumps(dict(transitions), ensure_ascii=False) + "\n"]
    for dimension, values in (("gold_label", LABELS), ("augmented_field", ("premise", "hypothesis"))):
        lines.append(f"\n## By {dimension}\n")
        table_rows = []
        for value in values:
            subset = [row for row in rows if (_gold_label(row.get("gold_label")) == value if dimension == "gold_label" else row.get(dimension) == value)]
            eligible = sum(str(row["source_nli_status"]) == "RELIABLE_GOLD" for row in subset)
            accepted = sum(_bool(row["v2_accepted"]) for row in subset)
            table_rows.append((value, len(subset), sum(_bool(row["old_accepted"]) for row in subset), accepted, eligible, _rate(accepted, eligible)))
        lines.append(_table([dimension, "total", "V1 accepted", "V2 accepted", "V2 eligible", "V2 rate"], table_rows))
    lines.extend(["\n## Newly accepted\n", _table(["category", "count"], sorted(normalization.items())),
                  "\nRepresentative newly accepted examples (up to 20):\n"])
    for row in new_accepted_rows[:20]:
        lines.append(f"- `{row['candidate_id']}` ({row['normalization_category']}): {row.get('original_sentence','')} → {row.get('back_translated_sentence','')}\n")
    lines.extend(["\n## Newly rejected\n", _table(["reason/category", "count"], sorted(new_reject_categories.items())),
                  "\n## Limitations\n",
                  "This is an offline reclassification of the persisted 1600-candidate audit. It does not load NLLB or rerun the verifier. "
                  "The historical audit did not record decoder EOS/max-new-token termination, so the new `generation_truncated` gate is not included. "
                  "Logical cue normalization is simulated from the persisted original/back-translated text; no new filtering decision was inferred from an unavailable model score.\n"])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return {"total": total, "v1_accepted": old_accepted, "v2_accepted": v2_accepted,
            "source_status": status_source, "transitions": transitions,
            "newly_accepted": normalization, "newly_rejected": new_reject_categories,
            "delta_mean": sum(deltas) / len(deltas) if deltas else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/nli/calibration/snli_calibration_800.audit_with_original_nli.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/nli/calibration/policy_v2_decisions.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/nli/calibration/POLICY_V2_SIMULATION_REPORT.md"))
    parser.add_argument("--source-commit", default="unknown")
    args = parser.parse_args(argv)
    result = simulate(read_csv(args.input))
    write_csv(args.output, result)
    summary = write_report(result, args.report, args.source_commit)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"source_status", "transitions", "newly_accepted", "newly_rejected"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
