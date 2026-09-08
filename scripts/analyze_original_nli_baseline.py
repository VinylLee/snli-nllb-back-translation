#!/usr/bin/env python3
"""Run the original-pair NLI baseline and build the calibration review package."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.quality_filter import LABEL_NAMES, TransformersNLI
except ModuleNotFoundError:  # direct execution
    from quality_filter import LABEL_NAMES, TransformersNLI


TRANSITIONS = (
    "STABLE_GOLD", "TRUE_FLIP", "VERIFIER_GOLD_DISAGREEMENT_STABLE",
    "RECOVERED_TO_GOLD", "NON_GOLD_TRANSITION",
)
REVIEW_COLUMNS = [
    "source_index", "original_source_index", "candidate_id", "augmented_field", "gold_label",
    "original_premise", "original_hypothesis", "augmented_premise", "augmented_hypothesis",
    "original_sentence", "back_translated_sentence", "semantic_forward", "semantic_backward",
    "semantic_min", "original_nli_predicted_label", "original_nli_gold_probability",
    "candidate_nli_predicted_label", "candidate_nli_gold_probability", "nli_gold_probability_delta",
    "nli_transition", "confidence_drop", "hard_cue_changes", "soft_cue_changes", "change_ratio",
    "length_ratio", "accepted", "reasons", "review_categories", "number_before_raw",
    "number_after_raw", "number_change_type", "review_semantic_equivalent",
    "review_label_preserved", "review_useful_augmentation", "review_verdict", "review_notes",
]
AUDIT_APPEND_COLUMNS = [
    "original_nli_predicted_label", "original_nli_gold_probability", "original_entailment_probability",
    "original_neutral_probability", "original_contradiction_probability", "candidate_nli_predicted_label",
    "candidate_nli_gold_probability", "original_matches_gold", "candidate_matches_gold",
    "nli_transition", "nli_gold_probability_delta", "confidence_drop", "large_confidence_drop",
    "large_confidence_increase", "original_source_index",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_original_baseline(source_path: Path, output_path: Path, model_name: str,
                          device: str, dtype: str, batch_size: int,
                          expected_revision: str | None = None) -> dict[str, Any]:
    sources = read_jsonl(source_path)
    verifier = TransformersNLI(model_name, device, dtype, batch_size)
    if expected_revision and verifier.model_revision != expected_revision:
        raise RuntimeError(f"Verifier revision mismatch: expected {expected_revision}, got {verifier.model_revision}")
    probabilities = verifier.predict_proba([(row["premise"], row["hypothesis"]) for row in sources])
    rows = []
    for calibration_index, (source, probability) in enumerate(zip(sources, probabilities)):
        gold_id = int(source["label"])
        predicted_id = max(LABEL_NAMES, key=lambda label: float(probability.get(LABEL_NAMES[label], 0.0)))
        rows.append({
            "calibration_index": calibration_index, "source_index": source["source_index"], "premise": source["premise"],
            "hypothesis": source["hypothesis"], "gold_label": gold_id,
            "original_nli_predicted_label": LABEL_NAMES[predicted_id],
            "original_nli_gold_probability": probability.get(LABEL_NAMES[gold_id], 0.0),
            "original_entailment_probability": probability.get("entailment", 0.0),
            "original_neutral_probability": probability.get("neutral", 0.0),
            "original_contradiction_probability": probability.get("contradiction", 0.0),
            "original_matches_gold": predicted_id == gold_id, "verifier_model_revision": verifier.model_revision,
        })
    columns = [
        "calibration_index", "source_index", "premise", "hypothesis", "gold_label", "original_nli_predicted_label",
        "original_nli_gold_probability", "original_entailment_probability",
        "original_neutral_probability", "original_contradiction_probability", "original_matches_gold", "verifier_model_revision",
    ]
    write_csv(output_path, rows, columns)
    return {"rows": rows, "model_revision": verifier.model_revision, "label_ids": verifier.label_ids}


def _json(value: Any) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return []


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def classify_transition(original_pred: str, candidate_pred: str, gold: str) -> str:
    if original_pred == gold and candidate_pred == gold:
        return "STABLE_GOLD"
    if original_pred == gold and candidate_pred != gold:
        return "TRUE_FLIP"
    if original_pred != gold and candidate_pred == original_pred:
        return "VERIFIER_GOLD_DISAGREEMENT_STABLE"
    if original_pred != gold and candidate_pred == gold:
        return "RECOVERED_TO_GOLD"
    return "NON_GOLD_TRANSITION"


def _canonical_number(value: str) -> int | float | None:
    words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "couple": 2, "dozen": 12,
        "hundred": 100, "thousand": 1000,
    }
    if value in words:
        return words[value]
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return None


def number_change_type(before_raw: str, after_raw: str) -> str:
    before, after = _json(before_raw), _json(after_raw)
    if not before and after:
        return "NUMBER_ADDED"
    if before and not after:
        return "NUMBER_REMOVED"
    if not before and not after:
        return ""
    before_values = sorted((_canonical_number(str(key)) for key in before for _ in range(int(before[key]))), key=str)
    after_values = sorted((_canonical_number(str(key)) for key in after for _ in range(int(after[key]))), key=str)
    if before_values == after_values and before != after:
        return "LEXICAL_NORMALIZATION"
    return "POSSIBLE_VALUE_CHANGE"


def join_audit(audit_path: Path, baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with audit_path.open(encoding="utf-8", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    baseline = {str(row["calibration_index"]): row for row in baseline_rows}
    joined = []
    for row in audit_rows:
        base = baseline[str(row["source_index"])]
        gold = str(row["gold_label"])
        raw_candidate_pred = str(row.get("nli_predicted_label", "")); candidate_pred = LABEL_NAMES.get(int(raw_candidate_pred), raw_candidate_pred) if raw_candidate_pred.lstrip("-").isdigit() else raw_candidate_pred
        original_pred = base["original_nli_predicted_label"]
        candidate_probability = _float(row.get("nli_gold_probability"))
        original_probability = float(base["original_nli_gold_probability"])
        delta = None if candidate_probability is None else candidate_probability - original_probability
        transition = classify_transition(original_pred, candidate_pred, gold)
        hard = _json(row.get("hard_cue_changes", "[]"))
        soft = _json(row.get("soft_cue_changes", "[]"))
        number_before, number_after = "", ""
        for change in hard:
            if change.get("group") == "number":
                number_before = json.dumps(change.get("original", {}), ensure_ascii=False, separators=(",", ":"))
                number_after = json.dumps(change.get("candidate", {}), ensure_ascii=False, separators=(",", ":"))
                break
        joined.append({
            **row,
            "original_source_index": base["source_index"], "original_nli_predicted_label": original_pred,
            "original_nli_gold_probability": original_probability,
            "original_entailment_probability": base["original_entailment_probability"],
            "original_neutral_probability": base["original_neutral_probability"],
            "original_contradiction_probability": base["original_contradiction_probability"],
            "candidate_nli_predicted_label": candidate_pred,
            "candidate_nli_gold_probability": candidate_probability,
            "original_matches_gold": original_pred == gold,
            "candidate_matches_gold": candidate_pred == gold,
            "nli_transition": transition,
            "nli_gold_probability_delta": delta,
            "confidence_drop": transition == "STABLE_GOLD" and delta is not None and delta < -0.15,
            "large_confidence_drop": delta is not None and delta <= -0.15,
            "large_confidence_increase": delta is not None and delta >= 0.15,
            "_hard_groups": {str(change.get("group")) for change in hard},
            "_soft_groups": {str(change.get("group")) for change in soft},
            "_number_before": number_before,
            "_number_after": number_after,
        })
    return joined


def _review_row(row: dict[str, Any]) -> dict[str, Any]:
    forward = row.get("semantic_forward_entailment")
    backward = row.get("semantic_backward_entailment")
    return {
        "source_index": row.get("source_index"), "original_source_index": row.get("original_source_index"), "candidate_id": row.get("candidate_id"),
        "augmented_field": row.get("augmented_field"), "gold_label": row.get("gold_label"),
        "original_premise": row.get("original_premise"), "original_hypothesis": row.get("original_hypothesis"),
        "augmented_premise": row.get("augmented_premise"), "augmented_hypothesis": row.get("augmented_hypothesis"),
        "original_sentence": row.get("original_sentence"), "back_translated_sentence": row.get("back_translated_sentence"),
        "semantic_forward": forward, "semantic_backward": backward, "semantic_min": row.get("semantic_min"),
        "original_nli_predicted_label": row.get("original_nli_predicted_label"),
        "original_nli_gold_probability": row.get("original_nli_gold_probability"),
        "candidate_nli_predicted_label": row.get("candidate_nli_predicted_label"),
        "candidate_nli_gold_probability": row.get("candidate_nli_gold_probability"),
        "nli_gold_probability_delta": row.get("nli_gold_probability_delta"),
        "nli_transition": row.get("nli_transition"), "confidence_drop": row.get("confidence_drop"),
        "hard_cue_changes": row.get("hard_cue_changes"), "soft_cue_changes": row.get("soft_cue_changes"),
        "change_ratio": row.get("change_ratio"), "length_ratio": row.get("length_ratio"),
        "accepted": row.get("accepted"), "reasons": row.get("reasons"),
        "review_categories": row.get("review_categories", ""),
        "number_before_raw": row.get("_number_before", ""), "number_after_raw": row.get("_number_after", ""),
        "number_change_type": number_change_type(row.get("_number_before", ""), row.get("_number_after", "")),
        "review_semantic_equivalent": "", "review_label_preserved": "",
        "review_useful_augmentation": "", "review_verdict": "", "review_notes": "",
    }


def _stratified_sample(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return list(rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("gold_label"))].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    labels = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for label in labels:
            if groups[label] and len(selected) < limit:
                selected.append(groups[label].pop())
    return selected


def _sample_random(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    return list(rows) if len(rows) <= limit else random.Random(seed).sample(rows, limit)


def _category_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    categories: dict[str, list[dict[str, Any]]] = {
        "TRUE_FLIP": [row for row in rows if row["nli_transition"] == "TRUE_FLIP"],
        "VERIFIER_GOLD_DISAGREEMENT": [row for row in rows if row["nli_transition"] == "VERIFIER_GOLD_DISAGREEMENT_STABLE"],
        "SCORE_BORDERLINE": [row for row in rows if _bool(row.get("accepted")) and ((row.get("semantic_min") not in (None, "") and float(row["semantic_min"]) < 0.92) or (row.get("candidate_nli_gold_probability") not in (None, "") and float(row["candidate_nli_gold_probability"]) < 0.92))],
        "HIGH_CONFIDENCE": [row for row in rows if _bool(row.get("accepted")) and all(row.get(key) not in (None, "") and float(row[key]) >= 0.97 for key in ("semantic_forward_entailment", "semantic_backward_entailment", "candidate_nli_gold_probability"))],
        "SEMANTIC_DRIFT": [row for row in rows if "semantic_drift" in _json(row.get("reasons", "[]"))],
        "NUMBER_CUE": [row for row in rows if "number" in row["_hard_groups"]],
        "NEGATION_CUE": [row for row in rows if "negation" in row["_hard_groups"]],
    }
    categories["LABEL_FLIP"] = [row for row in rows if "label_flip" in _json(row.get("reasons", "[]"))]
    categories["HARD_CUE"] = [row for row in rows if "hard_cue_changed" in _json(row.get("reasons", "[]"))]
    categories["LOW_CONFIDENCE"] = [row for row in rows if "low_nli_confidence" in _json(row.get("reasons", "[]"))]
    return categories


def build_review_package(rows: list[dict[str, Any]], output_dir: Path, seed: int) -> dict[str, int]:
    categories = _category_rows(rows)
    selections = {
        "review_true_flip.csv": _stratified_sample(categories["TRUE_FLIP"], 100, seed),
        "review_verifier_gold_disagreement.csv": _stratified_sample(categories["VERIFIER_GOLD_DISAGREEMENT"], 75, seed + 1),
        "review_accepted_score_borderline.csv": sorted(categories["SCORE_BORDERLINE"], key=lambda row: min(float(row["semantic_min"]), float(row["candidate_nli_gold_probability"])))[:100],
        "review_accepted_high_confidence.csv": _sample_random(categories["HIGH_CONFIDENCE"], 75, seed),
        "review_semantic_drift.csv": [],
        "number_cue_audit.csv": categories["NUMBER_CUE"],
        "review_negation_cue.csv": categories["NEGATION_CUE"],
    }
    semantic = categories["SEMANTIC_DRIFT"]
    obvious = [row for row in semantic if row.get("semantic_min") not in (None, "") and float(row["semantic_min"]) < 0.20]
    boundary = [row for row in semantic if row.get("semantic_min") not in (None, "") and 0.60 <= float(row["semantic_min"]) < 0.80]
    selected = _sample_random(obvious, 38, seed) + _sample_random(boundary, 37, seed + 1)
    if len(selected) < min(75, len(semantic)):
        selected_ids = {row["candidate_id"] for row in selected}
        selected.extend(_sample_random([row for row in semantic if row["candidate_id"] not in selected_ids], 75 - len(selected), seed + 2))
    selections["review_semantic_drift.csv"] = selected[:75]
    for filename, selected_rows in selections.items():
        write_csv(output_dir / filename, (_review_row(row) for row in selected_rows), REVIEW_COLUMNS)

    master: dict[str, dict[str, Any]] = {}
    for category, selected_rows in selections.items():
        for row in selected_rows:
            key = str(row["candidate_id"])
            if key not in master:
                master[key] = dict(row)
            existing = set(filter(None, str(master[key].get("review_categories", "")).split("|")))
            category_name = category.removesuffix(".csv").upper().removeprefix("REVIEW_").removesuffix("_AUDIT")
            existing.add(category_name)
            master[key]["review_categories"] = "|".join(sorted(existing))
    master_rows = sorted(master.values(), key=lambda row: (int(row["source_index"]), row["candidate_id"]))
    write_csv(output_dir / "MASTER_REVIEW.csv", (_review_row(row) for row in master_rows), REVIEW_COLUMNS)
    return {filename: len(selected_rows) for filename, selected_rows in selections.items()} | {"MASTER_REVIEW.csv": len(master_rows)}


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _pct(count: int, total: int) -> str:
    return f"{count / total * 100:.2f}%" if total else "n/a"


def write_report(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], output_dir: Path,
                 source_commit: str, baseline_model_revision: str, seed: int,
                 device: str, batch_size: int, runtime_seconds: float | None,
                 review_counts: dict[str, int]) -> None:
    total = len(rows)
    transition_counts = Counter(row["nli_transition"] for row in rows)
    transition_by_label = []
    for label in ("entailment", "neutral", "contradiction"):
        subset = [row for row in rows if row["gold_label"] == label]
        transition_by_label.extend((label, transition, sum(row["nli_transition"] == transition for row in subset)) for transition in TRANSITIONS)
    transition_by_field = []
    for field in ("premise", "hypothesis"):
        subset = [row for row in rows if row["augmented_field"] == field]
        transition_by_field.extend((field, transition, sum(row["nli_transition"] == transition for row in subset)) for transition in TRANSITIONS)
    label_flip = [row for row in rows if "label_flip" in _json(row.get("reasons", "[]"))]
    label_flip_counts = Counter(row["nli_transition"] for row in label_flip)
    deltas = [float(row["nli_gold_probability_delta"]) for row in rows if row["nli_gold_probability_delta"] is not None]
    accepted = [row for row in rows if _bool(row.get("accepted"))]
    rejected = [row for row in rows if not _bool(row.get("accepted"))]
    original_agreement = sum(_bool(row["original_matches_gold"]) for row in baseline_rows)
    by_label = []
    for label_id in (0, 1, 2):
        subset = [row for row in baseline_rows if int(row["gold_label"]) == label_id]
        agreement = sum(_bool(row["original_matches_gold"]) for row in subset)
        by_label.append((LABEL_NAMES[label_id], len(subset), agreement, len(subset) - agreement, _pct(agreement, len(subset))))
    candidate_stability = [
        ("all candidates", total, sum(row["candidate_nli_predicted_label"] == row["original_nli_predicted_label"] for row in rows)),
        ("accepted", len(accepted), sum(row["candidate_nli_predicted_label"] == row["original_nli_predicted_label"] for row in accepted)),
        ("rejected", len(rejected), sum(row["candidate_nli_predicted_label"] == row["original_nli_predicted_label"] for row in rejected)),
    ]
    report = ["# Original NLI Baseline Report\n", "## Original SNLI verifier baseline\n"]
    report.append(_table(["subset", "total", "agrees with gold", "disagrees", "accuracy"], [("overall", len(baseline_rows), original_agreement, len(baseline_rows) - original_agreement, _pct(original_agreement, len(baseline_rows)))] + by_label))
    report.append("\nVerifier model revision: `" + str(baseline_model_revision) + "`. NLI threshold remains `0.80`; no filtering decision was changed.\n")
    report.append("\n## Candidate transitions\n")
    report.append(_table(["transition", "count", "percentage"], [(transition, transition_counts[transition], _pct(transition_counts[transition], total)) for transition in TRANSITIONS]))
    report.append("\n### By gold label\n")
    report.append(_table(["label", "transition", "count"], transition_by_label))
    report.append("\n### By augmented field\n")
    report.append(_table(["field", "transition", "count"], transition_by_field))
    report.append("\n## Existing label_flip decomposition\n")
    report.append(_table(["category", "count", "percentage of label_flip"], [(category, label_flip_counts[category], _pct(label_flip_counts[category], len(label_flip))) for category in TRANSITIONS]))
    report.append(f"\nExisting label_flip total: {len(label_flip)}.\n")
    report.append("\n## Candidate-original prediction stability\n")
    report.append(_table(["subset", "total", "same prediction", "stability"], [(name, count, same, _pct(same, count)) for name, count, same in candidate_stability]))
    report.append("\n## Confidence transitions\n")
    report.append(_table(["metric", "value"], [
        ("mean candidate-original gold probability delta", f"{statistics.fmean(deltas):.6f}" if deltas else "n/a"),
        ("large confidence drop (delta <= -0.15)", sum(row["large_confidence_drop"] for row in rows)),
        ("large confidence increase (delta >= +0.15)", sum(row["large_confidence_increase"] for row in rows)),
        ("confidence_drop flag (STABLE_GOLD and delta < -0.15)", sum(row["confidence_drop"] for row in rows)),
    ]))
    report.append("\n## Review package\n")
    report.append(_table(["file", "rows"], sorted(review_counts.items())))
    report.append("\nAll review labels and verdict columns are intentionally blank for GPT/human annotation.\n")
    report.append("\n## Configuration and provenance\n")
    report.append(_table(["item", "value"], [
        ("calibration repository commit", source_commit), ("sample seed", seed),
        ("verifier model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
        ("verifier revision", baseline_model_revision), ("device", device),
        ("batch size", batch_size), ("wall-clock baseline seconds", runtime_seconds if runtime_seconds is not None else "not supplied"),
        ("NLI threshold", "0.80"),
    ]))
    report.append("\nThe original baseline was run on the 800 committed source pairs only; NLLB was not run in this phase.\n")
    (output_dir / "ORIGINAL_NLI_BASELINE_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def build_package(audit_path: Path, baseline_path: Path, output_dir: Path,
                  source_commit: str, seed: int, device: str, batch_size: int,
                  runtime_seconds: float | None) -> dict[str, Any]:
    baseline_rows = list(csv.DictReader(baseline_path.open(encoding="utf-8", newline="")))
    # Convert baseline CSV scalar values to the types expected by the report.
    for row in baseline_rows:
        row["calibration_index"] = int(row["calibration_index"]); row["gold_label"] = int(row["gold_label"])
        row["original_nli_gold_probability"] = float(row["original_nli_gold_probability"])
        for key in ("original_entailment_probability", "original_neutral_probability", "original_contradiction_probability"):
            row[key] = float(row[key])
    rows = join_audit(audit_path, baseline_rows)
    audit_columns = list(csv.DictReader(audit_path.open(encoding="utf-8", newline="")).fieldnames or []) + AUDIT_APPEND_COLUMNS
    write_csv(output_dir / "snli_calibration_800.audit_with_original_nli.csv", rows, audit_columns)
    review_counts = build_review_package(rows, output_dir, seed)
    write_report(rows, baseline_rows, output_dir, source_commit, str(baseline_rows[0].get("verifier_model_revision", "")) if baseline_rows else "", seed, device, batch_size, runtime_seconds, review_counts)
    return {"candidate_count": len(rows), "review_counts": review_counts, "baseline_rows": baseline_rows, "rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    parser.add_argument("--expected-revision", default="6f5cf0a2b59cabb106aca4c287eed12e357e90eb")
    parser.add_argument("--source-commit", default="9015a4d")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--runtime-seconds", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / "original_pair_verifier_baseline.csv"
    baseline_result = run_original_baseline(args.sources, baseline_path, args.model_name, args.device, args.dtype, args.batch_size, args.expected_revision)
    package = build_package(args.audit, baseline_path, args.output_dir, args.source_commit, args.seed, args.device, args.batch_size, args.runtime_seconds)
    print(json.dumps({"baseline_count": len(baseline_result["rows"]), "model_revision": baseline_result["model_revision"], "candidate_count": package["candidate_count"], "review_counts": package["review_counts"]}, ensure_ascii=False, indent=2))
