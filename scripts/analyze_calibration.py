#!/usr/bin/env python3
"""Analyze calibration JSONL outputs and create human-review artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
HUMAN_COLUMNS = [
    "human_semantic_equivalent", "human_label_preserved",
    "human_useful_augmentation", "human_notes",
]
CSV_COLUMNS = [
    "source_index", "candidate_id", "augmented_field", "pivot_lang", "gold_label",
    "original_premise", "original_hypothesis", "augmented_premise", "augmented_hypothesis",
    "original_sentence", "back_translated_sentence", "accepted", "reasons",
    "semantic_forward_entailment", "semantic_backward_entailment", "semantic_min",
    "nli_predicted_label", "nli_gold_probability", "hard_cue_changes", "soft_cue_changes", "number_cues_before", "number_cues_after",
    "change_ratio", "length_ratio", "source_to_pivot_over_limit", "source_to_pivot_truncated",
    "pivot_to_source_over_limit", "pivot_to_source_truncated", "was_truncated",
    "translation_model", "translation_model_revision", "verifier_model", "verifier_model_revision",
    "risk_score", *HUMAN_COLUMNS,
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _cue_groups(changes: Iterable[dict[str, Any]]) -> set[str]:
    return {str(change.get("group")) for change in changes}


def _number_maps(changes: Iterable[dict[str, Any]]) -> tuple[str, str]:
    for change in changes:
        if change.get("group") == "number":
            return json_cell(change.get("original", {})), json_cell(change.get("candidate", {}))
    return "", ""


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    quality = record.get("quality", {})
    field = str(record.get("augmented_field", ""))
    original_sentence = record.get("original_premise", "") if field == "premise" else record.get("original_hypothesis", "")
    back_translated_sentence = record.get("premise", "") if field == "premise" else record.get("hypothesis", "")
    forward = quality.get(f"semantic_{field}_forward_entailment")
    backward = quality.get(f"semantic_{field}_backward_entailment")
    hard = quality.get("hard_cue_changes", [])
    soft = quality.get("soft_cue_changes", [])
    semantic_min = min(float(forward), float(backward)) if forward is not None and backward is not None else None
    nli_probability = quality.get("nli_gold_probability")
    risk_score = ((1.0 - semantic_min) if semantic_min is not None else 1.0)
    risk_score += (1.0 - float(nli_probability)) if nli_probability is not None else 1.0
    risk_score += 0.10 * len(soft)
    truncation = record.get("truncation", {})
    provenance = record.get("provenance", {})
    generation = provenance.get("generation", {})
    # Keep all audit fields scalar/CSV-safe; the original structured cue data
    # remains available as compact JSON in the corresponding cells.
    row = {
        "source_index": record.get("source_index"), "candidate_id": record.get("candidate_id"),
        "augmented_field": field, "pivot_lang": record.get("pivot_lang"),
        "gold_label": LABEL_NAMES.get(int(record.get("gold_label", record.get("label", -1))), record.get("gold_label")),
        "original_premise": record.get("original_premise"), "original_hypothesis": record.get("original_hypothesis"),
        "augmented_premise": record.get("premise"), "augmented_hypothesis": record.get("hypothesis"),
        "original_sentence": original_sentence, "back_translated_sentence": back_translated_sentence,
        "accepted": bool(quality.get("accepted", False)), "reasons": json_cell(quality.get("reasons", [])),
        "semantic_forward_entailment": forward, "semantic_backward_entailment": backward,
        "semantic_min": semantic_min, "nli_predicted_label": quality.get("nli_predicted_label"),
        "nli_gold_probability": nli_probability, "hard_cue_changes": json_cell(hard),
        "soft_cue_changes": json_cell(soft), "number_cues_before": _number_maps(hard)[0], "number_cues_after": _number_maps(hard)[1], "change_ratio": quality.get(f"{field}_change_ratio"),
        "length_ratio": quality.get(f"{field}_length_ratio"),
        "source_to_pivot_over_limit": truncation.get("source_to_pivot_over_limit", False),
        "source_to_pivot_truncated": truncation.get("source_to_pivot_truncated", False),
        "pivot_to_source_over_limit": truncation.get("pivot_to_source_over_limit", False),
        "pivot_to_source_truncated": truncation.get("pivot_to_source_truncated", False),
        "was_truncated": record.get("was_truncated", truncation.get("was_truncated", False)),
        "translation_model": provenance.get("translation_model"),
        "translation_model_revision": provenance.get("translation_model_revision"),
        "verifier_model": provenance.get("verifier_model"),
        "verifier_model_revision": provenance.get("verifier_model_revision"),
        "risk_score": round(risk_score, 8),
        "_hard_groups": _cue_groups(hard), "_soft_groups": _cue_groups(soft),
        "_number_before": _number_maps(hard)[0], "_number_after": _number_maps(hard)[1],
        "_generation": generation,
    }
    for column in HUMAN_COLUMNS:
        row[column] = ""
    return row


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sample(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return list(rows)
    return random.Random(seed).sample(rows, limit)


def _rate(accepted: int, total: int) -> str:
    return f"{accepted / total * 100:.2f}%" if total else "n/a"


def _count_table(rows: list[dict[str, Any]], key: str, values: Iterable[str]) -> list[tuple[str, int, int, int, str]]:
    result = []
    for value in values:
        subset = [row for row in rows if str(row.get(key)) == value]
        accepted = sum(bool(row["accepted"]) for row in subset)
        result.append((value, len(subset), accepted, len(subset) - accepted, _rate(accepted, len(subset))))
    return result


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("mean", "median", "p10", "p25", "p50", "p75", "p90")}
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered), "median": statistics.median(ordered),
        "p10": _percentile(ordered, 10), "p25": _percentile(ordered, 25),
        "p50": _percentile(ordered, 50), "p75": _percentile(ordered, 75),
        "p90": _percentile(ordered, 90),
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _fmt_distribution(distribution: dict[str, float | None]) -> str:
    return "; ".join(f"{key}={value:.4f}" if value is not None else f"{key}=n/a" for key, value in distribution.items())


def _markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def analyze(accepted_path: Path, rejected_path: Path, source_path: Path, output_dir: Path,
            source_commit: str, pipeline_commit: str, seed: int = 42,
            device: str = "unknown", runtime_seconds: float | None = None,
            batch_size: int = 16, filter_batch_size: int = 32, chunk_size: int = 32) -> dict[str, Any]:
    records = read_jsonl(accepted_path) + read_jsonl(rejected_path)
    rows = [flatten(record) for record in records]
    rows.sort(key=lambda row: (int(row["source_index"]), str(row["augmented_field"])))
    prefix = output_dir / "snli_calibration_800"
    write_csv(prefix.with_suffix(".audit.csv"), rows)

    accepted = [row for row in rows if row["accepted"]]
    rejected = [row for row in rows if not row["accepted"]]
    borderline = [row for row in accepted if (
        (row["semantic_min"] is not None and 0.80 <= row["semantic_min"] < 0.90)
        or (row["nli_gold_probability"] is not None and 0.80 <= float(row["nli_gold_probability"]) < 0.90)
        or bool(row["_soft_groups"])
        or (not row["_hard_groups"] and row["change_ratio"] is not None and float(row["change_ratio"]) >= 0.30)
    )]
    borderline.sort(key=lambda row: float(row["risk_score"]), reverse=True)
    write_csv(prefix.with_name("accepted_borderline.csv"), borderline[:150])
    high_confidence = [row for row in accepted if (
        row["semantic_forward_entailment"] is not None and float(row["semantic_forward_entailment"]) >= 0.95
        and row["semantic_backward_entailment"] is not None and float(row["semantic_backward_entailment"]) >= 0.95
        and row["nli_gold_probability"] is not None and float(row["nli_gold_probability"]) >= 0.95
        and not row["_hard_groups"]
    )]
    write_csv(prefix.with_name("accepted_high_confidence.csv"), _sample(high_confidence, 100, seed))
    review_specs = {
        "semantic_drift": "rejected_semantic_drift.csv", "label_flip": "rejected_label_flip.csv",
        "hard_cue_changed": "rejected_hard_cue.csv", "low_nli_confidence": "rejected_low_confidence.csv",
    }
    for reason, filename in review_specs.items():
        subset = [row for row in rejected if reason in json.loads(row["reasons"])]
        write_csv(prefix.with_name(filename), _sample(subset, 75, seed))
    number_rows = [row for row in rejected if "number" in row["_hard_groups"]]
    write_csv(prefix.with_name("number_cue_audit.csv"), number_rows)

    source_rows = read_jsonl(source_path)
    source_labels = Counter(str(row.get("label")) for row in source_rows)
    unique_premises = len({row.get("premise") for row in source_rows})
    reason_counts = Counter(reason for row in rows for reason in json.loads(row["reasons"]))
    primary_counts = Counter(json.loads(row["reasons"])[0] for row in rejected if json.loads(row["reasons"]))
    field_table = _count_table(rows, "augmented_field", ("premise", "hypothesis"))
    label_table = _count_table(rows, "gold_label", ("entailment", "neutral", "contradiction"))
    label_field_table = []
    for label in ("entailment", "neutral", "contradiction"):
        for field in ("premise", "hypothesis"):
            subset = [row for row in rows if row["gold_label"] == label and row["augmented_field"] == field]
            accepted_count = sum(bool(row["accepted"]) for row in subset)
            label_field_table.append((label, field, len(subset), accepted_count, len(subset) - accepted_count, _rate(accepted_count, len(subset))))

    hard_counts = Counter(group for row in rejected for group in row["_hard_groups"])
    soft_counts = Counter(group for row in rows for group in row["_soft_groups"])
    soft_rows = [row for row in rows if row["_soft_groups"]]
    no_soft_rows = [row for row in rows if not row["_soft_groups"]]
    score_stats = {}
    for subset_name, subset in (("accepted", accepted), ("rejected", rejected)):
        score_stats[subset_name] = {
            "semantic_min": _distribution([float(row["semantic_min"]) for row in subset if row["semantic_min"] is not None]),
            "nli_gold_probability": _distribution([float(row["nli_gold_probability"]) for row in subset if row["nli_gold_probability"] is not None]),
        }
    number_pairs = Counter(f"{row['_number_before']} -> {row['_number_after']}" for row in number_rows)
    provenance = next((row for row in rows if row.get("translation_model")), {})
    generation = provenance.get("_generation", {})
    acceptance_rate = _rate(len(accepted), len(rows))
    borderline_pct = _rate(len(borderline), len(accepted))
    label_rates = [float(value[4][:-1]) for value in label_table if value[4] != "n/a"]
    field_rates = [float(value[4][:-1]) for value in field_table if value[4] != "n/a"]
    report = []
    report.append("# SNLI Calibration Report\n")
    report.append("## Configuration\n")
    report.append(_markdown_table(["item", "value"], [
        ("source commit", source_commit), ("pipeline commit", pipeline_commit), ("sample seed", seed),
        ("source count", len(source_rows)), ("unique premise count", unique_premises),
        ("sampling strategy", "label-balanced random traversal, global unique premise first"),
        ("translation model", provenance.get("translation_model", "unknown")),
        ("translation revision", provenance.get("translation_model_revision", "unknown")),
        ("verifier model", provenance.get("verifier_model", "unknown")),
        ("verifier revision", provenance.get("verifier_model_revision", "unknown")),
        ("pivot language", provenance.get("pivot_lang", "fra_Latn")), ("augmentation mode", "separate"),
        ("num_beams", generation.get("num_beams", 2)), ("max_input_tokens", generation.get("max_input_tokens", 128)),
        ("max_new_tokens", generation.get("max_new_tokens", 64)), ("semantic threshold", 0.80),
        ("NLI threshold", 0.80), ("change threshold", 0.03), ("allow truncation", False),
        ("device", device), ("batch size", batch_size), ("filter batch size", filter_batch_size),
        ("chunk size", chunk_size), ("wall-clock seconds", runtime_seconds if runtime_seconds is not None else "not supplied"),
    ]))
    report.append("\n## Overall results\n")
    report.append(_markdown_table(["metric", "value"], [("source_count", len(source_rows)), ("candidate_count", len(rows)), ("accepted", len(accepted)), ("rejected", len(rejected)), ("acceptance_rate", acceptance_rate)]))
    report.append("\n## Source label distribution\n")
    report.append(_markdown_table(["label", "source count"], [(LABEL_NAMES.get(int(label), label), count) for label, count in sorted(source_labels.items())]))
    report.append("\n## Field distribution\n")
    report.append(_markdown_table(["field", "total", "accepted", "rejected", "acceptance rate"], field_table))
    report.append("\n## Label distribution\n")
    report.append(_markdown_table(["label", "total", "accepted", "rejected", "acceptance rate"], label_table))
    report.append("\n## Label × field\n")
    report.append(_markdown_table(["label", "field", "total", "accepted", "rejected", "acceptance rate"], label_field_table))
    report.append("\n## Rejection reasons\n")
    report.append(_markdown_table(["reason", "count"], sorted(reason_counts.items())))
    report.append("\nPrimary reason is the first reason in the pipeline's reason list.\n")
    report.append(_markdown_table(["primary reason", "count"], sorted(primary_counts.items())))
    report.append("\n## Cue statistics\n")
    report.append(_markdown_table(["cue type", "candidate count"], [(group, hard_counts.get(group, 0)) for group in ("negation", "number")] + [(group, soft_counts.get(group, 0)) for group in ("quantifier", "modal", "time", "space")]))
    report.append("\n")
    report.append(_markdown_table(["soft cue status", "total", "accepted", "acceptance rate"], [
        ("soft-cue changed", len(soft_rows), sum(row["accepted"] for row in soft_rows), _rate(sum(row["accepted"] for row in soft_rows), len(soft_rows))),
        ("no soft-cue change", len(no_soft_rows), sum(row["accepted"] for row in no_soft_rows), _rate(sum(row["accepted"] for row in no_soft_rows), len(no_soft_rows))),
    ]))
    report.append("\n## Score distributions\n")
    report.append(_markdown_table(["subset", "metric", "distribution"], [(subset, metric, _fmt_distribution(values)) for subset, metrics in score_stats.items() for metric, values in metrics.items()]))
    report.append("\n## Borderline accepted\n")
    report.append(f"{len(borderline)} candidates matched the borderline criteria ({borderline_pct} of accepted); {min(len(borderline), 150)} exported to `accepted_borderline.csv`.\n")
    report.append("\n## Number cue observations\n")
    report.append(f"Number-cue rejected candidate count: {len(number_rows)}.\n")
    report.append(_markdown_table(["number cue change", "count"], number_pairs.most_common(20)) if number_pairs else "No number cue changes were observed.\n")
    report.append("\n## Acceptance imbalance\n")
    report.append(f"Label acceptance-rate range: {max(label_rates) - min(label_rates):.2f} percentage points; field range: {max(field_rates) - min(field_rates):.2f} percentage points. These are reported observations only; no threshold was changed.\n")
    report.append("\n## Review artifacts\n")
    report.append("All review CSVs reserve `human_semantic_equivalent`, `human_label_preserved`, `human_useful_augmentation`, and `human_notes` as empty columns.\n")
    report.append("\n## Human annotation standard\n")
    report.append("human_semantic_equivalent: yes, no, or borderline; judge whether the back-translated sentence is an approximately equivalent paraphrase.\n\nhuman_label_preserved: yes, no, or uncertain; judge whether the augmented pair retains the original gold NLI label.\n\nhuman_useful_augmentation: yes, no, or borderline; judge whether the example is natural, sufficiently changed, and useful for training.\n\nhuman_notes: free-form reviewer context.\n")
    report.append("\n## Observations\n")
    observations = []
    if label_rates and max(label_rates) - min(label_rates) >= 10:
        observations.append("label acceptance rates differ by at least 10 percentage points")
    if field_rates and max(field_rates) - min(field_rates) >= 10:
        observations.append("premise and hypothesis acceptance rates differ by at least 10 percentage points")
    if number_rows:
        observations.append("number hard-cue changes occur in rejected candidates and require human review for lexical false positives")
    if not observations:
        observations.append("no automatic calibration conclusion was drawn")
    report.extend(f"- {observation}\n" for observation in observations)
    report.append("\nThresholds remain fixed at semantic=0.80 and NLI=0.80; this report is for human calibration, not automatic threshold tuning.\n")
    report_path = prefix.with_name("CALIBRATION_REPORT.md")
    report_path.write_text("\n".join(report), encoding="utf-8")
    return {
        "source_count": len(source_rows), "unique_premise_count": unique_premises, "candidate_count": len(rows),
        "accepted_count": len(accepted), "rejected_count": len(rejected), "acceptance_rate": acceptance_rate,
        "field": field_table, "label": label_table, "label_field": label_field_table,
        "reasons": dict(reason_counts), "hard_cues": dict(hard_counts), "soft_cues": dict(soft_counts),
        "soft_changed": (len(soft_rows), sum(row["accepted"] for row in soft_rows)),
        "no_soft_changed": (len(no_soft_rows), sum(row["accepted"] for row in no_soft_rows)),
        "score_stats": score_stats, "borderline_count": len(borderline), "number_cue_rejected_count": len(number_rows),
        "report_path": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--pipeline-commit", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="unknown")
    parser.add_argument("--runtime-seconds", type=float)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--filter-batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = analyze(args.accepted, args.rejected, args.sources, args.output_dir,
                     args.source_commit, args.pipeline_commit, args.seed, args.device,
                     args.runtime_seconds, args.batch_size, args.filter_batch_size, args.chunk_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
