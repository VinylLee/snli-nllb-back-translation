#!/usr/bin/env python3
"""Build audit and comparison artifacts for a real Policy v2 calibration run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
AUDIT_COLUMNS = [
    "input_position", "source_index", "candidate_id", "augmented_field", "gold_label",
    "source_nli_status", "original_nli_predicted_label", "original_nli_gold_probability",
    "translation_performed", "original_sentence", "back_translated_sentence",
    "semantic_forward", "semantic_backward", "semantic_min", "candidate_nli_predicted_label",
    "candidate_nli_gold_probability", "generation_truncated", "source_to_pivot_generation_truncated",
    "pivot_to_source_generation_truncated", "logical_cue_changes", "hard_cue_changes",
    "soft_cue_changes", "change_ratio", "accepted", "reasons", "translation_model",
    "translation_model_revision", "verifier_model", "verifier_model_revision",
]
GENERATION_COLUMNS = [
    "source_index", "input_position", "candidate_id", "augmented_field", "original_sentence",
    "back_translated_sentence", "source_to_pivot_generation_truncated",
    "pivot_to_source_generation_truncated", "accepted", "reasons",
]
COMPARISON_COLUMNS = [
    "input_position", "source_index", "candidate_id", "augmented_field", "old_accepted",
    "real_accepted", "old_reasons", "real_reasons", "source_nli_status", "semantic_min",
    "candidate_nli_gold_probability", "generation_truncated", "decision_change", "difference_category",
]


def _json(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value) if value not in (None, "") else []
    except (TypeError, json.JSONDecodeError):
        return []


def _bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def label_name(value: Any) -> str:
    try:
        return LABEL_NAMES[int(value)]
    except (TypeError, ValueError, KeyError):
        return str(value)


def audit_row(record: dict[str, Any]) -> dict[str, Any]:
    quality = record.get("quality", {})
    truncation = record.get("truncation", {})
    field = record.get("augmented_field", "")
    original_sentence = record.get("original_hypothesis") if field == "hypothesis" else record.get("original_premise")
    back_sentence = record.get("hypothesis") if field == "hypothesis" else record.get("premise")
    forward = quality.get(f"semantic_{field}_forward_entailment")
    backward = quality.get(f"semantic_{field}_backward_entailment")
    return {
        "input_position": record.get("input_position"), "source_index": record.get("source_index"),
        "candidate_id": record.get("candidate_id"), "augmented_field": field,
        "gold_label": label_name(record.get("gold_label")), "source_nli_status": record.get("source_nli_status"),
        "original_nli_predicted_label": record.get("original_nli_predicted_label"),
        "original_nli_gold_probability": record.get("original_nli_gold_probability"),
        "translation_performed": record.get("translation_performed", False),
        "original_sentence": original_sentence, "back_translated_sentence": back_sentence,
        "semantic_forward": forward, "semantic_backward": backward,
        "semantic_min": min(float(forward), float(backward)) if forward is not None and backward is not None else "",
        "candidate_nli_predicted_label": label_name(quality.get("nli_predicted_label")),
        "candidate_nli_gold_probability": quality.get("nli_gold_probability"),
        "generation_truncated": truncation.get("generation_truncated", False),
        "source_to_pivot_generation_truncated": truncation.get("source_to_pivot_generation_truncated", False),
        "pivot_to_source_generation_truncated": truncation.get("pivot_to_source_generation_truncated", False),
        "logical_cue_changes": json.dumps(quality.get("logical_cue_changes", []), ensure_ascii=False, separators=(",", ":")),
        "hard_cue_changes": json.dumps(quality.get("hard_cue_changes", []), ensure_ascii=False, separators=(",", ":")),
        "soft_cue_changes": json.dumps(quality.get("soft_cue_changes", []), ensure_ascii=False, separators=(",", ":")),
        "change_ratio": quality.get(f"{field}_change_ratio", ""), "accepted": quality.get("accepted", False),
        "reasons": json.dumps(quality.get("reasons", []), ensure_ascii=False, separators=(",", ":")),
        "translation_model": record.get("provenance", {}).get("translation_model"),
        "translation_model_revision": record.get("provenance", {}).get("translation_model_revision"),
        "verifier_model": record.get("provenance", {}).get("verifier_model"),
        "verifier_model_revision": record.get("provenance", {}).get("verifier_model_revision"),
    }


def build_audit(records: list[dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    rows = [audit_row(record) for record in records]
    write_csv(output_path, rows, AUDIT_COLUMNS)
    return rows


def build_mapping(source_path: Path, output_path: Path) -> list[dict[str, Any]]:
    rows = []
    for input_position, record in enumerate(read_jsonl(source_path)):
        rows.append({"input_position": input_position, "source_index": record.get("source_index", input_position),
                     "premise": record.get("premise", ""), "hypothesis": record.get("hypothesis", "")})
    write_csv(output_path, rows, ["input_position", "source_index", "premise", "hypothesis"])
    return rows


def build_generation_audit(rows: list[dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    selected = [row for row in rows if _bool(row["generation_truncated"])]
    write_csv(output_path, selected, GENERATION_COLUMNS)
    return selected


def compare(real_rows: list[dict[str, Any]], offline_path: Path, mapping: list[dict[str, Any]], output_path: Path) -> tuple[list[dict[str, Any]], Counter]:
    index_to_position = {str(row["source_index"]): str(row["input_position"]) for row in mapping}
    offline = {}
    for row in read_csv(offline_path):
        position = index_to_position.get(str(row.get("source_index")), str(row.get("source_index")))
        offline[(position, str(row.get("augmented_field")))] = row
    comparison = []
    categories = Counter()
    for row in real_rows:
        old = offline.get((str(row["input_position"]), str(row["augmented_field"])), {})
        old_accepted = _bool(old.get("v2_accepted"))
        real_accepted = _bool(row["accepted"])
        if old_accepted == real_accepted:
            change, category = "MATCH", "none"
        else:
            change = "REAL_NEWLY_ACCEPTED" if real_accepted else "REAL_NEWLY_REJECTED"
            if not real_accepted and _bool(row["generation_truncated"]):
                category = "generation_truncation_new"
            elif str(row["source_nli_status"]) != str(old.get("source_nli_status", "")):
                category = "source-gate difference"
            elif ((_float(row.get("semantic_min")) is not None and abs((_float(row.get("semantic_min")) or 0) - .90) < .02)
                  or (_float(row.get("candidate_nli_gold_probability")) is not None and abs((_float(row.get("candidate_nli_gold_probability")) or 0) - .80) < .02)):
                category = "floating_point / verifier score boundary"
            else:
                category = "translation difference"
            categories[category] += 1
        comparison.append({"input_position": row["input_position"], "source_index": row["source_index"],
                           "candidate_id": row["candidate_id"], "augmented_field": row["augmented_field"],
                           "old_accepted": old.get("v2_accepted", False), "real_accepted": real_accepted,
                           "old_reasons": old.get("v2_reasons", "[]"), "real_reasons": row["reasons"],
                           "source_nli_status": row["source_nli_status"], "semantic_min": row["semantic_min"],
                           "candidate_nli_gold_probability": row["candidate_nli_gold_probability"],
                           "generation_truncated": row["generation_truncated"], "decision_change": change,
                           "difference_category": category})
    write_csv(output_path, comparison, COMPARISON_COLUMNS)
    return comparison, categories


def _rate(count: int, total: int) -> str:
    return f"{count / total * 100:.2f}%" if total else "n/a"


def write_report(rows: list[dict[str, Any]], comparison: list[dict[str, Any]], categories: Counter,
                 mapping: list[dict[str, Any]], output_path: Path, offline_accepted: int = 774) -> None:
    total = len(rows)
    accepted = sum(_bool(row["accepted"]) for row in rows)
    translated = sum(_bool(row["translation_performed"]) for row in rows)
    status_sources = {}
    for row in rows:
        status_sources.setdefault(str(row["input_position"]), row["source_nli_status"])
    statuses = Counter(status_sources.values())
    reasons = Counter(reason for row in rows for reason in _json(row["reasons"]))
    first = rows[0] if rows else {}
    lines = ["# Real Policy V2 Calibration Report\n", "## Configuration\n",
             "- source sample: `data/nli/calibration/snli_calibration_800_sources.jsonl` (800 records, seed 42)\n",
             "- pivot: `fra_Latn`; augmentation mode: `separate`\n",
             "- translation model: `facebook/nllb-200-distilled-600M`\n",
             f"- translation model revision: `{first.get('translation_model_revision', 'unknown')}`\n",
             "- verifier: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`\n",
             f"- verifier revision: `{first.get('verifier_model_revision', 'unknown')}`\n",
             "- num_beams: 2; do_sample: false; max_input_tokens: 128; max_new_tokens: 64\n",
             "- source NLI gate: on; source NLI threshold: 0.80; semantic threshold: 0.90; candidate NLI threshold: 0.80\n",
             "- batch_size: 16; filter_batch_size: 32; chunk_size: 32; allow_truncation: false\n",
             "- device: CPU; dtype: float32 (CUDA run was unavailable because the installed PyTorch CUDA runtime required a newer driver)\n",
             "\n## Run summary\n",
             f"- source total: {len(mapping)}\n- candidate total: {total}\n- translation performed: {translated}\n- translation skipped: {total-translated}\n",
             f"- accepted: {accepted}\n- rejected: {total-accepted}\n- overall acceptance: {_rate(accepted,total)}\n- accepted / translated eligible: {_rate(accepted, translated)}\n",
             "\n## Source gate\n", "| status | source count | candidate count |\n| --- | ---: | ---: |\n"]
    lines.extend(f"| {status} | {statuses[status]} | {sum(row['source_nli_status'] == status for row in rows)} |\n" for status in ("RELIABLE_GOLD", "GOLD_LOW_CONFIDENCE", "GOLD_DISAGREEMENT"))
    lines.extend(["\n## By label\n", "| label | total | accepted | eligible | overall rate | eligible rate |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"])
    for label in ("entailment", "neutral", "contradiction"):
        subset = [row for row in rows if row["gold_label"] == label]
        eligible = sum(row["source_nli_status"] == "RELIABLE_GOLD" for row in subset)
        good = sum(_bool(row["accepted"]) for row in subset)
        lines.append(f"| {label} | {len(subset)} | {good} | {eligible} | {_rate(good,len(subset))} | {_rate(good,eligible)} |\n")
    lines.extend(["\n## By field\n", "| field | total | accepted | eligible | overall rate | eligible rate |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"])
    for field in ("premise", "hypothesis"):
        subset = [row for row in rows if row["augmented_field"] == field]
        eligible = sum(row["source_nli_status"] == "RELIABLE_GOLD" for row in subset)
        good = sum(_bool(row["accepted"]) for row in subset)
        lines.append(f"| {field} | {len(subset)} | {good} | {eligible} | {_rate(good,len(subset))} | {_rate(good,eligible)} |\n")
    lines.extend(["\n## Rejection reasons\n", "| reason | count |\n| --- | ---: |\n"])
    lines.extend(f"| {reason} | {count} |\n" for reason, count in sorted(reasons.items()))
    lines.extend(["\n## Offline vs real\n", f"- offline V2 accepted: {offline_accepted}\n- real V2 accepted: {accepted}\n- matching decisions: {sum(row['decision_change'] == 'MATCH' for row in comparison)}\n- different decisions: {sum(row['decision_change'] != 'MATCH' for row in comparison)}\n",
                  "\nDifference categories:\n", "| category | count |\n| --- | ---: |\n"])
    lines.extend(f"| {category} | {count} |\n" for category, count in sorted(categories.items()))
    canaries = {"148356": "Canary A", "227619": "Canary B", "512053": "Canary C"}
    lines.append("\n## Canary examples\n")
    for source_index, title in canaries.items():
        matches = [row for row in rows if str(row["source_index"]) == source_index and row["augmented_field"] == "hypothesis"]
        if not matches:
            lines.append(f"### {title} ({source_index})\nNot found in real output.\n")
            continue
        row = matches[0]
        lines.append(f"### {title} ({source_index})\n- original: {row['original_sentence']}\n- new BT: {row['back_translated_sentence']}\n- semantic forward/backward/min: {row['semantic_forward']} / {row['semantic_backward']} / {row['semantic_min']}\n- candidate NLI: {row['candidate_nli_predicted_label']} (gold probability {row['candidate_nli_gold_probability']})\n- accepted: {row['accepted']}\n- reasons: {row['reasons']}\n")
    lines.extend(["\n## Limitations\n", "This report uses the real rerun outputs and the frozen source sample. The old offline simulation used the previous calibration translation/verifier scores and cannot observe generation termination; differences are therefore not expected to be zero. No full SNLI run was started.\n"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("data/nli/calibration")
    parser.add_argument("--accepted", type=Path, default=base / "snli_calibration_800_v2.accepted.jsonl")
    parser.add_argument("--rejected", type=Path, default=base / "snli_calibration_800_v2.rejected.jsonl")
    parser.add_argument("--sources", type=Path, default=base / "snli_calibration_800_sources.jsonl")
    parser.add_argument("--offline", type=Path, default=base / "policy_v2_decisions.csv")
    parser.add_argument("--output-dir", type=Path, default=base)
    args = parser.parse_args(argv)
    records = read_jsonl(args.accepted) + read_jsonl(args.rejected)
    rows = build_audit(records, args.output_dir / "snli_calibration_800_v2.audit.csv")
    mapping = build_mapping(args.sources, args.output_dir / "v2_source_index_mapping.csv")
    build_generation_audit(rows, args.output_dir / "v2_generation_truncated.csv")
    comparison, categories = compare(rows, args.offline, mapping, args.output_dir / "v2_real_vs_offline_decisions.csv")
    write_report(rows, comparison, categories, mapping, args.output_dir / "V2_REAL_VS_OFFLINE_REPORT.md")
    print(json.dumps({"source_total": len(mapping), "candidate_total": len(rows),
                      "translation_performed": sum(_bool(row["translation_performed"]) for row in rows),
                      "accepted": sum(_bool(row["accepted"]) for row in rows),
                      "rejected": sum(not _bool(row["accepted"]) for row in rows),
                      "generation_truncated": sum(_bool(row["generation_truncated"]) for row in rows),
                      "different_decisions": sum(row["decision_change"] != "MATCH" for row in comparison)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
