#!/usr/bin/env python3
"""Build a stratified, model-blind accepted-candidate precision audit package."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Iterable

LABEL_NAMES = {"0": "entailment", "1": "neutral", "2": "contradiction"}
LABEL_IDS = {name: index for index, name in enumerate(("entailment", "neutral", "contradiction"))}
STRATA = tuple((label, field) for label in LABEL_IDS for field in ("premise", "hypothesis"))
SAMPLE_PER_STRATUM = 40
SEED = 20260908

BLIND_COLUMNS = [
    "audit_id", "gold_label", "gold_label_name", "augmented_field",
    "original_premise", "original_hypothesis", "augmented_premise",
    "augmented_hypothesis", "original_sentence", "back_translated_sentence",
    "review_semantic_equivalent", "review_label_preserved",
    "review_useful_augmentation", "review_severity", "review_error_types",
    "review_notes",
]
REVIEW_COLUMNS = BLIND_COLUMNS
KEY_COLUMNS = [
    "audit_id", "source_index", "input_position", "candidate_id", "gold_label",
    "augmented_field", "semantic_forward", "semantic_backward", "semantic_min",
    "candidate_nli_predicted_label", "candidate_nli_gold_probability",
    "source_nli_status", "original_nli_gold_probability", "change_ratio",
    "hard_cue_changes", "soft_cue_changes", "secondary_sts_score",
    "secondary_sts_large_score", "secondary_bleurt_score",
    "secondary_roberta_min_entailment", "reasons", "stratum_population_size",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def label_name(value: Any) -> str:
    text = str(value).strip().lower()
    return LABEL_NAMES.get(text, text)


def bool_value(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def source_lookup(source_path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for position, row in enumerate(read_jsonl(source_path)):
        source_index = row.get("source_index")
        if source_index is None:
            source_index = position
        result[str(source_index)] = {
            "premise": row.get("premise", ""),
            "hypothesis": row.get("hypothesis", ""),
            "label": row.get("label", ""),
        }
    return result


def population_rows(audit_path: Path, scored_path: Path | None = None) -> list[dict[str, str]]:
    rows = read_csv(audit_path)
    accepted = [
        dict(row) for row in rows
        if bool_value(row.get("accepted")) and bool_value(row.get("translation_performed"))
    ]
    if len(accepted) != 774:
        raise ValueError(
            f"Expected exactly 774 accepted translated V2 candidates, found {len(accepted)}; refusing to sample"
        )
    if scored_path is not None and scored_path.exists():
        scored = {row.get("candidate_id", ""): row for row in read_csv(scored_path)}
        for row in accepted:
            prior = scored.get(row.get("candidate_id", ""), {})
            for field in ("secondary_sts_score", "secondary_sts_large_score",
                          "secondary_bleurt_score", "secondary_roberta_min_entailment"):
                row[field] = prior.get(field, "")
    return sorted(accepted, key=lambda row: row.get("candidate_id", ""))


def construct_rows(audit_path: Path, source_path: Path, seed: int = SEED,
                   sample_per_stratum: int = SAMPLE_PER_STRATUM,
                   scored_path: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], int]]:
    population = population_rows(audit_path, scored_path)
    sources = source_lookup(source_path)
    groups: dict[tuple[str, str], list[dict[str, str]]] = {key: [] for key in STRATA}
    for row in population:
        key = (label_name(row.get("gold_label")), row.get("augmented_field", ""))
        if key not in groups:
            raise ValueError(f"Unexpected accepted stratum: {key}")
        groups[key].append(row)
    sizes = {key: len(value) for key, value in groups.items()}
    if any(size < sample_per_stratum for size in sizes.values()):
        raise ValueError(f"A stratum has fewer than {sample_per_stratum} candidates: {sizes}")

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in STRATA:
        selected.extend(rng.sample(groups[key], sample_per_stratum))
    rng.shuffle(selected)

    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        audit_id = f"A{index:04d}"
        field = row.get("augmented_field", "")
        source = sources.get(str(row.get("source_index")), {})
        original_premise = source.get("premise", "")
        original_hypothesis = source.get("hypothesis", "")
        augmented_premise = row.get("back_translated_sentence", "") if field == "premise" else original_premise
        augmented_hypothesis = row.get("back_translated_sentence", "") if field == "hypothesis" else original_hypothesis
        blind_rows.append({
            "audit_id": audit_id,
            "gold_label": LABEL_IDS[label_name(row.get("gold_label"))],
            "gold_label_name": label_name(row.get("gold_label")),
            "augmented_field": field,
            "original_premise": original_premise,
            "original_hypothesis": original_hypothesis,
            "augmented_premise": augmented_premise,
            "augmented_hypothesis": augmented_hypothesis,
            "original_sentence": row.get("original_sentence", ""),
            "back_translated_sentence": row.get("back_translated_sentence", ""),
            "review_semantic_equivalent": "",
            "review_label_preserved": "",
            "review_useful_augmentation": "",
            "review_severity": "",
            "review_error_types": "",
            "review_notes": "",
        })
        key_rows.append({
            "audit_id": audit_id,
            "source_index": row.get("source_index", ""),
            "input_position": row.get("input_position", ""),
            "candidate_id": row.get("candidate_id", ""),
            "gold_label": label_name(row.get("gold_label")),
            "augmented_field": field,
            "semantic_forward": row.get("semantic_forward", ""),
            "semantic_backward": row.get("semantic_backward", ""),
            "semantic_min": row.get("semantic_min", ""),
            "candidate_nli_predicted_label": row.get("candidate_nli_predicted_label", ""),
            "candidate_nli_gold_probability": row.get("candidate_nli_gold_probability", ""),
            "source_nli_status": row.get("source_nli_status", ""),
            "original_nli_gold_probability": row.get("original_nli_gold_probability", ""),
            "change_ratio": row.get("change_ratio", ""),
            "hard_cue_changes": row.get("hard_cue_changes", ""),
            "soft_cue_changes": row.get("soft_cue_changes", ""),
            "secondary_sts_score": row.get("secondary_sts_score", ""),
            "secondary_sts_large_score": row.get("secondary_sts_large_score", ""),
            "secondary_bleurt_score": row.get("secondary_bleurt_score", ""),
            "secondary_roberta_min_entailment": row.get("secondary_roberta_min_entailment", ""),
            "reasons": row.get("reasons", ""),
            "stratum_population_size": sizes[(label_name(row.get("gold_label")), field)],
        })
    return blind_rows, key_rows, sizes


def build_package(audit_path: Path, source_path: Path, output_dir: Path,
                  seed: int = SEED, sample_per_stratum: int = SAMPLE_PER_STRATUM,
                  scored_path: Path | None = None) -> dict[str, Any]:
    blind, key, sizes = construct_rows(audit_path, source_path, seed, sample_per_stratum, scored_path)
    if len(blind) != 240:
        raise AssertionError(f"Expected 240 blind rows, found {len(blind)}")
    write_csv(output_dir / "accepted_precision_audit_blind.csv", blind, BLIND_COLUMNS)
    write_csv(output_dir / "accepted_precision_audit_key.csv", key, KEY_COLUMNS)
    for batch_number in range(6):
        start = batch_number * 40
        write_csv(output_dir / f"accepted_precision_audit_blind_{batch_number + 1:02d}.csv",
                  blind[start:start + 40], BLIND_COLUMNS)
    return {"population": 774, "sample": len(blind), "seed": seed, "stratum_sizes": sizes,
            "blind": blind, "key": key}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("data/nli/calibration/snli_calibration_800_v2.audit.csv"))
    parser.add_argument("--sources", type=Path, default=Path("data/nli/calibration/snli_calibration_800_sources.jsonl"))
    parser.add_argument("--scored", type=Path, default=Path("data/nli/calibration/snli_calibration_800_v2.final_semantic_scores.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/nli/calibration"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = build_package(args.audit, args.sources, args.output_dir, args.seed, SAMPLE_PER_STRATUM, args.scored)
    print({"population": result["population"], "sample": result["sample"],
           "seed": result["seed"], "stratum_sizes": result["stratum_sizes"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
