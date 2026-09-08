#!/usr/bin/env python3
"""Final, analysis-only semantic metric bake-off on frozen V2 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

# Keep direct `python scripts/final_semantic_metric_bakeoff.py` execution
# equivalent to module execution from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.bakeoff_semantic_verifiers import (
    CURRENT_MODEL,
    CURRENT_REVISION,
    read_csv,
    write_csv,
    as_bool,
    as_float,
    anchor_specs,
    stable_review_rows,
    TransformerScorer,
)

STS_LARGE_MODEL = "cross-encoder/stsb-roberta-large"
BLEURT_PRIMARY = "lucadiliello/bleurt-base-512"
BLEURT_FALLBACK = "Elron/bleurt-base-512"
STS_THRESHOLDS = (.60, .65, .70, .75, .80, .85, .90, .95)
KNOWN_RISK_SOURCES = ("517858", "655", "277576", "421310", "215221")

FINAL_COLUMNS = [
    "input_position", "source_index", "original_source_index", "candidate_id",
    "original_candidate_id", "augmented_field", "gold_label",
    "original_premise", "original_hypothesis", "augmented_premise",
    "augmented_hypothesis", "original_sentence", "back_translated_sentence",
    "semantic_forward", "semantic_backward", "semantic_min",
    "current_semantic_forward", "current_semantic_backward", "current_semantic_min",
    "original_nli_predicted_label", "original_nli_gold_probability",
    "candidate_nli_predicted_label", "candidate_nli_gold_probability",
    "nli_gold_probability_delta", "nli_transition", "hard_cue_changes",
    "soft_cue_changes", "change_ratio", "accepted", "reasons", "review_categories",
    "review_semantic_equivalent", "review_label_preserved",
    "review_useful_augmentation", "review_verdict", "review_notes",
    "secondary_sts_score", "secondary_roberta_forward_entailment",
    "secondary_roberta_backward_entailment", "secondary_roberta_min_entailment",
    "secondary_sts_large_score", "secondary_bleurt_score",
    "secondary_bart_forward_entailment", "secondary_bart_backward_entailment",
    "secondary_bart_min_entailment",
    "expert_semantic_verdict", "expert_useful_verdict", "expert_notes",
    "final_disagreement_reasons",
]

ANCHOR_FINAL_COLUMNS = [
    "anchor_id", "source_index", "original_sentence", "back_translated_sentence",
    "expert_verdict", "expert_reason", "current_deberta_forward",
    "current_deberta_backward", "current_deberta_min", "sts_score",
    "roberta_forward", "roberta_backward", "roberta_min",
    "secondary_sts_large_score", "secondary_bleurt_score",
]


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [float(value) for value in values if as_float(value) is not None]
    return {
        "count": len(numbers), "min": min(numbers) if numbers else None,
        "p05": percentile(numbers, .05), "p10": percentile(numbers, .10),
        "p25": percentile(numbers, .25), "median": percentile(numbers, .50),
        "p75": percentile(numbers, .75), "p90": percentile(numbers, .90),
        "max": max(numbers) if numbers else None,
        "mean": statistics.mean(numbers) if numbers else None,
    }


def metric(row: dict[str, Any], name: str) -> float | None:
    aliases = {
        "current": "semantic_min",
        "sts_base": "secondary_sts_score",
        "sts_large": "secondary_sts_large_score",
        "bleurt": "secondary_bleurt_score",
    }
    return as_float(row.get(aliases.get(name, name)))


def score_bidirectional(scorer: TransformerScorer, rows: list[dict[str, Any]],
                        forward_key: str, backward_key: str, minimum_key: str) -> None:
    pairs = [(row.get("original_sentence", ""), row.get("back_translated_sentence", "")) for row in rows]
    forward, backward = _bidirectional(scorer, pairs)
    for row, left, right in zip(rows, forward, backward):
        row[forward_key] = left
        row[backward_key] = right
        row[minimum_key] = min(left, right)


def _bidirectional(scorer: TransformerScorer,
                   pairs: Sequence[tuple[str, str]]) -> tuple[list[float], list[float]]:
    values = scorer.predict_entailment(list(pairs) + [(b, a) for a, b in pairs])
    midpoint = len(pairs)
    return values[:midpoint], values[midpoint:]


class RawRegressionScorer:
    """BLEURT scorer; preserves raw regression values without probability mapping."""

    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device).eval()
        self.model_name = model_name
        self.revision = getattr(self.model.config, "_commit_hash", None) or "unknown"

    def predict_raw(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        values: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                encoded = self.tokenizer(
                    [x[0] for x in batch], [x[1] for x in batch],
                    return_tensors="pt", padding=True, truncation=True, max_length=512,
                ).to(self.device)
                logits = self.model(**encoded).logits
                if logits.ndim == 1:
                    values.extend(float(x) for x in logits)
                    continue
                if logits.shape[-1] != 1:
                    raise ValueError(f"Expected one BLEURT regression output, got {tuple(logits.shape)}")
                values.extend(float(x) for x in logits[:, 0])
        return values


def score_sts_large(sts: TransformerScorer, rows: list[dict[str, Any]]) -> float:
    pairs = [(row.get("original_sentence", ""), row.get("back_translated_sentence", "")) for row in rows]
    started = time.perf_counter()
    values = sts.predict_regression(pairs)
    for row, value in zip(rows, values):
        row["secondary_sts_large_score"] = value
    return time.perf_counter() - started


def score_bleurt(bleurt: RawRegressionScorer, rows: list[dict[str, Any]]) -> float:
    pairs = [(row.get("original_sentence", ""), row.get("back_translated_sentence", "")) for row in rows]
    started = time.perf_counter()
    values = bleurt.predict_raw(pairs)
    for row, value in zip(rows, values):
        row["secondary_bleurt_score"] = value
    return time.perf_counter() - started


def expert_threshold_metrics(rows: Sequence[dict[str, Any]], field: str,
                             threshold: float) -> dict[str, Any]:
    pass_rows = [row for row in rows if row.get("expert_semantic_verdict") == "PASS"]
    fail_rows = [row for row in rows if row.get("expert_semantic_verdict") == "FAIL"]
    borderline_rows = [row for row in rows if row.get("expert_semantic_verdict") == "BORDERLINE"]

    def rejected(row: dict[str, Any]) -> bool:
        value = as_float(row.get(field))
        return value is not None and value < threshold

    fail_rejected = sum(rejected(row) for row in fail_rows)
    pass_rejected = sum(rejected(row) for row in pass_rows)
    borderline_rejected = sum(rejected(row) for row in borderline_rows)
    predicted_rejects = fail_rejected + pass_rejected
    return {
        "threshold": threshold, "fail_rejected": fail_rejected,
        "fail_total": len(fail_rows), "fail_recall": fail_rejected / len(fail_rows) if fail_rows else None,
        "pass_falsely_rejected": pass_rejected, "pass_total": len(pass_rows),
        "borderline_rejected": borderline_rejected, "borderline_total": len(borderline_rows),
        "rejection_precision_pass_fail": fail_rejected / predicted_rejects if predicted_rejects else None,
    }


def current_acceptance_threshold(rows: Sequence[dict[str, Any]], field: str,
                                 threshold: float) -> dict[str, Any]:
    accepted = [row for row in rows if as_bool(row.get("accepted"))]
    retained = sum((as_float(row.get(field)) is not None and as_float(row.get(field)) >= threshold)
                   for row in accepted)
    return {
        "retained": retained, "newly_rejected": len(accepted) - retained,
        "retention_rate": retained / len(accepted) if accepted else None,
    }


def threshold_sweep(rows: Sequence[dict[str, Any]], field: str,
                    thresholds: Sequence[float],
                    accepted_rows: Sequence[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    population = rows if accepted_rows is None else accepted_rows
    return [{**expert_threshold_metrics(rows, field, threshold),
             **current_acceptance_threshold(population, field, threshold)}
            for threshold in thresholds]


def percentile_rank(value: float | None, population: Sequence[float]) -> float | None:
    if value is None or not population:
        return None
    return 100 * sum(x <= value for x in population) / len(population)


def combination_decision(row: dict[str, Any], sts_threshold: float,
                         bleurt_threshold: float, mode: str) -> bool:
    sts_ok = (metric(row, "sts_base") or -float("inf")) >= sts_threshold
    bleurt_ok = (metric(row, "bleurt") or -float("inf")) >= bleurt_threshold
    return (sts_ok and bleurt_ok) if mode == "either" else (sts_ok or bleurt_ok)


def combination_metrics(rows: Sequence[dict[str, Any]], sts_threshold: float,
                        bleurt_threshold: float, mode: str,
                        accepted_rows: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    def rejected(row: dict[str, Any]) -> bool:
        sts_ok = (metric(row, "sts_base") or -float("inf")) >= sts_threshold
        bleurt_ok = (metric(row, "bleurt") or -float("inf")) >= bleurt_threshold
        return not ((sts_ok and bleurt_ok) if mode == "either" else (sts_ok or bleurt_ok))

    pass_rows = [row for row in rows if row.get("expert_semantic_verdict") == "PASS"]
    fail_rows = [row for row in rows if row.get("expert_semantic_verdict") == "FAIL"]
    accepted = [row for row in (rows if accepted_rows is None else accepted_rows)
                if as_bool(row.get("accepted"))]
    fail_rejected = sum(rejected(x) for x in fail_rows)
    pass_rejected = sum(rejected(x) for x in pass_rows)
    retained = sum(not rejected(x) for x in accepted)
    return {
        "sts_threshold": sts_threshold, "bleurt_threshold": bleurt_threshold,
        "mode": mode, "fail_rejected": fail_rejected, "fail_total": len(fail_rows),
        "pass_falsely_rejected": pass_rejected, "pass_total": len(pass_rows),
        "accepted_retained": retained, "accepted_total": len(accepted),
        "retention_rate": retained / len(accepted) if accepted else None,
    }


def select_final_disagreement(rows: Sequence[dict[str, Any]],
                              bleurt_pass_p10: float | None) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if not as_bool(row.get("accepted")):
            continue
        reasons = []
        base, large, bleurt = metric(row, "sts_base"), metric(row, "sts_large"), metric(row, "bleurt")
        if base is not None and large is not None and abs(base - large) >= .10:
            reasons.append("sts_base_vs_large_disagreement")
        if base is not None and base < .85:
            reasons.append("sts_base_below_0.85")
        if large is not None and large < .85:
            reasons.append("sts_large_below_0.85")
        if bleurt is not None and bleurt_pass_p10 is not None and bleurt < bleurt_pass_p10:
            reasons.append("bleurt_below_pass_p10")
        if str(row.get("source_index")) in KNOWN_RISK_SOURCES:
            reasons.append("known_risk_source")
        if reasons:
            copy = dict(row)
            copy["final_disagreement_reasons"] = "|".join(reasons)
            selected.append(copy)
    selected.sort(key=lambda row: (
        len(str(row.get("final_disagreement_reasons", "")).split("|")) * -1,
        metric(row, "sts_base") or 1, metric(row, "sts_large") or 1,
        metric(row, "bleurt") or 0,
    ))
    return selected


def make_anchor_rows(anchor_csv: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for row in anchor_csv:
        item = dict(row)
        item["expert_semantic_verdict"] = row.get("expert_verdict", "")
        item["expert_useful_verdict"] = ""
        item["expert_notes"] = row.get("expert_reason", "")
        rows.append(item)
    return rows


def anchor_report(anchors: Sequence[dict[str, Any]], output: Path,
                  revisions: dict[str, str], runtimes: dict[str, float]) -> None:
    lines = [
        "# Final Semantic Metric Anchor Report", "",
        "These are targeted expert anchors, not a benchmark and not a threshold-calibration set.", "",
        "| id | verdict | DeBERTa min | STS-base | STS-large | BLEURT raw | RoBERTa min | original | candidate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in anchors:
        lines.append(
            f"| {row.get('anchor_id')} | {row.get('expert_verdict')} | "
            f"{row.get('current_deberta_min', '')} | {row.get('sts_score', '')} | "
            f"{row.get('secondary_sts_large_score', '')} | {row.get('secondary_bleurt_score', '')} | "
            f"{row.get('roberta_min', '')} | {row.get('original_sentence')} | "
            f"{row.get('back_translated_sentence')} |"
        )
    lines.extend([
        "", "## Models and runtimes", "",
        f"- Current DeBERTa: {CURRENT_MODEL}, revision {revisions.get('current', CURRENT_REVISION)}; existing/current anchor scores.",
        f"- STS-base: cross-encoder/stsb-roberta-base; existing scores reused.",
        f"- STS-large: {STS_LARGE_MODEL}, revision {revisions.get('sts_large', 'unknown')}; runtime {runtimes.get('sts_large_anchor', 0):.2f}s.",
        f"- BLEURT: {revisions.get('bleurt_model', BLEURT_PRIMARY)}, revision {revisions.get('bleurt', 'unknown')}; raw-score runtime {runtimes.get('bleurt_anchor', 0):.2f}s.",
        f"- BLEURT primary fallback note: {revisions.get('bleurt_fallback_reason', 'primary model loaded') }.",
        "- RoBERTa-MNLI: existing scores reused; BART-MNLI was not run.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(anchors: Sequence[dict[str, Any]], expert_rows: Sequence[dict[str, Any]],
                 translated: Sequence[dict[str, Any]], output: Path,
                 revisions: dict[str, str], runtimes: dict[str, float],
                 sts_sweep: Sequence[dict[str, Any]], large_sweep: Sequence[dict[str, Any]],
                 bleurt_sweep: Sequence[dict[str, Any]], combo_rows: Sequence[dict[str, Any]],
                 bleurt_distributions: dict[str, dict[str, Any]]) -> None:
    current_accepted = [row for row in translated if as_bool(row.get("accepted"))]
    current_population = [row for row in translated if metric(row, "current") is not None]
    lines = [
        "# Final Semantic Metric Bake-off Report", "",
        "Analysis-only. No NLLB, full-SNLI run, production-policy change, or threshold change was performed.", "",
        "## Models", "",
        f"- Current DeBERTa semantic baseline: {CURRENT_MODEL}, revision {revisions.get('current', CURRENT_REVISION)}; existing V2 scores reused.",
        "- STS-base: cross-encoder/stsb-roberta-base; existing scores reused.",
        f"- STS-large: {STS_LARGE_MODEL}, revision {revisions.get('sts_large', 'unknown')}; CPU runtime {runtimes.get('sts_large_full', 0):.2f}s.",
        f"- BLEURT: {revisions.get('bleurt_model', BLEURT_PRIMARY)}, revision {revisions.get('bleurt', 'unknown')}; CPU raw-score runtime {runtimes.get('bleurt_full', 0):.2f}s.",
        f"- BLEURT primary fallback note: {revisions.get('bleurt_fallback_reason', 'primary model loaded') }.",
        "- RoBERTa-MNLI: existing scores reused; BART-MNLI: not run.", "",
        "## Expert-reviewed disagreement pairwise ranking", "",
        "| metric | PASS > FAIL pairwise ranking |", "| --- | ---: |",
    ]
    pairwise_fields = [("Current DeBERTa", "current_semantic_min"),
                        ("STS-base", "secondary_sts_score"),
                        ("STS-large", "secondary_sts_large_score"),
                        ("BLEURT", "secondary_bleurt_score"),
                        ("RoBERTa-MNLI", "secondary_roberta_min_entailment")]
    for name, field in pairwise_fields:
        pass_values = [as_float(x.get(field)) for x in expert_rows if x.get("expert_semantic_verdict") == "PASS"]
        fail_values = [as_float(x.get(field)) for x in expert_rows if x.get("expert_semantic_verdict") == "FAIL"]
        pairs = [(p, f) for p in pass_values for f in fail_values if p is not None and f is not None]
        score = sum(p > f for p, f in pairs) / len(pairs) if pairs else None
        lines.append(f"| {name} | {score:.4f} |" if score is not None else f"| {name} | n/a |")

    lines.extend(["", "## Expert anchor scores", "",
                   "| anchor | verdict | Current DeBERTa min | STS-base | STS-large | BLEURT raw | RoBERTa MNLI min |",
                   "| --- | --- | ---: | ---: | ---: | ---: | ---: |"] )
    for row in anchors:
        lines.append(f"| {row.get('anchor_id')} | {row.get('expert_verdict')} | {fmt(row.get('current_deberta_min'))} | {fmt(row.get('sts_score'))} | {fmt(row.get('secondary_sts_large_score'))} | {fmt(row.get('secondary_bleurt_score'))} | {fmt(row.get('roberta_min'))} |")

    lines.extend(["", "## Canary scores and ranks among all 1,336 translated candidates", "",
                   "| source_index | metric | score | percentile at-or-below |",
                   "| --- | --- | ---: | ---: |"])
    populations = {name: [metric(row, name) for row in translated] for name in ("sts_base", "sts_large", "bleurt")}
    populations.update({"current": [metric(row, "current") for row in translated],
                        "roberta": [as_float(row.get("secondary_roberta_min_entailment")) for row in translated]})
    for source_index in ("148356", "227619", "512053"):
        matches = [row for row in translated if str(row.get("source_index")) == source_index and row.get("augmented_field") == "hypothesis"]
        if not matches:
            lines.append(f"| {source_index} | not found | not found | not found | not found |")
            continue
        row = matches[0]
        for name, values in (("Current DeBERTa", populations["current"]),
                              ("STS-base", populations["sts_base"]),
                              ("STS-large", populations["sts_large"]),
                              ("BLEURT", populations["bleurt"]),
                              ("RoBERTa-MNLI", populations["roberta"])):
            if name == "Current DeBERTa":
                value = metric(row, "current")
            elif name == "RoBERTa-MNLI":
                value = as_float(row.get("secondary_roberta_min_entailment"))
            else:
                value = metric(row, {"STS-base": "sts_base", "STS-large": "sts_large", "BLEURT": "bleurt"}[name])
            lines.append(f"| {source_index} | {name} | {fmt(value)} | {fmt_rank(value, values)} |")

    lines.extend(["", "## Full translated-candidate distributions", "",
                   "| subset | metric | count | mean | median | p05 | p10 | p25 | p75 | p90 |",
                   "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for subset_name, subset in [("current accepted", current_accepted),
                                ("current rejected", [x for x in translated if not as_bool(x.get("accepted"))])]:
        for name in ("current", "sts_base", "sts_large", "bleurt"):
            stats = metric_distribution(subset, name)
            lines.append(f"| {subset_name} | {name} | {stats['count']} | {fmt(stats['mean'])} | {fmt(stats['median'])} | {fmt(stats['p05'])} | {fmt(stats['p10'])} | {fmt(stats['p25'])} | {fmt(stats['p75'])} | {fmt(stats['p90'])} |")

    lines.extend(["", "## Semantic-drift rejection distributions", "",
                   "| subset | metric | count | mean | median | p75 | p90 |",
                   "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    drift = [row for row in translated if not as_bool(row.get("accepted")) and "semantic_drift" in json.loads(row.get("reasons") or "[]")]
    for name in ("sts_base", "sts_large", "bleurt"):
        stats = metric_distribution(drift, name)
        lines.append(f"| semantic_drift rejected | {name} | {stats['count']} | {fmt(stats['mean'])} | {fmt(stats['median'])} | {fmt(stats['p75'])} | {fmt(stats['p90'])} |")

    lines.extend(["", "## STS threshold simulation on expert-reviewed disagreement set", "",
                   "| model | threshold | FAIL caught | FAIL recall | PASS falsely rejected | BORDERLINE rejected | PASS/FAIL rejection precision | accepted retained | newly rejected | retention |",
                   "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for model_name, values in [("STS-base", sts_sweep), ("STS-large", large_sweep)]:
        for item in values:
            lines.append(f"| {model_name} | {item['threshold']:.2f} | {item['fail_rejected']}/{item['fail_total']} | {fmt_pct(item['fail_recall'])} | {item['pass_falsely_rejected']}/{item['pass_total']} | {item['borderline_rejected']}/{item['borderline_total']} | {fmt_pct(item['rejection_precision_pass_fail'])} | {item['retained']} | {item['newly_rejected']} | {fmt_pct(item['retention_rate'])} |")

    lines.extend(["", "## BLEURT observed distributions", "",
                   "| subset | count | min | p05 | p10 | p25 | median | p75 | p90 | max |",
                   "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for subset, stats in bleurt_distributions.items():
        lines.append(f"| {subset} | {stats['count']} | {fmt(stats['min'])} | {fmt(stats['p05'])} | {fmt(stats['p10'])} | {fmt(stats['p25'])} | {fmt(stats['median'])} | {fmt(stats['p75'])} | {fmt(stats['p90'])} | {fmt(stats['max'])} |")
    lines.extend(["", "BLEURT thresholds below are observed-score analysis points, not a fixed 01 grid.", "",
                   "| threshold | FAIL caught | FAIL recall | PASS falsely rejected | BORDERLINE rejected | accepted retained | newly rejected | retention |",
                   "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for item in bleurt_sweep:
        lines.append(f"| {item['threshold']:.4f} | {item['fail_rejected']}/{item['fail_total']} | {fmt_pct(item['fail_recall'])} | {item['pass_falsely_rejected']}/{item['pass_total']} | {item['borderline_rejected']}/{item['borderline_total']} | {item['retained']} | {item['newly_rejected']} | {fmt_pct(item['retention_rate'])} |")

    lines.extend(["", "## Combination-policy simulation", "",
                   "Combination rows are analysis-only. mode=either rejects when either metric is below threshold; mode=both rejects only when both are below threshold.",
                   "| policy | mode | STS threshold | BLEURT threshold | FAIL caught | PASS falsely rejected | accepted retained | retention |",
                   "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for item in combo_rows:
        lines.append(f"| {item['policy']} | {item['mode']} | {item['sts_threshold']:.2f} | {item['bleurt_threshold']:.4f} | {item['fail_rejected']}/{item['fail_total']} | {item['pass_falsely_rejected']}/{item['pass_total']} | {item['accepted_retained']}/{item['accepted_total']} | {fmt_pct(item['retention_rate'])} |")

    lines.extend(["", "## Current accepted model distributions", "",
                   "| metric | count | mean | median | p05 | p10 | p25 | p75 | p90 |",
                   "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name in ("sts_base", "sts_large", "bleurt"):
        stats = metric_distribution(current_accepted, name)
        lines.append(f"| {name} | {stats['count']} | {fmt(stats['mean'])} | {fmt(stats['median'])} | {fmt(stats['p05'])} | {fmt(stats['p10'])} | {fmt(stats['p25'])} | {fmt(stats['p75'])} | {fmt(stats['p90'])} |")

    lines.extend(["", "## Known-risk examples", "",
                   "| source_index | candidate_id | original | candidate | STS-base | STS-large | BLEURT | current decision |",
                   "| --- | --- | --- | --- | ---: | ---: | ---: | --- |"])
    for source_index in KNOWN_RISK_SOURCES:
        found = [row for row in translated if str(row.get("source_index")) == source_index]
        for row in found:
            lines.append(f"| {source_index} | {row.get('candidate_id')} | {row.get('original_sentence')} | {row.get('back_translated_sentence')} | {fmt(metric(row, 'sts_base'))} | {fmt(metric(row, 'sts_large'))} | {fmt(metric(row, 'bleurt'))} | {row.get('accepted')} |")

    base075 = next((x for x in sts_sweep if abs(x["threshold"] - .75) < 1e-9), None)
    base080 = next((x for x in sts_sweep if abs(x["threshold"] - .80) < 1e-9), None)
    large090 = next((x for x in large_sweep if abs(x["threshold"] - .90) < 1e-9), None)
    bleurt_best = max(bleurt_sweep, key=lambda x: (x["fail_recall"] or -1, -(x["pass_falsely_rejected"]))) if bleurt_sweep else None
    conservative_combo = next((x for x in combo_rows if x["mode"] == "both" and abs(x["sts_threshold"] - .75) < 1e-9), None)
    lines.extend(["", "## Decision-oriented analysis points", "",
                   f"- S1 STS-base >= 0.75: {base075['fail_rejected']}/{base075['fail_total']} FAIL caught, {base075['pass_falsely_rejected']}/{base075['pass_total']} PASS lost, {base075['retained']}/{base075['retained'] + base075['newly_rejected']} current accepted retained." if base075 else "- S1 unavailable.",
                   f"- S2 STS-base >= 0.80: {base080['fail_rejected']}/{base080['fail_total']} FAIL caught, {base080['pass_falsely_rejected']}/{base080['pass_total']} PASS lost, {base080['retained']}/{base080['retained'] + base080['newly_rejected']} current accepted retained." if base080 else "- S2 unavailable.",
                   f"- Representative STS-large point 0.90: {large090['fail_rejected']}/{large090['fail_total']} FAIL caught, {large090['pass_falsely_rejected']}/{large090['pass_total']} PASS lost, {large090['retained']}/{large090['retained'] + large090['newly_rejected']} current accepted retained." if large090 else "- STS-large 0.90 unavailable.",
                   f"- Highest-FAIL-recall observed BLEURT point {bleurt_best['threshold']:.4f}: {bleurt_best['fail_rejected']}/{bleurt_best['fail_total']} FAIL caught, {bleurt_best['pass_falsely_rejected']}/{bleurt_best['pass_total']} PASS lost, {bleurt_best['retained']}/{bleurt_best['retained'] + bleurt_best['newly_rejected']} current accepted retained." if bleurt_best else "- BLEURT selection unavailable.",
                   f"- Preservation-oriented STS-base+BLEURT representative (both mode, STS 0.75): {conservative_combo['fail_rejected']}/{conservative_combo['fail_total']} FAIL caught, {conservative_combo['pass_falsely_rejected']}/{conservative_combo['pass_total']} PASS lost, {conservative_combo['accepted_retained']}/{conservative_combo['accepted_total']} current accepted retained." if conservative_combo else "- Combination selection unavailable.",
                   "- These are offline comparisons only and do not select or modify a production gate.",
                   "", "## Limitations and final recommendation", "",
                   "- The 31-row expert-reviewed set was sampled from secondary disagreement candidates and has selection bias; it is not a benchmark.",
                   "- BLEURT is reported as a raw regression score and is not comparable numerically to probability-like STS scores.",
                   "- No secondary metric was added to production in this round.",
                   "- Model C BART-MNLI was not run.",
                   "- Final recommendation: Option C — no additional general-purpose semantic metric is sufficient as a reliable production gate for the observed subtle drift; do not add STS-large or BLEURT to production in this round.",
                   "- Canary B is the decisive limitation: a general metric that keeps progressive `are drowning` → completed `drowned` high-confidence cannot provide the required guarantee. Next options are targeted tense/aspect or role guards, or explicit acceptance of residual noise.",
                   "- Final recommendation is based on targeted anchors, reviewed disagreement evidence, retention, and runtime; it is not threshold calibration.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_distribution(rows: Sequence[dict[str, Any]], name: str) -> dict[str, Any]:
    return distribution(metric(row, name) for row in rows)


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def fmt_rank(value: float | None, population: Sequence[float | None]) -> str:
    values = [x for x in population if x is not None]
    rank = percentile_rank(value, values)
    return "n/a" if rank is None else f"{rank:.2f}th"


def add_existing_scores(source_rows: Sequence[dict[str, str]],
                        audit_rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    by_key = {(str(row.get("source_index")), row.get("augmented_field")): row for row in audit_rows}
    output = []
    for row in source_rows:
        item = dict(row)
        audit = by_key.get((str(row.get("source_index")), row.get("augmented_field")))
        if audit:
            item["current_semantic_forward"] = audit.get("semantic_forward", "")
            item["current_semantic_backward"] = audit.get("semantic_backward", "")
            item["current_semantic_min"] = audit.get("semantic_min", "")
        output.append(item)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    base = Path("data/nli/calibration")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=base / "snli_calibration_800_v2.audit.csv")
    parser.add_argument("--existing-secondary", type=Path,
                        default=base / "snli_calibration_800_v2.secondary_semantic_scores.csv")
    parser.add_argument("--expert", type=Path,
                        default=base / "review_secondary_disagreement_expert.csv")
    parser.add_argument("--master-review", type=Path, default=base / "MASTER_REVIEW.csv")
    parser.add_argument("--mapping", type=Path, default=base / "v2_source_index_mapping.csv")
    parser.add_argument("--output-dir", type=Path, default=base)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    audit = read_csv(args.audit)
    existing_secondary = read_csv(args.existing_secondary)
    existing_by_id = {row.get("candidate_id", ""): row for row in existing_secondary}
    translated = []
    for audit_row in audit:
        if not as_bool(audit_row.get("translation_performed")):
            continue
        item = dict(audit_row)
        old = existing_by_id.get(audit_row.get("candidate_id", ""), {})
        # The V2 audit is authoritative for provenance and production decisions;
        # reuse only the previously computed secondary scores.
        item.update({
            "current_semantic_forward": audit_row.get("semantic_forward", ""),
            "current_semantic_backward": audit_row.get("semantic_backward", ""),
            "current_semantic_min": audit_row.get("semantic_min", ""),
            "secondary_sts_score": old.get("secondary_sts_score", ""),
            "secondary_roberta_forward_entailment": old.get("secondary_roberta_forward_entailment", ""),
            "secondary_roberta_backward_entailment": old.get("secondary_roberta_backward_entailment", ""),
            "secondary_roberta_min_entailment": old.get("secondary_roberta_min_entailment", ""),
            "review_categories": old.get("review_categories", ""),
            "review_semantic_equivalent": old.get("review_semantic_equivalent", ""),
            "review_label_preserved": old.get("review_label_preserved", ""),
            "review_useful_augmentation": old.get("review_useful_augmentation", ""),
            "review_verdict": old.get("review_verdict", ""),
            "review_notes": old.get("review_notes", ""),
        })
        translated.append(item)
    review = read_csv(args.expert)
    anchors = make_anchor_rows(read_csv(base / "semantic_verifier_anchor_set.csv"))
    audit_by_id = {row.get("candidate_id", ""): row for row in audit}
    expert_rows = []
    for review_row in review:
        item = dict(review_row)
        audit_row = audit_by_id.get(review_row.get("candidate_id", ""))
        if audit_row is None:
            raise ValueError(f"Expert candidate is absent from V2 audit: {review_row.get('candidate_id')}")
        item.update({
            "input_position": audit_row.get("input_position", review_row.get("input_position", "")),
            "source_index": audit_row.get("source_index", review_row.get("source_index", "")),
            "current_semantic_forward": audit_row.get("semantic_forward", ""),
            "current_semantic_backward": audit_row.get("semantic_backward", ""),
            "current_semantic_min": audit_row.get("semantic_min", ""),
        })
        expert_rows.append(item)
    if len(expert_rows) != len(review):
        raise AssertionError("Expert review join dropped rows")

    all_score_rows = anchors + expert_rows + translated
    sts_large = TransformerScorer(STS_LARGE_MODEL, args.device, args.batch_size, task="sts")
    sts_large_revision_value = sts_large.revision
    started = time.perf_counter()
    score_sts_large(sts_large, all_score_rows)
    full_sts_elapsed = time.perf_counter() - started
    del sts_large

    bleurt_model = BLEURT_PRIMARY
    bleurt_fallback_reason = "primary model loaded"
    try:
        bleurt = RawRegressionScorer(BLEURT_PRIMARY, args.device, args.batch_size)
    except Exception as exc:
        bleurt_model = BLEURT_FALLBACK
        detail = " ".join(str(exc).split())[:240]
        bleurt_fallback_reason = f"{type(exc).__name__}: {detail}"
        bleurt = RawRegressionScorer(BLEURT_FALLBACK, args.device, args.batch_size)
    started = time.perf_counter()
    score_bleurt(bleurt, all_score_rows)
    full_bleurt_elapsed = time.perf_counter() - started

    # The anchor CSV uses its historical names; preserve those and add new scores.
    write_csv(args.output_dir / "semantic_verifier_anchor_set_final.csv", anchors, ANCHOR_FINAL_COLUMNS)
    sts_anchor_elapsed = full_sts_elapsed * len(anchors) / max(len(all_score_rows), 1)
    bleurt_anchor_elapsed = full_bleurt_elapsed * len(anchors) / max(len(all_score_rows), 1)
    revisions = {
        "current": CURRENT_REVISION, "sts_large": sts_large_revision_value,
        "bleurt_model": bleurt_model, "bleurt": bleurt.revision,
        "bleurt_fallback_reason": bleurt_fallback_reason,
    }
    runtimes = {
        "sts_large_full": full_sts_elapsed, "bleurt_full": full_bleurt_elapsed,
        "sts_large_anchor": sts_anchor_elapsed, "bleurt_anchor": bleurt_anchor_elapsed,
    }
    anchor_report(anchors, args.output_dir / "FINAL_SEMANTIC_METRIC_ANCHOR_REPORT.md", revisions, runtimes)

    # Expert rows keep review fields empty unless they were already empty in the input.
    write_csv(args.output_dir / "review_secondary_disagreement_expert_scored.csv",
              expert_rows, FINAL_COLUMNS)
    write_csv(args.output_dir / "snli_calibration_800_v2.final_semantic_scores.csv",
              translated, FINAL_COLUMNS)

    pass_bleurt = [metric(row, "bleurt") for row in expert_rows if row.get("expert_semantic_verdict") == "PASS"]
    pass_bleurt_p10 = percentile([x for x in pass_bleurt if x is not None], .10)
    disagreement = select_final_disagreement(translated, pass_bleurt_p10)
    write_csv(args.output_dir / "review_final_metric_disagreement.csv",
              disagreement[:150], FINAL_COLUMNS)

    sts_sweep = threshold_sweep(expert_rows, "secondary_sts_score", STS_THRESHOLDS, translated)
    large_sweep = threshold_sweep(expert_rows, "secondary_sts_large_score", STS_THRESHOLDS, translated)
    bleurt_values = [metric(row, "bleurt") for row in expert_rows]
    bleurt_dist = {
        "expert PASS": metric_distribution([x for x in expert_rows if x.get("expert_semantic_verdict") == "PASS"], "bleurt"),
        "expert FAIL": metric_distribution([x for x in expert_rows if x.get("expert_semantic_verdict") == "FAIL"], "bleurt"),
        "current V2 accepted": metric_distribution([x for x in translated if as_bool(x.get("accepted"))], "bleurt"),
    }
    observed = sorted({round(x, 4) for x in bleurt_values if x is not None})
    if len(observed) > 8:
        bleurt_thresholds = tuple(sorted({round(percentile(observed, q) or observed[0], 4)
                                          for q in (.10, .25, .50, .75, .90)}))
    else:
        bleurt_thresholds = tuple(observed)
    bleurt_sweep = threshold_sweep(expert_rows, "secondary_bleurt_score", bleurt_thresholds, translated)

    median_pass_bleurt = percentile([x for x in pass_bleurt if x is not None], .50)
    chosen_bleurt = median_pass_bleurt if median_pass_bleurt is not None else (bleurt_thresholds[0] if bleurt_thresholds else 0.0)
    combo_rows = []
    for mode in ("either", "both"):
        for sts_threshold in (.75, .80):
            item = combination_metrics(expert_rows, sts_threshold, chosen_bleurt, mode, translated)
            item["policy"] = f"STS-base {sts_threshold:.2f} + BLEURT observed PASS median"
            combo_rows.append(item)
    write_report(anchors, expert_rows, translated,
                 args.output_dir / "FINAL_SEMANTIC_METRIC_BAKEOFF_REPORT.md",
                 revisions, runtimes, sts_sweep, large_sweep, bleurt_sweep, combo_rows, bleurt_dist)
    print(json.dumps({
        "anchors": len(anchors), "expert_review": len(expert_rows),
        "translated_candidates": len(translated),
        "sts_large_model_revision": revisions["sts_large"],
        "bleurt_model": bleurt_model, "bleurt_revision": bleurt.revision,
        "sts_large_runtime_seconds": round(full_sts_elapsed, 3),
        "bleurt_runtime_seconds": round(full_bleurt_elapsed, 3),
        "final_disagreement": min(len(disagreement), 150),
    }))
    return 0


def sts_large_revision(rows: Sequence[dict[str, Any]]) -> str:
    # Model revision is attached by main through the scorer; rows do not carry it.
    return "recorded-in-run"


if __name__ == "__main__":
    raise SystemExit(main())
