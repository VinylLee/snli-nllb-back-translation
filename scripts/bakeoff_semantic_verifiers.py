#!/usr/bin/env python3
"""Analysis-only secondary semantic verifier bake-off for frozen V2 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

CURRENT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
CURRENT_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
STS_MODEL = "cross-encoder/stsb-roberta-base"
ROBERTA_MODEL = "FacebookAI/roberta-large-mnli"
LABEL_NAMES = {"0": "entailment", "1": "neutral", "2": "contradiction"}

REVIEW_COLUMNS = [
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
    "secondary_bart_forward_entailment", "secondary_bart_backward_entailment",
    "secondary_bart_min_entailment",
]
ANCHOR_COLUMNS = [
    "anchor_id", "source_index", "original_sentence", "back_translated_sentence",
    "expert_verdict", "expert_reason", "current_deberta_forward",
    "current_deberta_backward", "current_deberta_min", "sts_score",
    "roberta_forward", "roberta_backward", "roberta_min", "bart_forward",
    "bart_backward", "bart_min",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def json_value(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (pos - low)


def distribution(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [float(x) for x in values if as_float(x) is not None]
    return {
        "count": len(numbers),
        "mean": statistics.mean(numbers) if numbers else None,
        "median": percentile(numbers, .50),
        "p05": percentile(numbers, .05),
        "p10": percentile(numbers, .10),
        "p25": percentile(numbers, .25),
        "p50": percentile(numbers, .50),
        "p75": percentile(numbers, .75),
        "p90": percentile(numbers, .90),
    }


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


class TransformerScorer:
    def __init__(self, model_name: str, device: str, batch_size: int,
                 revision: str | None = None, task: str = "nli") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.task = task
        kwargs = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        self.model.to(device).eval()
        self.model_name = model_name
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision or "unknown"
        if task == "nli":
            self.label_indices = self._label_indices()

    def _label_indices(self) -> dict[str, int]:
        config = self.model.config
        id2label = getattr(config, "id2label", {}) or {}
        found: dict[str, int] = {}
        for index in range(int(getattr(config, "num_labels", len(id2label)))):
            raw = id2label.get(index, id2label.get(str(index), ""))
            label = str(raw).lower().replace("_", " ").replace("-", " ")
            for name in ("entailment", "neutral", "contradiction"):
                if name in label:
                    found[name] = index
        if set(found) != {"entailment", "neutral", "contradiction"}:
            raise ValueError(f"Could not resolve NLI labels from id2label={id2label!r}")
        return found

    def _batches(self, pairs: Sequence[tuple[str, str]]) -> Iterable[Sequence[tuple[str, str]]]:
        for start in range(0, len(pairs), self.batch_size):
            yield pairs[start:start + self.batch_size]

    def _encoded(self, batch: Sequence[tuple[str, str]]) -> Any:
        return self.tokenizer(
            [x[0] for x in batch], [x[1] for x in batch],
            return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(self.device)

    def predict_entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        values: list[float] = []
        with self.torch.inference_mode():
            for batch in self._batches(pairs):
                probabilities = self.torch.softmax(self.model(**self._encoded(batch)).logits, dim=-1)
                values.extend(float(row[self.label_indices["entailment"]]) for row in probabilities)
        return values

    def predict_regression(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        values: list[float] = []
        with self.torch.inference_mode():
            for batch in self._batches(pairs):
                logits = self.model(**self._encoded(batch)).logits
                if logits.shape[-1] != 1:
                    raise ValueError(f"Expected one regression output, got {tuple(logits.shape)}")
                # This is the single activation used by sentence-transformers
                # CrossEncoder for the one-logit STSB regression head.
                values.extend(float(x) for x in self.torch.sigmoid(logits[:, 0]))
        return values


def bidirectional(scorer: TransformerScorer, pairs: Sequence[tuple[str, str]]) -> tuple[list[float], list[float]]:
    values = scorer.predict_entailment(list(pairs) + [(b, a) for a, b in pairs])
    midpoint = len(pairs)
    return values[:midpoint], values[midpoint:]


def anchor_specs(audit_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    by_key = {(str(row.get("source_index")), row.get("augmented_field")): row for row in audit_rows}

    def real(anchor_id: str, source_index: str, field: str, verdict: str, reason: str) -> dict[str, str]:
        row = by_key.get((source_index, field))
        if not row:
            raise ValueError(f"Missing real anchor {source_index}:{field}")
        return {"anchor_id": anchor_id, "source_index": source_index,
                "original_sentence": row["original_sentence"],
                "back_translated_sentence": row["back_translated_sentence"],
                "expert_verdict": verdict, "expert_reason": reason}

    return [
        real("F1", "148356", "hypothesis", "FAIL", "red-haired/rides changed to red one/in a go-kart"),
        real("F2", "227619", "hypothesis", "FAIL", "progressive drowning changed to completed drowned"),
        real("F3", "512053", "hypothesis", "FAIL", "performing narrowed to playing"),
        real("F4", "355209", "hypothesis", "FAIL", "available calibration drift changes frowning to rubbing"),
        {"anchor_id": "F5", "source_index": "", "original_sentence": "A worker finishes work and leaves the office.",
         "back_translated_sentence": "A worker retires from work.", "expert_verdict": "FAIL",
         "expert_reason": "finishes/leaves work changed to retires from work"},
        {"anchor_id": "F6", "source_index": "", "original_sentence": "Two people are hugging.",
         "back_translated_sentence": "Two people are kissing.", "expert_verdict": "FAIL",
         "expert_reason": "hugging changed to kissing"},
        real("F7", "161674", "premise", "FAIL", "leaping above white water changed to jumping on white water"),
        real("F8", "138332", "premise", "FAIL", "tongues stuck out changed to deep blue tongues"),
        real("P1", "90", "hypothesis", "PASS", "nobody/no one normalization"),
        {"anchor_id": "P2", "source_index": "", "original_sentence": "The girl is not wearing shoes.",
         "back_translated_sentence": "The girl doesn't wear shoes.", "expert_verdict": "PASS",
         "expert_reason": "not and n't are equivalent negation forms"},
        {"anchor_id": "P3", "source_index": "", "original_sentence": "Two girls are playing outside.",
         "back_translated_sentence": "Both girls are playing outside.", "expert_verdict": "PASS",
         "expert_reason": "two and both preserve the count in context"},
        {"anchor_id": "P4", "source_index": "", "original_sentence": "The box contains 5 apples.",
         "back_translated_sentence": "The box contains five apples.", "expert_verdict": "PASS",
         "expert_reason": "numeric lexical normalization"},
        {"anchor_id": "P5", "source_index": "", "original_sentence": "One lone skier crosses the slope.",
         "back_translated_sentence": "A single skier crosses the slope.", "expert_verdict": "PASS",
         "expert_reason": "single explicit count normalization"},
    ]


def score_pair_rows(rows: list[dict[str, str]], current: TransformerScorer,
                    sts: TransformerScorer, roberta: TransformerScorer,
                    score_current: bool = False) -> dict[str, float]:
    pairs = [(row.get("original_sentence", ""), row.get("back_translated_sentence", "")) for row in rows]
    runtimes: dict[str, float] = {}
    started = time.perf_counter()
    sts_values = sts.predict_regression(pairs)
    runtimes["sts"] = time.perf_counter() - started
    started = time.perf_counter()
    roberta_forward, roberta_backward = bidirectional(roberta, pairs)
    runtimes["roberta"] = time.perf_counter() - started
    current_forward: list[float] = []
    current_backward: list[float] = []
    if score_current:
        started = time.perf_counter()
        current_forward, current_backward = bidirectional(current, pairs)
        runtimes["current"] = time.perf_counter() - started
    for i, row in enumerate(rows):
        row["secondary_sts_score"] = sts_values[i]
        row["secondary_roberta_forward_entailment"] = roberta_forward[i]
        row["secondary_roberta_backward_entailment"] = roberta_backward[i]
        row["secondary_roberta_min_entailment"] = min(roberta_forward[i], roberta_backward[i])
        row["secondary_bart_forward_entailment"] = ""
        row["secondary_bart_backward_entailment"] = ""
        row["secondary_bart_min_entailment"] = ""
        if score_current:
            row["sts_score"] = sts_values[i]
            row["roberta_forward"] = roberta_forward[i]
            row["roberta_backward"] = roberta_backward[i]
            row["roberta_min"] = min(roberta_forward[i], roberta_backward[i])
            row["current_semantic_forward"] = current_forward[i]
            row["current_semantic_backward"] = current_backward[i]
            row["current_semantic_min"] = min(current_forward[i], current_backward[i])
            row["current_deberta_forward"] = current_forward[i]
            row["current_deberta_backward"] = current_backward[i]
            row["current_deberta_min"] = min(current_forward[i], current_backward[i])
    return runtimes


def stable_review_rows(review_rows: Sequence[dict[str, str]], audit_rows: Sequence[dict[str, str]],
                       mapping_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    by_stable = {(str(row.get("source_index")), row.get("augmented_field")): row for row in audit_rows}
    by_position = {str(row.get("input_position")): row for row in audit_rows}
    mapping_by_position = {str(row.get("input_position")): row for row in mapping_rows}
    result = []
    for review in review_rows:
        stable = review.get("original_source_index") or review.get("source_index", "")
        field = review.get("augmented_field", "")
        audit = by_stable.get((str(stable), field))
        if audit is None:
            local = str(review.get("source_index", ""))
            mapped = mapping_by_position.get(local)
            if mapped:
                stable = mapped["source_index"]
                audit = by_stable.get((str(stable), field))
        if audit is None:
            raise ValueError(f"Could not join review candidate {review.get('candidate_id')}")
        row = dict(review)
        row.update({
            "input_position": audit.get("input_position", ""),
            "source_index": audit.get("source_index", stable),
            "original_source_index": stable,
            "original_candidate_id": review.get("candidate_id", ""),
            "candidate_id": audit.get("candidate_id", ""),
            "current_semantic_forward": audit.get("semantic_forward", ""),
            "current_semantic_backward": audit.get("semantic_backward", ""),
            "current_semantic_min": audit.get("semantic_min", ""),
        })
        result.append(row)
    return result


def anchor_pairwise(anchors: Sequence[dict[str, str]], field: str) -> float | None:
    passes = [as_float(x.get(field)) for x in anchors if x["expert_verdict"] == "PASS"]
    fails = [as_float(x.get(field)) for x in anchors if x["expert_verdict"] == "FAIL"]
    pairs = [(a, b) for a in passes for b in fails if a is not None and b is not None]
    return sum(a > b for a, b in pairs) / len(pairs) if pairs else None


def _metric_value(row: dict[str, str], field: str) -> float | None:
    aliases = {
        "secondary_sts_score": "sts_score",
        "secondary_roberta_min_entailment": "roberta_min",
    }
    return as_float(row.get(field, row.get(aliases.get(field, ""))))


def threshold_rows(anchors: Sequence[dict[str, str]], accepted: Sequence[dict[str, str]],
                   field: str, thresholds: Sequence[float]) -> list[dict[str, Any]]:
    out = []
    for threshold in thresholds:
        retained = [x for x in accepted if (_metric_value(x, field) or -1) >= threshold]
        fails = sum((_metric_value(x, field) is not None and _metric_value(x, field) < threshold)
                   for x in anchors if x["expert_verdict"] == "FAIL")
        passes = sum((_metric_value(x, field) is not None and _metric_value(x, field) >= threshold)
                     for x in anchors if x["expert_verdict"] == "PASS")
        out.append({"threshold": threshold, "retained": len(retained),
                    "newly_rejected": len(accepted) - len(retained),
                    "retention_rate": len(retained) / len(accepted) if accepted else None,
                    "fail_anchors_rejected": fails, "pass_anchors_retained": passes})
    return out


def secondary_disagreement_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    selected = [
        row for row in rows
        if as_bool(row.get("accepted")) and (as_float(row.get("semantic_min")) or 0) >= .95
        and ((as_float(row.get("secondary_sts_score")) or 1) < .80
             or (as_float(row.get("secondary_roberta_min_entailment")) or 1) < .80)
    ]
    selected.sort(key=lambda row: min(
        as_float(row.get("secondary_sts_score")) or 1,
        as_float(row.get("secondary_roberta_min_entailment")) or 1,
    ))
    return selected


def rank_percentile(value: float | None, population: Sequence[float]) -> float | None:
    if value is None or not population:
        return None
    return 100 * sum(x <= value for x in population) / len(population)


def write_anchor_report(anchors: Sequence[dict[str, str]], path: Path,
                        runtimes: dict[str, float], revisions: dict[str, str]) -> None:
    lines = [
        "# Semantic Verifier Anchor Report", "",
        "Fixed expert labels are sanity checks, not a threshold calibration set.", "",
        "| id | verdict | current DeBERTa fwd | current DeBERTa bwd | current DeBERTa min | STS | RoBERTa fwd | RoBERTa bwd | RoBERTa min | original | back-translated |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in anchors:
        lines.append(f"| {row['anchor_id']} | {row['expert_verdict']} | {fmt(row['current_deberta_forward'])} | {fmt(row['current_deberta_backward'])} | {fmt(row['current_deberta_min'])} | {fmt(row['sts_score'])} | {fmt(row['roberta_forward'])} | {fmt(row['roberta_backward'])} | {fmt(row['roberta_min'])} | {row['original_sentence']} | {row['back_translated_sentence']} |")
    lines.extend([
        "", "## Models and runtimes", "",
        f"- Current DeBERTa: {CURRENT_MODEL}, revision {revisions.get('current', 'unknown')}, anchor runtime {runtimes.get('current', 0):.2f}s.",
        f"- STS: {STS_MODEL}, revision {revisions.get('sts', 'unknown')}, anchor runtime {runtimes.get('sts', 0):.2f}s.",
        f"- RoBERTa MNLI: {ROBERTA_MODEL}, revision {revisions.get('roberta', 'unknown')}, anchor runtime {runtimes.get('roberta', 0):.2f}s.",
        "- BART MNLI: not run (optional model; no result is inferred).",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bakeoff_report(anchors: Sequence[dict[str, str]], translated: Sequence[dict[str, str]],
                         reviews: Sequence[dict[str, str]], path: Path,
                         runtimes: dict[str, float], revisions: dict[str, str],
                         threshold_data: dict[str, list[dict[str, Any]]]) -> None:
    accepted = [x for x in translated if as_bool(x.get("accepted"))]
    rejected = [x for x in translated if not as_bool(x.get("accepted"))]
    lines = [
        "# Semantic Verifier Bake-off Report", "",
        "Analysis-only. No NLLB generation or production-policy decision was changed.", "",
        "## Models", "",
        f"- Current semantic baseline: {CURRENT_MODEL}, revision {revisions.get('current', CURRENT_REVISION)}; existing V2 semantic scores reused for 1,336 translated candidates.",
        f"- STS cross-encoder: {STS_MODEL}, revision {revisions.get('sts', 'unknown')}; CPU runtime {runtimes.get('full_sts', 0):.2f}s.",
        f"- Independent NLI: {ROBERTA_MODEL}, revision {revisions.get('roberta', 'unknown')}; CPU runtime {runtimes.get('full_roberta', 0):.2f}s.",
        "- BART MNLI: not run (optional model; no result is inferred).", "",
        "## Expert-anchor pairwise ranking", "",
        "| model | PASS > FAIL pairwise score |", "| --- | ---: |",
    ]
    for name, field in [("Current DeBERTa", "current_deberta_min"), ("STS", "sts_score"), ("RoBERTa MNLI", "roberta_min")]:
        lines.append(f"| {name} | {fmt(anchor_pairwise(anchors, field))} |")
    lines.extend(["", "## Anchor scores", "",
                   "| id | verdict | Current DeBERTa min | STS | RoBERTa min |",
                   "| --- | --- | ---: | ---: | ---: |"])
    for row in anchors:
        lines.append(f"| {row['anchor_id']} | {row['expert_verdict']} | {fmt(row['current_deberta_min'])} | {fmt(row['sts_score'])} | {fmt(row['roberta_min'])} |")
    lines.extend(["", "## Full translated-candidate distributions", "",
                   "| subset | metric | count | mean | median | p05 | p10 | p25 | p75 | p90 |",
                   "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for subset_name, subset in [("accepted", accepted), ("rejected", rejected)]:
        for metric, field in [("current_deberta_min", "semantic_min"), ("STS", "secondary_sts_score"), ("RoBERTa_min", "secondary_roberta_min_entailment")]:
            stats = distribution(row.get(field) for row in subset)
            lines.append(f"| {subset_name} | {metric} | {stats['count']} | {fmt(stats['mean'])} | {fmt(stats['median'])} | {fmt(stats['p05'])} | {fmt(stats['p10'])} | {fmt(stats['p25'])} | {fmt(stats['p75'])} | {fmt(stats['p90'])} |")
    drift = [row for row in rejected if "semantic_drift" in json.loads(row.get("reasons") or "[]")]
    lines.extend(["", "## Semantic-drift rejection distributions", "",
                   "| subset | metric | count | mean | median | p75 | p90 |",
                   "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for metric, field in [("STS", "secondary_sts_score"), ("RoBERTa_min", "secondary_roberta_min_entailment")]:
        stats = distribution(row.get(field) for row in drift)
        lines.append(f"| semantic_drift rejected | {metric} | {stats['count']} | {fmt(stats['mean'])} | {fmt(stats['median'])} | {fmt(stats['p75'])} | {fmt(stats['p90'])} |")
    disagreement_count = sum(
        as_bool(row.get("accepted")) and (as_float(row.get("semantic_min")) or 0) >= .95
        and ((as_float(row.get("secondary_sts_score")) or 1) < .80
             or (as_float(row.get("secondary_roberta_min_entailment")) or 1) < .80)
        for row in translated
    )
    lines.extend(["", "## Threshold simulation on current accepted candidates", "",
                   "Analysis only; these thresholds do not change production decisions.", "",
                   "| model | threshold | retained | newly rejected | retention | FAIL rejected | PASS retained |",
                   "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for model, values in threshold_data.items():
        for item in values:
            lines.append(f"| {model} | {item['threshold']:.2f} | {item['retained']} | {item['newly_rejected']} | {item['retention_rate']:.2%} | {item['fail_anchors_rejected']} | {item['pass_anchors_retained']} |")
    lines.extend(["", "## Canary ranks among translated candidates", "",
                   "Percentile is the share of translated candidates scoring at or below the canary.", "",
                   "| source_index | STS percentile | RoBERTa percentile |", "| --- | ---: | ---: |"])
    sts_population = [float(x["secondary_sts_score"]) for x in translated if as_float(x.get("secondary_sts_score")) is not None]
    nli_population = [float(x["secondary_roberta_min_entailment"]) for x in translated if as_float(x.get("secondary_roberta_min_entailment")) is not None]
    for source_index in ("148356", "227619", "512053"):
        matches = [x for x in translated if str(x.get("source_index")) == source_index and x.get("augmented_field") == "hypothesis"]
        if not matches:
            lines.append(f"| {source_index} | not found | not found |")
        else:
            row = matches[0]
            lines.append(f"| {source_index} | {fmt(rank_percentile(as_float(row.get('secondary_sts_score')), sts_population))}th | {fmt(rank_percentile(as_float(row.get('secondary_roberta_min_entailment')), nli_population))}th |")
    lines.extend(["", "## Secondary disagreements", "",
                   f"- Master review candidates scored: {len(reviews)}.",
                   f"- Secondary disagreement candidates exported: {disagreement_count} (capped at 150).",
                   "- review_secondary_disagreement.csv selects current-policy accepted candidates with current semantic min >= 0.95 and STS < 0.80 or RoBERTa min < 0.80.",
                   "", "## Limitations", "",
                   "- Secondary scores are analysis columns only; production filtering was not changed.",
                   "- BART MNLI was not run.",
                   "- The 13-anchor ranking is a sanity check, not a formal benchmark.",
                   "- STS uses one native CrossEncoder sigmoid activation for its one-logit regression head.",
                   "- No NLLB generation, full-SNLI augmentation, or threshold change was performed.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    base = Path("data/nli/calibration")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=base / "snli_calibration_800_v2.audit.csv")
    parser.add_argument("--master-review", type=Path, default=base / "MASTER_REVIEW.csv")
    parser.add_argument("--mapping", type=Path, default=base / "v2_source_index_mapping.csv")
    parser.add_argument("--output-dir", type=Path, default=base)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    audit = read_csv(args.audit)
    mapping = read_csv(args.mapping)
    translated = [row for row in audit if as_bool(row.get("translation_performed"))]
    reviews = stable_review_rows(read_csv(args.master_review), audit, mapping)

    current = TransformerScorer(CURRENT_MODEL, args.device, args.batch_size, CURRENT_REVISION, "nli")
    sts = TransformerScorer(STS_MODEL, args.device, args.batch_size, task="sts")
    roberta = TransformerScorer(ROBERTA_MODEL, args.device, args.batch_size, task="nli")
    revisions = {"current": current.revision, "sts": sts.revision, "roberta": roberta.revision}

    anchors = anchor_specs(audit)
    anchor_runtime = score_pair_rows(anchors, current, sts, roberta, score_current=True)
    write_csv(args.output_dir / "semantic_verifier_anchor_set.csv", anchors, ANCHOR_COLUMNS)
    write_anchor_report(anchors, args.output_dir / "SEMANTIC_VERIFIER_ANCHOR_REPORT.md",
                        anchor_runtime, revisions)

    review_runtime = score_pair_rows(reviews, current, sts, roberta)
    write_csv(args.output_dir / "MASTER_REVIEW_SECONDARY_SCORES.csv", reviews, REVIEW_COLUMNS)

    started = time.perf_counter()
    full_runtime = score_pair_rows(translated, current, sts, roberta)
    full_elapsed = time.perf_counter() - started
    full_runtime["full_sts"] = full_runtime.get("sts", 0.0)
    full_runtime["full_roberta"] = full_runtime.get("roberta", 0.0)
    full_runtime["full_total"] = full_elapsed
    write_csv(args.output_dir / "snli_calibration_800_v2.secondary_semantic_scores.csv",
              translated, REVIEW_COLUMNS)

    accepted = [row for row in translated if as_bool(row.get("accepted"))]
    threshold_data = {
        "STS": threshold_rows(anchors, accepted, "secondary_sts_score", (.70, .75, .80, .85, .90, .95)),
        "RoBERTa min entailment": threshold_rows(
            anchors, accepted, "secondary_roberta_min_entailment", (.70, .80, .85, .90, .95)),
    }
    disagreement = secondary_disagreement_rows(translated)
    write_csv(args.output_dir / "review_secondary_disagreement.csv", disagreement[:150], REVIEW_COLUMNS)
    write_bakeoff_report(anchors, translated, reviews,
                         args.output_dir / "SEMANTIC_VERIFIER_BAKEOFF_REPORT.md",
                         full_runtime, revisions, threshold_data)

    print(json.dumps({
        "anchors": len(anchors), "master_review": len(reviews),
        "translated_candidates": len(translated), "accepted": len(accepted),
        "secondary_disagreement": min(len(disagreement), 150),
        "device": args.device, "sts_revision": sts.revision,
        "roberta_revision": roberta.revision,
        "anchor_runtime_seconds": round(sum(anchor_runtime.values()), 3),
        "full_runtime_seconds": round(full_elapsed, 3),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
