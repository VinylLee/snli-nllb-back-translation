#!/usr/bin/env python3
"""Analyze reviewer labels for the blind accepted-candidate precision audit.

This script intentionally does not inspect or recompute model scores. It joins a
future reviewer-completed blind CSV to the secret key only after review.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SEED = 20260908
BOOTSTRAP_ITERATIONS = 20_000
STRATA_FIELDS = ("gold_label_name", "augmented_field")
REVIEW_FIELDS = (
    "review_semantic_equivalent", "review_label_preserved",
    "review_useful_augmentation", "review_severity", "review_error_types",
    "review_notes",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_unique(rows: Sequence[dict[str, str]], field: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(field, "")
        if not value or value in result:
            raise ValueError(f"Missing or duplicate {field}: {value!r}")
        result[value] = row
    return result


def merge_review_rows(review_rows: Sequence[dict[str, str]], key_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    key_by_id = _require_unique(key_rows, "audit_id")
    review_by_id = _require_unique(review_rows, "audit_id")
    if set(key_by_id) != set(review_by_id):
        missing = sorted(set(key_by_id) - set(review_by_id))
        extra = sorted(set(review_by_id) - set(key_by_id))
        raise ValueError(f"Review/key audit_id mismatch; missing={missing[:5]}, extra={extra[:5]}")
    merged = []
    for review in review_rows:
        key = key_by_id[review["audit_id"]]
        item = dict(review)
        item.update({f"key_{name}": value for name, value in key.items() if name != "audit_id"})
        item["stratum_population_size"] = key.get("stratum_population_size", "")
        merged.append(item)
    return merged


def _is_empty(value: Any) -> bool:
    return str(value or "").strip() == ""


def validate_review_values(rows: Sequence[dict[str, str]]) -> None:
    semantic = {"PASS", "FAIL", "BORDERLINE"}
    label = {"PASS", "FAIL", "UNCERTAIN"}
    useful = {"PASS", "FAIL", "BORDERLINE"}
    severity = {"NONE", "MINOR", "MAJOR"}
    for row in rows:
        for field, allowed in (
            ("review_semantic_equivalent", semantic),
            ("review_label_preserved", label),
            ("review_useful_augmentation", useful),
            ("review_severity", severity),
        ):
            if row.get(field, "") not in allowed:
                raise ValueError(f"Invalid {field} for {row.get('audit_id')}: {row.get(field)!r}")


def _groups(rows: Sequence[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    result: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[(row.get("gold_label_name", row.get("key_gold_label", "")),
                row.get("augmented_field", row.get("key_augmented_field", "")))].append(row)
    return result


def _population_size(group: Sequence[dict[str, str]]) -> float:
    if not group:
        return 0.0
    value = group[0].get("stratum_population_size", "")
    if not value:
        value = group[0].get("key_stratum_population_size", "")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing stratum population size for {group[0].get('audit_id')}") from exc


def sample_rate(rows: Sequence[dict[str, str]], predicate: Callable[[dict[str, str]], bool]) -> float | None:
    return sum(predicate(row) for row in rows) / len(rows) if rows else None


def weighted_rate(rows: Sequence[dict[str, str]], predicate: Callable[[dict[str, str]], bool]) -> float | None:
    groups = _groups(rows)
    denominator = sum(_population_size(group) for group in groups.values())
    if not groups or denominator == 0:
        return None
    return sum(_population_size(group) * (sample_rate(group, predicate) or 0.0)
               for group in groups.values()) / denominator


def bootstrap_ci(rows: Sequence[dict[str, str]], predicate: Callable[[dict[str, str]], bool],
                 iterations: int = BOOTSTRAP_ITERATIONS, seed: int = SEED) -> tuple[float | None, float | None]:
    groups = _groups(rows)
    denominator = sum(_population_size(group) for group in groups.values())
    if not groups or denominator == 0:
        return None, None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        weighted_sum = 0.0
        for group in groups.values():
            successes = sum(predicate(group[rng.randrange(len(group))]) for _ in group)
            weighted_sum += _population_size(group) * successes / len(group)
        estimates.append(weighted_sum / denominator)
    estimates.sort()
    return _percentile(estimates, 2.5), _percentile(estimates, 97.5)


def _percentile(values: Sequence[float], percent: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * percent / 100.0
    lower, upper = int(position), min(int(position) + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def strict_quality(row: dict[str, str]) -> bool:
    return (row.get("review_semantic_equivalent") == "PASS"
            and row.get("review_label_preserved") == "PASS"
            and row.get("review_useful_augmentation") == "PASS")


def optimistic_quality(row: dict[str, str]) -> bool:
    return (row.get("review_semantic_equivalent") in {"PASS", "BORDERLINE"}
            and row.get("review_label_preserved") in {"PASS", "UNCERTAIN"}
            and row.get("review_useful_augmentation") in {"PASS", "BORDERLINE"})


def metric_summary(rows: Sequence[dict[str, str]], predicate: Callable[[dict[str, str]], bool]) -> dict[str, float | None]:
    low, high = bootstrap_ci(rows, predicate)
    return {"sample_rate": sample_rate(rows, predicate), "weighted_rate": weighted_rate(rows, predicate),
            "ci_lower": low, "ci_upper": high}


def error_counts(rows: Iterable[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for error in str(row.get("review_error_types", "")).split("|"):
            error = error.strip()
            if error:
                counts[error] += 1
    return counts


def operational_status(strict_weighted: float | None, strict_lower: float | None,
                       major_weighted: float | None) -> str:
    if strict_weighted is None or strict_lower is None or major_weighted is None:
        return "NOT AVAILABLE — review columns are incomplete"
    if strict_weighted >= .97 and strict_lower >= .95 and major_weighted <= .02:
        return "GREEN — full SNLI candidate generation can proceed"
    if strict_weighted < .93 or major_weighted > .05:
        return "RED — do not run full SNLI yet"
    return "YELLOW — targeted guard refinement before full run"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def write_report(rows: Sequence[dict[str, str]], output: Path, seed: int = SEED,
                 iterations: int = BOOTSTRAP_ITERATIONS) -> dict[str, Any]:
    strict = metric_summary(rows, strict_quality)
    optimistic = metric_summary(rows, optimistic_quality)
    semantic_pass = lambda row: row.get("review_semantic_equivalent") == "PASS"
    semantic_fail = lambda row: row.get("review_semantic_equivalent") == "FAIL"
    label_fail = lambda row: row.get("review_label_preserved") == "FAIL"
    major = lambda row: row.get("review_severity") == "MAJOR"
    major_summary = metric_summary(rows, major)
    lines = ["# Accepted Precision Audit Analysis", "",
             f"Bootstrap seed: `{seed}`; iterations: `{iterations}`.", "",
             "This report is only valid after reviewer labels are merged with the secret key. "
             "The blind sample is stratified and population-weighted; its source set is the 774 accepted translated V2 candidates.", "",
             "## Overall metrics", "",
             "| metric | unweighted sample | population-weighted | bootstrap 95% CI |", "| --- | ---: | ---: | ---: |",
             f"| Strict quality PASS | {_fmt(strict['sample_rate'])} | {_fmt(strict['weighted_rate'])} | {_fmt(strict['ci_lower'])}–{_fmt(strict['ci_upper'])} |",
             f"| Optimistic quality PASS | {_fmt(optimistic['sample_rate'])} | {_fmt(optimistic['weighted_rate'])} | {_fmt(optimistic['ci_lower'])}–{_fmt(optimistic['ci_upper'])} |",
             f"| Semantic equivalent PASS | {_fmt(sample_rate(rows, semantic_pass))} | {_fmt(weighted_rate(rows, semantic_pass))} | {_fmt(bootstrap_ci(rows, semantic_pass)[0])}–{_fmt(bootstrap_ci(rows, semantic_pass)[1])} |",
             f"| Semantic equivalent FAIL | {_fmt(sample_rate(rows, semantic_fail))} | {_fmt(weighted_rate(rows, semantic_fail))} | — |",
             f"| Label preservation FAIL | {_fmt(sample_rate(rows, label_fail))} | {_fmt(weighted_rate(rows, label_fail))} | — |",
             f"| MAJOR error | {_fmt(major_summary['sample_rate'])} | {_fmt(major_summary['weighted_rate'])} | {_fmt(major_summary['ci_lower'])}–{_fmt(major_summary['ci_upper'])} |",
             "", "## Per-stratum metrics", "",
             "| gold label | field | sample n | population N | strict quality | semantic FAIL | label FAIL | MAJOR |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, group in sorted(_groups(rows).items()):
        lines.append(f"| {key[0]} | {key[1]} | {len(group)} | {int(_population_size(group))} | "
                     f"{_fmt(sample_rate(group, strict_quality))} | {_fmt(sample_rate(group, semantic_fail))} | "
                     f"{_fmt(sample_rate(group, label_fail))} | {_fmt(sample_rate(group, major))} |")
    lines.extend(["", "## Error taxonomy", "", "| error type | count |", "| --- | ---: |"] )
    for error, count in sorted(error_counts(rows).items()):
        lines.append(f"| {error} | {count} |")
    lines.extend(["", "## Operational interpretation", "", f"- {operational_status(strict['weighted_rate'], strict['ci_lower'], major_summary['weighted_rate'])}",
                  "- GREEN: weighted strict precision ≥97%, CI lower bound ≥95%, and weighted MAJOR rate ≤2%.",
                  "- YELLOW: strict precision 93–97%, CI lower bound <95%, or MAJOR rate 2–5%.",
                  "- RED: weighted strict precision <93% or MAJOR rate >5%.",
                  "", "BORDERLINE and UNCERTAIN are excluded from strict quality and admitted only by the optimistic definition."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"strict": strict, "optimistic": optimistic, "major": major_summary,
            "error_counts": error_counts(rows)}


def analyze(review_path: Path, key_path: Path, output: Path) -> dict[str, Any]:
    rows = merge_review_rows(read_csv(review_path), read_csv(key_path))
    validate_review_values(rows)
    return write_report(rows, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", type=Path, default=Path("data/nli/calibration/accepted_precision_audit_blind_reviewed.csv"))
    parser.add_argument("--key", type=Path, default=Path("data/nli/calibration/accepted_precision_audit_key.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/nli/calibration/ACCEPTED_PRECISION_AUDIT_ANALYSIS.md"))
    args = parser.parse_args()
    result = analyze(args.reviewed, args.key, args.output)
    print({"rows": len(read_csv(args.reviewed)), "strict_weighted": result["strict"]["weighted_rate"],
           "major_weighted": result["major"]["weighted_rate"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
