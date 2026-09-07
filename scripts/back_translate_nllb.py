#!/usr/bin/env python3
"""Stream NLI back-translation candidates through an explainable quality gate.

The recommended mode is asymmetric (``separate``): one candidate changes only
the premise and one changes only the hypothesis. Candidates are written to
accepted/rejected JSONL files as soon as each input chunk is processed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

try:
    from scripts.quality_filter import Decision, QualityFilter, canonical_label, validate_record
except ModuleNotFoundError:
    from quality_filter import Decision, QualityFilter, canonical_label, validate_record


DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_VERIFIER = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Read JSONL or a JSON array; JSONL is streamed line by line."""
    with path.open("r", encoding="utf-8") as handle:
        first_non_empty = next((line.lstrip() for line in handle if line.strip()), "")
    if not first_non_empty:
        return
    if first_non_empty.startswith("["):
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON array in {path}")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Every dataset item must be a JSON object")
            yield record
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield record


def indexed_chunks(records: Iterable[dict[str, Any]], chunk_size: int) -> Iterator[list[tuple[int, dict[str, Any]]]]:
    chunk: list[tuple[int, dict[str, Any]]] = []
    for source_index, record in enumerate(records):
        chunk.append((source_index, record))
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def batched(items: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    return "float16" if device.startswith("cuda") else "float32"


class NLLBBackTranslator:
    """Batched NLLB translator with explicit truncation reporting."""

    def __init__(self, model_name: str, source_lang: str, pivot_lang: str, device: str, dtype: str,
                 max_input_tokens: int, max_new_tokens: int, num_beams: int) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
        self.torch = torch
        self.device = resolve_device(device)
        self.dtype_name = resolve_dtype(dtype, self.device)
        self.source_lang = source_lang
        self.pivot_lang = pivot_lang
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.model_name = model_name
        torch_dtype = getattr(torch, self.dtype_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=source_lang)
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch_dtype)
        except TypeError:  # transformers < 4.47
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model = self.model.to(self.device).eval()
        self.model_revision = getattr(self.model.config, "_commit_hash", None)

    def token_length(self, text: str, source_lang: str | None = None) -> int:
        self.tokenizer.src_lang = source_lang or self.source_lang
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def _language_id(self, language: str) -> int:
        language_ids = getattr(self.tokenizer, "lang_code_to_id", {})
        if language in language_ids:
            return language_ids[language]
        language_id = self.tokenizer.convert_tokens_to_ids(language)
        if language_id is None or language_id < 0:
            raise ValueError(f"Unsupported NLLB language code: {language}")
        return language_id

    def _translate(self, texts: Sequence[str], source_lang: str, target_lang: str,
                   allow_truncation: bool) -> tuple[list[str], list[bool]]:
        if not texts:
            return [], []
        self.tokenizer.src_lang = source_lang
        was_truncated = [self.token_length(text, source_lang) > self.max_input_tokens for text in texts]
        encoded = self.tokenizer(list(texts), return_tensors="pt", padding=True,
                                 truncation=allow_truncation,
                                 max_length=self.max_input_tokens if allow_truncation else None).to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**encoded, forced_bos_token_id=self._language_id(target_lang),
                                            max_new_tokens=self.max_new_tokens, num_beams=self.num_beams,
                                            do_sample=False)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True), was_truncated

    def translate_with_metadata(self, texts: Sequence[str], batch_size: int, allow_truncation: bool) -> tuple[list[str], list[bool]]:
        translated: list[str] = []
        truncated: list[bool] = []
        for batch in batched(texts, batch_size):
            pivot, source_truncated = self._translate(batch, self.source_lang, self.pivot_lang, allow_truncation)
            restored, _ = self._translate(pivot, self.pivot_lang, self.source_lang, allow_truncation)
            translated.extend(restored)
            truncated.extend(source_truncated)
        return translated, truncated

    def translate(self, texts: Sequence[str], batch_size: int) -> list[str]:
        return self.translate_with_metadata(texts, batch_size, False)[0]


@dataclass(frozen=True)
class Candidate:
    source_index: int
    record: dict[str, Any]
    premise: str
    hypothesis: str
    augmented_field: str
    was_truncated: bool

    @property
    def candidate_id(self) -> str:
        return f"{self.source_index}:{self.augmented_field}"


def candidate_fields(mode: str) -> tuple[str, ...]:
    return ("premise", "hypothesis") if mode == "separate" else (mode,)


def translation_fields(mode: str) -> tuple[str, ...]:
    return ("premise", "hypothesis") if mode == "both" else candidate_fields(mode)


def make_candidates(source_index: int, record: dict[str, Any], translations: dict[str, str],
                    truncated: dict[str, bool], mode: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for field in candidate_fields(mode):
        premise = translations.get("premise", record["premise"]) if field in ("premise", "both") else record["premise"]
        hypothesis = translations.get("hypothesis", record["hypothesis"]) if field in ("hypothesis", "both") else record["hypothesis"]
        candidates.append(Candidate(source_index, record, premise, hypothesis, field,
                                    any(truncated.get(name, False) for name in translation_fields(mode))))
    return candidates


def candidate_key(record: dict[str, Any]) -> str | None:
    if record.get("candidate_id"):
        return str(record["candidate_id"])
    if "source_index" in record and "augmented_field" in record:
        return f"{record['source_index']}:{record['augmented_field']}"
    return None


def completed_candidate_ids(paths: Sequence[Path]) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        if path.exists():
            for record in read_records(path):
                key = candidate_key(record)
                if key is not None:
                    completed.add(key)
    return completed


def rejected_path(accepted_path: Path) -> Path:
    if accepted_path.suffix == ".jsonl" and accepted_path.stem.endswith(".accepted"):
        return accepted_path.with_name(accepted_path.stem[:-9] + ".rejected.jsonl")
    return accepted_path.with_name(accepted_path.stem + ".rejected" + accepted_path.suffix)


def provenance(args: argparse.Namespace, translator: NLLBBackTranslator | None) -> dict[str, Any]:
    return {
        "translation_model": args.model_name,
        "translation_model_revision": getattr(translator, "model_revision", None),
        "verifier_model": args.verifier_model if args.quality_filter == "on" else None,
        "verifier_model_revision": getattr(args, "verifier_revision", None),
        "generation": {"num_beams": args.num_beams, "do_sample": False,
                        "max_input_tokens": args.max_input_tokens, "max_new_tokens": args.max_new_tokens},
        "quality": {"semantic_threshold": args.semantic_threshold, "nli_threshold": args.nli_threshold,
                    "min_change_ratio": args.min_change_ratio},
    }


def output_record(candidate: Candidate, decision: Decision, args: argparse.Namespace,
                  translator: NLLBBackTranslator | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_index": candidate.source_index, "candidate_id": candidate.candidate_id,
        "label": candidate.record.get("label"),
        "original_premise": candidate.record.get("premise"),
        "original_hypothesis": candidate.record.get("hypothesis"),
        "premise": candidate.premise, "hypothesis": candidate.hypothesis,
        "augmentation": "back_translation", "augmented_field": candidate.augmented_field,
        "pivot_lang": args.pivot_lang, "was_truncated": candidate.was_truncated,
        "quality": {**decision.scores, "accepted": decision.accepted,
                    "reasons": decision.reasons, "flags": decision.flags},
        "provenance": provenance(args, translator),
    }
    if args.no_metadata and decision.accepted:
        return {key: record[key] for key in ("premise", "hypothesis", "label")}
    return record


def invalid_output(source_index: int, record: dict[str, Any], field: str, reasons: list[str],
                   args: argparse.Namespace) -> dict[str, Any]:
    candidate = Candidate(source_index, record,
                          record.get("premise", "") if isinstance(record.get("premise", ""), str) else "",
                          record.get("hypothesis", "") if isinstance(record.get("hypothesis", ""), str) else "",
                          field, "input_too_long" in reasons)
    return output_record(candidate, Decision(False, list(dict.fromkeys(reasons))), args, None)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Accepted JSONL path (legacy alias)")
    parser.add_argument("--accepted-output", type=Path)
    parser.add_argument("--rejected-output", type=Path)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--source-lang", default="eng_Latn")
    parser.add_argument("--pivot-lang", default="fra_Latn")
    parser.add_argument("--augmentation-mode", choices=("separate", "premise", "hypothesis", "both"), default="separate")
    parser.add_argument("--quality-filter", choices=("on", "off"), default="on")
    parser.add_argument("--verifier-model", default=DEFAULT_VERIFIER)
    parser.add_argument("--semantic-threshold", type=float, default=0.80)
    parser.add_argument("--nli-threshold", type=float, default=0.80)
    parser.add_argument("--min-change-ratio", type=float, default=0.03)
    parser.add_argument("--filter-batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    args = parser.parse_args(argv)
    if args.output is None and args.accepted_output is None:
        parser.error("one of --output or --accepted-output is required")
    if args.output is not None and args.accepted_output is not None:
        parser.error("--output and --accepted-output are mutually exclusive")
    if args.quality_filter == "on" and args.no_metadata:
        parser.error("--no-metadata is only supported with --quality-filter off")
    if min(args.batch_size, args.chunk_size, args.filter_batch_size, args.max_input_tokens, args.max_new_tokens, args.num_beams) < 1:
        parser.error("batch sizes, token limits, and num_beams must be positive")
    if not 0 <= args.semantic_threshold <= 1 or not 0 <= args.nli_threshold <= 1 or not 0 <= args.min_change_ratio <= 1:
        parser.error("quality thresholds must be between 0 and 1")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    accepted_path = args.accepted_output or args.output
    assert accepted_path is not None
    rejected_output = args.rejected_output or rejected_path(accepted_path)
    for path in (accepted_path, rejected_output):
        if path.exists() and not (args.resume or args.overwrite):
            raise FileExistsError(f"Output exists: {path}. Use --resume or --overwrite.")
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    completed = completed_candidate_ids((accepted_path, rejected_output)) if args.resume else set()
    translator = NLLBBackTranslator(args.model_name, args.source_lang, args.pivot_lang, args.device, args.dtype,
                                    args.max_input_tokens, args.max_new_tokens, args.num_beams)
    quality = None
    if args.quality_filter == "on":
        try:
            from scripts.quality_filter import TransformersNLI
        except ModuleNotFoundError:
            from quality_filter import TransformersNLI
        verifier = TransformersNLI(args.verifier_model, resolve_device(args.device), args.dtype, args.filter_batch_size)
        args.verifier_revision = verifier.model_revision
        quality = QualityFilter(verifier, args.semantic_threshold, args.nli_threshold, args.min_change_ratio)

    accepted_mode = "a" if args.resume and accepted_path.exists() else "w"
    rejected_mode = "a" if args.resume and rejected_output.exists() else "w"
    accepted_count = rejected_count = seen = 0
    with accepted_path.open(accepted_mode, encoding="utf-8") as accepted_handle, rejected_output.open(rejected_mode, encoding="utf-8") as rejected_handle:
        for chunk in indexed_chunks(read_records(args.input), args.chunk_size):
            if args.max_samples is not None and seen >= args.max_samples:
                break
            if args.max_samples is not None:
                chunk = chunk[: args.max_samples - seen]
            seen += len(chunk)
            valid: list[tuple[int, dict[str, Any]]] = []
            for source_index, record in chunk:
                reasons = validate_record(record)
                if not reasons and any(translator.token_length(record[field]) > args.max_input_tokens for field in ("premise", "hypothesis") if field in record) and not args.allow_truncation:
                    reasons = ["input_too_long"]
                if reasons:
                    for field in candidate_fields(args.augmentation_mode):
                        candidate_id = f"{source_index}:{field}"
                        if candidate_id in completed:
                            continue
                        rejected_handle.write(json.dumps(invalid_output(source_index, record, field, reasons, args), ensure_ascii=False, separators=(",", ":")) + "\n")
                        rejected_count += 1
                    rejected_handle.flush()
                else:
                    valid.append((source_index, record))

            translations: dict[str, dict[int, str]] = {field: {} for field in translation_fields(args.augmentation_mode)}
            truncations: dict[str, dict[int, bool]] = {field: {} for field in translation_fields(args.augmentation_mode)}
            for field in translation_fields(args.augmentation_mode):
                targets = [(index, record) for index, record in valid if any(f"{index}:{candidate_field}" not in completed for candidate_field in candidate_fields(args.augmentation_mode))]
                if not targets:
                    continue
                texts, flags = translator.translate_with_metadata([record[field] for _, record in targets], args.batch_size, args.allow_truncation)
                for (index, _), text_value, flag in zip(targets, texts, flags):
                    translations[field][index] = text_value
                    truncations[field][index] = flag

            for source_index, record in valid:
                translated = {field: translations[field].get(source_index, record[field]) for field in translation_fields(args.augmentation_mode)}
                truncated = {field: truncations[field].get(source_index, False) for field in translation_fields(args.augmentation_mode)}
                for candidate in make_candidates(source_index, record, translated, truncated, args.augmentation_mode):
                    if candidate.candidate_id in completed:
                        continue
                    if quality is None:
                        decision = Decision(True)
                    else:
                        decision = quality.evaluate(record["premise"], record["hypothesis"], candidate.premise,
                                                    candidate.hypothesis, int(record["label"]), candidate.augmented_field,
                                                    candidate.was_truncated)
                    value = output_record(candidate, decision, args, translator)
                    handle = accepted_handle if decision.accepted else rejected_handle
                    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                    if decision.accepted:
                        accepted_count += 1
                    else:
                        rejected_count += 1
            print(f"processed={seen} accepted={accepted_count} rejected={rejected_count}", file=sys.stderr)
    print(f"Accepted: {accepted_path}")
    print(f"Rejected: {rejected_output}")
    return 0


if __name__ == "__main__":
    main()
