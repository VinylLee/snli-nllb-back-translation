#!/usr/bin/env python3
"""Generate NLI data with back translation using an NLLB-200 model.

The input is JSONL (one JSON object per line) or a JSON array.  By default the
script back-translates both ``premise`` and ``hypothesis`` independently:

    eng_Latn -> fra_Latn -> eng_Latn

The output keeps the input schema and adds small provenance fields.  Those
fields can be disabled with ``--no-metadata`` when a strict schema is needed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_FIELDS = ("premise", "hypothesis")


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Read JSONL or a JSON array without loading the whole dataset."""
    with path.open("r", encoding="utf-8") as handle:
        first_non_empty = ""
        for line in handle:
            if line.strip():
                first_non_empty = line.lstrip()
                break

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
    if device.startswith("cuda"):
        return "float16"
    return "float32"


class NLLBBackTranslator:
    """Thin batched wrapper around ``AutoModelForSeq2SeqLM``."""

    def __init__(
        self,
        model_name: str,
        source_lang: str,
        pivot_lang: str,
        device: str,
        dtype: str,
        max_input_tokens: int,
        max_new_tokens: int,
        num_beams: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "This script requires torch and transformers. "
                "Install them with: pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        self.device = resolve_device(device)
        self.dtype_name = resolve_dtype(dtype, self.device)
        torch_dtype = getattr(torch, self.dtype_name)
        self.source_lang = source_lang
        self.pivot_lang = pivot_lang
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, src_lang=source_lang
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(self.device)
        self.model.eval()

    def _language_id(self, language: str) -> int:
        # NLLB tokenizers expose either lang_code_to_id or convert_tokens_to_ids
        # depending on the transformers version.
        language_ids = getattr(self.tokenizer, "lang_code_to_id", {})
        if language in language_ids:
            return language_ids[language]
        language_id = self.tokenizer.convert_tokens_to_ids(language)
        if language_id is None or language_id < 0:
            raise ValueError(f"Unsupported NLLB language code: {language}")
        return language_id

    def _translate(self, texts: Sequence[str], source_lang: str, target_lang: str) -> list[str]:
        if not texts:
            return []
        self.tokenizer.src_lang = source_lang
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=self._language_id(target_lang),
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def translate(self, texts: Sequence[str], batch_size: int) -> list[str]:
        """Translate source -> pivot -> source in batches."""
        result: list[str] = []
        for batch in batched(texts, batch_size):
            pivot = self._translate(batch, self.source_lang, self.pivot_lang)
            result.extend(self._translate(pivot, self.pivot_lang, self.source_lang))
        return result


def make_augmented_records(
    records: Sequence[dict[str, Any]],
    translated_fields: dict[str, Sequence[str]],
    start_index: int,
    pivot_lang: str,
    include_metadata: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for offset, record in enumerate(records):
        augmented = dict(record)
        for field, translations in translated_fields.items():
            augmented[field] = translations[offset]
        if include_metadata:
            augmented["augmentation"] = "back_translation"
            augmented["source_index"] = start_index + offset
            augmented["pivot_lang"] = pivot_lang
        output.append(augmented)
    return output


def write_records(handle: Any, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        count += 1
    return count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL/JSON dataset")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--source-lang", default="eng_Latn", help="NLLB source language code")
    parser.add_argument("--pivot-lang", default="fra_Latn", help="NLLB pivot language code")
    parser.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--offline", action="store_true", help="Only use locally cached model files")
    parser.add_argument("--resume", action="store_true", help="Append after existing complete output records")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    parser.add_argument("--no-metadata", action="store_true", help="Keep only the input fields")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_input_tokens < 1 or args.max_new_tokens < 1:
        parser.error("batch and token limits must be positive")
    if args.num_beams < 1:
        parser.error("--num-beams must be positive")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    input_path = args.input
    output_path = args.output
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"Output exists: {output_path}. Use --resume or --overwrite explicitly."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    if args.resume and output_path.exists():
        completed = sum(1 for _ in read_records(output_path))

    records = list(read_records(input_path))
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        records = records[: args.max_samples]
    if completed > len(records):
        raise ValueError("Output has more records than the selected input range")
    pending = records[completed:]
    if not pending:
        print(f"Nothing to do; output already contains {completed} records.")
        return 0

    if args.offline:
        # The environment variable is read by transformers before model load.
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"

    translator = NLLBBackTranslator(
        model_name=args.model_name,
        source_lang=args.source_lang,
        pivot_lang=args.pivot_lang,
        device=args.device,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    translated_fields: dict[str, list[str]] = {}
    for field in args.fields:
        missing = [i for i, record in enumerate(pending) if field not in record]
        if missing:
            raise KeyError(f"Field '{field}' is missing from pending record {missing[0]}")
        texts = [str(record[field]) for record in pending]
        print(f"Translating {field}: {len(texts)} records", file=sys.stderr)
        translated_fields[field] = translator.translate(texts, args.batch_size)

    augmented = make_augmented_records(
        pending,
        translated_fields,
        start_index=completed,
        pivot_lang=args.pivot_lang,
        include_metadata=not args.no_metadata,
    )
    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        write_records(handle, augmented)
    print(f"Wrote {len(augmented)} augmented records to {output_path}")
    return 0


if __name__ == "__main__":
    main()
