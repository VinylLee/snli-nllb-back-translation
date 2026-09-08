# SNLI NLLB back-translation augmentation

This project augments SNLI with `facebook/nllb-200-distilled-600M` and an
explainable quality gate:

```text
candidate generation -> batched bidirectional entailment -> batched pair NLI
-> hard/soft cue and corruption checks -> accepted or rejected JSONL
```

The recommended mode is `separate`, which creates two asymmetric candidates
per source item: `(BT(premise), hypothesis)` and `(premise, BT(hypothesis))`.
`both` creates one two-sided candidate; `premise` and `hypothesis` create one-
sided output. The old `--fields premise` option is replaced by
`--augmentation-mode premise`.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended quality-filtered run:

```bash
python scripts/back_translate_nllb.py \
  --input data/nli/original_dataset/snli/train.json \
  --accepted-output data/nli/back_translated/snli_train_fra.accepted.jsonl \
  --rejected-output data/nli/back_translated/snli_train_fra.rejected.jsonl \
  --pivot-lang fra_Latn \
  --augmentation-mode separate \
  --quality-filter on \
  --semantic-threshold 0.80 \
  --nli-threshold 0.80 \
  --filter-batch-size 32 \
  --chunk-size 64 \
  --batch-size 16 \
  --device cuda:0 \
  --dtype float16
```

Quality filtering is on by default and uses
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. The `0.80` thresholds are
engineering starting points, not calibrated optima. Semantic checks require
both original-to-candidate and candidate-to-original entailment. Pair-level
NLI must retain the gold label with sufficient probability. Verifier calls are
batched per chunk and internally respect `--filter-batch-size`.

Logical cues are split into two policies. Negation and numbers are hard cues:
changes reject a candidate. Numeric literals are open-ended, including
integers such as `11` and `100` and decimals such as `3.5`; common number words
are also tracked. Quantifiers, modals, time, and space are soft cues: changes
are recorded in `soft_cue_changes` and passed to semantic/NLI verification.

Truncation metadata distinguishes detected overflow from actual tokenizer
truncation. Without `--allow-truncation`, an over-limit source is rejected as
`input_too_long` and an over-limit generated pivot is rejected as
`intermediate_input_too_long`; neither is marked truncated. With the flag,
both stages may truncate and continue to quality filtering. Metadata records
`source_to_pivot_over_limit`, `source_to_pivot_truncated`,
`pivot_to_source_over_limit`, `pivot_to_source_truncated`, and `was_truncated`.
In `separate` mode these flags belong only to the translated field; `both`
combines the two fields by OR, while `was_truncated` reflects actual
truncation only.

Accepted and rejected outputs contain stable IDs of the form
`source_index:augmented_field:pivot_lang`, scores, hard/soft cue changes,
reasons, truncation metadata, and model/generation provenance. JSONL input is
processed in chunks and each output is flushed. JSON arrays are supported as a
fallback but are parsed into memory, so JSONL is preferred for large data.
`--resume` reads candidate IDs from both outputs, so rejected candidates are
also considered complete. `--no-metadata` cannot be combined with `--resume`;
the command fails with `--resume requires candidate metadata` rather than
silently regenerating candidates.

For a small offline generation after NLLB is cached:

```bash
python scripts/back_translate_nllb.py \
  --input data/nli/original_dataset/snli/train.json \
  --output data/nli/back_translated/snli_smoke.accepted.jsonl \
  --max-samples 18 --quality-filter off --offline \
  --augmentation-mode separate --device cpu --dtype float32
```

`--quality-filter off` preserves legacy accept-all behavior except for
structural and input-length failures. With deterministic beam search
(`do_sample=false`), `--seed` does not create beam diversity.

## Data and tests

Original SNLI files are under `data/nli/original_dataset/snli/` and are not
modified. Run the model-free test suite with:

```bash
python -m unittest discover -s tests -v
```
