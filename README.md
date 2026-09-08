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
  --source-nli-gate on \
  --source-nli-threshold 0.80 \
  --semantic-threshold 0.90 \
  --nli-threshold 0.80 \
  --filter-batch-size 32 \
  --chunk-size 64 \
  --batch-size 16 \
  --device cuda:0 \
  --dtype float16
```

Quality filtering is on by default and uses
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. The candidate NLI threshold `0.80` remains an engineering starting point; the default semantic threshold is now `0.90` as a conservative calibration-informed candidate, not a calibrated optimum. Semantic checks require
both original-to-candidate and candidate-to-original entailment. Pair-level
NLI must retain the gold label with sufficient probability. Verifier calls are
batched per chunk and internally respect `--filter-batch-size`.

Logical cue changes are primarily diagnostic flags. Negation forms such as `not`/`n't` and `nobody`/`no one` are canonicalized, as are equivalent number forms such as `5`/`five`. Only a reliable explicit numeric value conflict is an independent hard failure; other negation, number, quantifier, modal, time, and space changes are passed to semantic/NLI verification.

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

If an input record contains a nonnegative integer `source_index`, it is
preserved as the stable dataset provenance index and is used in the candidate
ID. Records without one use their zero-based input position. Every output
also records `input_position`; invalid or duplicate explicit indices are
reported rather than silently cast or reused.

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

## Quality Policy v2

The current conservative production profile enables the original-pair NLI source gate (`--source-nli-gate on`, threshold `0.80`) before NLLB generation. Only `RELIABLE_GOLD` sources are augmented. Candidate NLI preservation remains at `0.80`, while the default bidirectional semantic threshold is `0.90`; this is a calibration-informed conservative candidate, not a proven optimum.

Logical cue changes are primarily diagnostic flags. Negation forms such as `not`/`n't` and `nobody`/`no one` are canonicalized, as are equivalent number forms such as `5`/`five`; only a reliable explicit numeric value conflict is an independent hard failure. Decoder EOS/max-new-token cutoffs are detected in both translation stages and rejected as `generation_truncated`.

The existing calibration audit can be reclassified offline with `python scripts/simulate_policy_v2.py`. Historical candidates do not contain decoder EOS/max-new-token metadata, so that simulation cannot apply the generation-truncation gate.

## Secondary semantic verifier bake-off

The analysis-only bake-off is run with
`python scripts/bakeoff_semantic_verifiers.py --device cpu --batch-size 32`.
It reads the frozen V2 audit and review artifacts, scores
`cross-encoder/stsb-roberta-base` and `FacebookAI/roberta-large-mnli`, and
writes anchor, score, threshold-simulation, and disagreement artifacts. It
does not call NLLB or alter production decisions. The optional BART model is
not required.

## Data and tests

Original SNLI files are under `data/nli/original_dataset/snli/` and are not
modified. Run the model-free test suite with:

```bash
python -m unittest discover -s tests -v
```
