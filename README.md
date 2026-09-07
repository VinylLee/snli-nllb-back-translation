# SNLI NLLB back-translation augmentation

This project augments SNLI with `facebook/nllb-200-distilled-600M`. The
recommended pipeline generates asymmetric candidates, then applies:

```text
candidate generation -> bidirectional semantic check -> NLI label check
-> logical-cue/corruption checks -> accepted or rejected JSONL
```

The verifier is `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`; its label order
is read from the model config rather than assumed. SNLI files are copied under
`data/nli/original_dataset/snli/`. Original train/validation/test files are not
modified.

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
  --batch-size 16 \
  --filter-batch-size 32 \
  --chunk-size 64 \
  --device cuda:0 \
  --dtype float16
```

`separate` is the default and produces two candidates per source item:
`(BT(premise), hypothesis)` and `(premise, BT(hypothesis))`. Use
`--augmentation-mode both` for the legacy two-sided candidate, or `premise` /
`hypothesis` for one-sided generation.

Quality filtering is on by default. It requires both directions of sentence
entailment to exceed `--semantic-threshold`, and the augmented pair's predicted
label and gold probability to pass `--nli-threshold`. It also rejects logical
cue changes (negation, quantity, modality, time, and space), empty/corrupt
outputs, and near-copy candidates below `--min-change-ratio`. Length ratios are
recorded as risk flags. Inputs over `--max-input-tokens` are rejected as
`input_too_long`; pass `--allow-truncation` only when that tradeoff is explicit.

Both output files contain provenance, candidate IDs, scores, flags, and reasons.
When only `--output` is supplied, it is the accepted path and a sibling
`.rejected.jsonl` path is created automatically. `--resume` uses candidate IDs
from both files, so rejected candidates count as processed. Each chunk is
flushed before the next chunk is read.

For a small offline regression after the two models are cached:

```bash
python scripts/back_translate_nllb.py \
  --input data/nli/original_dataset/snli/train.json \
  --output data/nli/back_translated/snli_train_smoke.accepted.jsonl \
  --max-samples 18 --quality-filter on --offline \
  --device cpu --dtype float32 --batch-size 4
```

`--quality-filter off` preserves the simple legacy acceptance behavior. With
deterministic beam search (`do_sample=False`), `--seed` does not create beam
search diversity; use multiple pivots only after the quality gate is calibrated
on a manually reviewed set.

Run tests without downloading models:

```bash
python -m unittest discover -s tests -v
```
