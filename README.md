# SNLI back translation augmentation

The SNLI files are under `data/nli/original_dataset/snli/`.  The augmentation
script uses `facebook/nllb-200-distilled-600M` to translate each NLI text from
English to a pivot language and back to English, while preserving the label.

Install dependencies:

```bash
pip install -r requirements.txt
```

Example using French as the pivot language:

```bash
python scripts/back_translate_nllb.py \
  --input data/nli/original_dataset/snli/train.json \
  --output data/nli/back_translated/snli_train_fra.jsonl \
  --pivot-lang fra_Latn \
  --device cuda:0 \
  --dtype float16 \
  --batch-size 16
```

For an offline run, download/cache the model first and add `--offline`.  A
smaller smoke test can use `--max-samples 100`.  Existing output is protected;
use `--resume` to continue a partial JSONL output or `--overwrite` to replace
it.  Use `--no-metadata` if downstream code requires exactly the original
`premise`, `hypothesis`, and `label` fields.

Useful NLLB language codes include `fra_Latn` (French), `deu_Latn` (German),
`zho_Hans` (Simplified Chinese), and `spa_Latn` (Spanish).
