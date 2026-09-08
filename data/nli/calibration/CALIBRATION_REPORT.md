# SNLI Calibration Report

## Configuration

| item | value |
| --- | --- |
| source commit | dfec9cf |
| pipeline commit | 5d7b3c3 |
| sample seed | 42 |
| source count | 800 |
| unique premise count | 800 |
| sampling strategy | label-balanced random traversal, global unique premise first |
| translation model | facebook/nllb-200-distilled-600M |
| translation revision | f8d333a098d19b4fd9a8b18f94170487ad3f821d |
| verifier model | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli |
| verifier revision | 6f5cf0a2b59cabb106aca4c287eed12e357e90eb |
| pivot language | fra_Latn |
| augmentation mode | separate |
| num_beams | 2 |
| max_input_tokens | 128 |
| max_new_tokens | 64 |
| semantic threshold | 0.8 |
| NLI threshold | 0.8 |
| change threshold | 0.03 |
| allow truncation | False |
| device | cpu |
| batch size | 16 |
| filter batch size | 32 |
| chunk size | 32 |
| wall-clock seconds | 1055.56 |

## Overall results

| metric | value |
| --- | --- |
| source_count | 800 |
| candidate_count | 1600 |
| accepted | 804 |
| rejected | 796 |
| acceptance_rate | 50.25% |

## Source label distribution

| label | source count |
| --- | --- |
| entailment | 267 |
| neutral | 266 |
| contradiction | 267 |

## Field distribution

| field | total | accepted | rejected | acceptance rate |
| --- | --- | --- | --- | --- |
| premise | 800 | 403 | 397 | 50.38% |
| hypothesis | 800 | 401 | 399 | 50.12% |

## Label distribution

| label | total | accepted | rejected | acceptance rate |
| --- | --- | --- | --- | --- |
| entailment | 534 | 297 | 237 | 55.62% |
| neutral | 532 | 245 | 287 | 46.05% |
| contradiction | 534 | 262 | 272 | 49.06% |

## Label × field

| label | field | total | accepted | rejected | acceptance rate |
| --- | --- | --- | --- | --- | --- |
| entailment | premise | 267 | 137 | 130 | 51.31% |
| entailment | hypothesis | 267 | 160 | 107 | 59.93% |
| neutral | premise | 266 | 128 | 138 | 48.12% |
| neutral | hypothesis | 266 | 117 | 149 | 43.98% |
| contradiction | premise | 267 | 138 | 129 | 51.69% |
| contradiction | hypothesis | 267 | 124 | 143 | 46.44% |

## Rejection reasons

| reason | count |
| --- | --- |
| hard_cue_changed | 42 |
| label_flip | 277 |
| low_nli_confidence | 56 |
| semantic_drift | 365 |
| trivial_copy | 232 |

Primary reason is the first reason in the pipeline's reason list.

| primary reason | count |
| --- | --- |
| hard_cue_changed | 42 |
| label_flip | 140 |
| low_nli_confidence | 33 |
| semantic_drift | 349 |
| trivial_copy | 232 |

## Cue statistics

| cue type | candidate count |
| --- | --- |
| negation | 11 |
| number | 34 |
| quantifier | 43 |
| modal | 3 |
| time | 11 |
| space | 381 |


| soft cue status | total | accepted | acceptance rate |
| --- | --- | --- | --- |
| soft-cue changed | 422 | 228 | 54.03% |
| no soft-cue change | 1178 | 576 | 48.90% |

## Score distributions

| subset | metric | distribution |
| --- | --- | --- |
| accepted | semantic_min | mean=0.9858; median=0.9944; p10=0.9679; p25=0.9891; p50=0.9944; p75=0.9964; p90=0.9973 |
| accepted | nli_gold_probability | mean=0.9860; median=0.9967; p10=0.9587; p25=0.9895; p50=0.9967; p75=0.9990; p90=0.9994 |
| rejected | semantic_min | mean=0.5978; median=0.9719; p10=0.0016; p25=0.0233; p50=0.9719; p75=0.9957; p90=0.9969 |
| rejected | nli_gold_probability | mean=0.6544; median=0.9670; p10=0.0061; p25=0.1265; p50=0.9670; p75=0.9977; p90=0.9993 |

## Borderline accepted

290 candidates matched the borderline criteria (36.07% of accepted); 150 exported to `accepted_borderline.csv`.


## Number cue observations

Number-cue rejected candidate count: 34.

| number cue change | count |
| --- | --- |
| {"one":1} -> {} | 10 |
| {} -> {"one":1} | 9 |
| {"two":1} -> {} | 2 |
| {"one":1,"three":1} -> {"three":1} | 2 |
| {"two":2} -> {"two":1} | 1 |
| {"5":1} -> {"five":1} | 1 |
| {"3":1,"1":1} -> {"one":2,"three":1} | 1 |
| {"one":2,"two":1} -> {"one":1,"two":1} | 1 |
| {"3":1} -> {"three":1} | 1 |
| {"one":1,"two":1} -> {"two":1} | 1 |
| {"one":2,"two":1} -> {"two":1} | 1 |
| {"three":1} -> {"one":1,"three":1} | 1 |
| {"2":1} -> {"two":1} | 1 |
| {"couple":1} -> {"two":1} | 1 |
| {"couple":1} -> {} | 1 |

## Acceptance imbalance

Label acceptance-rate range: 9.57 percentage points; field range: 0.26 percentage points. These are reported observations only; no threshold was changed.


## Review artifacts

All review CSVs reserve `human_semantic_equivalent`, `human_label_preserved`, `human_useful_augmentation`, and `human_notes` as empty columns.


## Human annotation standard

human_semantic_equivalent: yes, no, or borderline; judge whether the back-translated sentence is an approximately equivalent paraphrase.

human_label_preserved: yes, no, or uncertain; judge whether the augmented pair retains the original gold NLI label.

human_useful_augmentation: yes, no, or borderline; judge whether the example is natural, sufficiently changed, and useful for training.

human_notes: free-form reviewer context.


## Observations

- number hard-cue changes occur in rejected candidates and require human review for lexical false positives


Thresholds remain fixed at semantic=0.80 and NLI=0.80; this report is for human calibration, not automatic threshold tuning.
