# Original NLI Baseline Report

## Original SNLI verifier baseline

| subset | total | agrees with gold | disagrees | accuracy |
| --- | --- | --- | --- | --- |
| overall | 800 | 700 | 100 | 87.50% |
| entailment | 267 | 246 | 21 | 92.13% |
| neutral | 266 | 219 | 47 | 82.33% |
| contradiction | 267 | 235 | 32 | 88.01% |

Verifier model revision: `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. NLI threshold remains `0.80`; no filtering decision was changed.


## Candidate transitions

| transition | count | percentage |
| --- | --- | --- |
| STABLE_GOLD | 1293 | 80.81% |
| TRUE_FLIP | 107 | 6.69% |
| VERIFIER_GOLD_DISAGREEMENT_STABLE | 163 | 10.19% |
| RECOVERED_TO_GOLD | 30 | 1.88% |
| NON_GOLD_TRANSITION | 7 | 0.44% |

### By gold label

| label | transition | count |
| --- | --- | --- |
| entailment | STABLE_GOLD | 446 |
| entailment | TRUE_FLIP | 46 |
| entailment | VERIFIER_GOLD_DISAGREEMENT_STABLE | 33 |
| entailment | RECOVERED_TO_GOLD | 7 |
| entailment | NON_GOLD_TRANSITION | 2 |
| neutral | STABLE_GOLD | 398 |
| neutral | TRUE_FLIP | 40 |
| neutral | VERIFIER_GOLD_DISAGREEMENT_STABLE | 76 |
| neutral | RECOVERED_TO_GOLD | 14 |
| neutral | NON_GOLD_TRANSITION | 4 |
| contradiction | STABLE_GOLD | 449 |
| contradiction | TRUE_FLIP | 21 |
| contradiction | VERIFIER_GOLD_DISAGREEMENT_STABLE | 54 |
| contradiction | RECOVERED_TO_GOLD | 9 |
| contradiction | NON_GOLD_TRANSITION | 1 |

### By augmented field

| field | transition | count |
| --- | --- | --- |
| premise | STABLE_GOLD | 648 |
| premise | TRUE_FLIP | 52 |
| premise | VERIFIER_GOLD_DISAGREEMENT_STABLE | 84 |
| premise | RECOVERED_TO_GOLD | 13 |
| premise | NON_GOLD_TRANSITION | 3 |
| hypothesis | STABLE_GOLD | 645 |
| hypothesis | TRUE_FLIP | 55 |
| hypothesis | VERIFIER_GOLD_DISAGREEMENT_STABLE | 79 |
| hypothesis | RECOVERED_TO_GOLD | 17 |
| hypothesis | NON_GOLD_TRANSITION | 4 |

## Existing label_flip decomposition

| category | count | percentage of label_flip |
| --- | --- | --- |
| STABLE_GOLD | 0 | 0.00% |
| TRUE_FLIP | 107 | 38.63% |
| VERIFIER_GOLD_DISAGREEMENT_STABLE | 163 | 58.84% |
| RECOVERED_TO_GOLD | 0 | 0.00% |
| NON_GOLD_TRANSITION | 7 | 2.53% |

Existing label_flip total: 277.


## Candidate-original prediction stability

| subset | total | same prediction | stability |
| --- | --- | --- | --- |
| all candidates | 1600 | 1456 | 91.00% |
| accepted | 804 | 789 | 98.13% |
| rejected | 796 | 667 | 83.79% |

## Confidence transitions

| metric | value |
| --- | --- |
| mean candidate-original gold probability delta | -0.042557 |
| large confidence drop (delta <= -0.15) | 145 |
| large confidence increase (delta >= +0.15) | 58 |
| confidence_drop flag (STABLE_GOLD and delta < -0.15) | 26 |

## Review package

| file | rows |
| --- | --- |
| MASTER_REVIEW.csv | 406 |
| number_cue_audit.csv | 34 |
| review_accepted_high_confidence.csv | 75 |
| review_accepted_score_borderline.csv | 59 |
| review_negation_cue.csv | 11 |
| review_semantic_drift.csv | 75 |
| review_true_flip.csv | 100 |
| review_verifier_gold_disagreement.csv | 75 |

All review labels and verdict columns are intentionally blank for GPT/human annotation.


## Configuration and provenance

| item | value |
| --- | --- |
| calibration repository commit | 9015a4d |
| sample seed | 42 |
| verifier model | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli |
| verifier revision | 6f5cf0a2b59cabb106aca4c287eed12e357e90eb |
| device | cpu |
| batch size | 32 |
| wall-clock baseline seconds | 31.75 |
| NLI threshold | 0.80 |

The original baseline was run on the 800 committed source pairs only; NLLB was not run in this phase.
