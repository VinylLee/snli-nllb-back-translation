# Semantic Verifier Bake-off Report

Analysis-only. No NLLB generation or production-policy decision was changed.

## Models

- Current semantic baseline: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, revision 6f5cf0a2b59cabb106aca4c287eed12e357e90eb; existing V2 semantic scores reused for 1,336 translated candidates.
- STS cross-encoder: cross-encoder/stsb-roberta-base, revision d576534b67143e2c70ee9966d7fdbf5835728d13; CPU runtime 11.64s.
- Independent NLI: FacebookAI/roberta-large-mnli, revision 2a8f12d27941090092df78e4ba6f0928eb5eac98; CPU runtime 70.85s.
- BART MNLI: not run (optional model; no result is inferred).

## Expert-anchor pairwise ranking

| model | PASS > FAIL pairwise score |
| --- | ---: |
| Current DeBERTa | 0.9500 |
| STS | 0.9750 |
| RoBERTa MNLI | 1.0000 |

## Anchor scores

| id | verdict | Current DeBERTa min | STS | RoBERTa min |
| --- | --- | ---: | ---: | ---: |
| F1 | FAIL | 0.9404 | 0.8051 | 0.9818 |
| F2 | FAIL | 0.9717 | 0.9634 | 0.9887 |
| F3 | FAIL | 0.9956 | 0.5978 | 0.9814 |
| F4 | FAIL | 0.0002 | 0.3959 | 0.0006 |
| F5 | FAIL | 0.1976 | 0.8459 | 0.9682 |
| F6 | FAIL | 0.0005 | 0.2183 | 0.6746 |
| F7 | FAIL | 0.8687 | 0.9120 | 0.8293 |
| F8 | FAIL | 0.8193 | 0.9914 | 0.9037 |
| P1 | PASS | 0.9971 | 0.9966 | 0.9931 |
| P2 | PASS | 0.9951 | 0.9786 | 0.9926 |
| P3 | PASS | 0.9932 | 0.9967 | 0.9932 |
| P4 | PASS | 0.9966 | 0.9964 | 0.9936 |
| P5 | PASS | 0.9971 | 0.9962 | 0.9925 |

## Full translated-candidate distributions

| subset | metric | count | mean | median | p05 | p10 | p25 | p75 | p90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accepted | current_deberta_min | 774 | 0.9896 | 0.9947 | 0.9604 | 0.9769 | 0.9901 | 0.9965 | 0.9973 |
| accepted | STS | 774 | 0.9697 | 0.9961 | 0.8603 | 0.9058 | 0.9681 | 0.9967 | 0.9968 |
| accepted | RoBERTa_min | 774 | 0.9661 | 0.9916 | 0.9017 | 0.9565 | 0.9843 | 0.9930 | 0.9936 |
| rejected | current_deberta_min | 562 | 0.5283 | 0.6441 | 0.0006 | 0.0012 | 0.0154 | 0.9952 | 0.9968 |
| rejected | STS | 562 | 0.8721 | 0.9248 | 0.5420 | 0.6263 | 0.8143 | 0.9966 | 0.9968 |
| rejected | RoBERTa_min | 562 | 0.6670 | 0.9386 | 0.0023 | 0.0100 | 0.1989 | 0.9927 | 0.9937 |

## Semantic-drift rejection distributions

| subset | metric | count | mean | median | p75 | p90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| semantic_drift rejected | STS | 326 | 0.7936 | 0.8457 | 0.9116 | 0.9630 |
| semantic_drift rejected | RoBERTa_min | 326 | 0.4399 | 0.3739 | 0.8694 | 0.9638 |

## Threshold simulation on current accepted candidates

Analysis only; these thresholds do not change production decisions.

| model | threshold | retained | newly rejected | retention | FAIL rejected | PASS retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| STS | 0.70 | 767 | 7 | 99.10% | 3 | 5 |
| STS | 0.75 | 764 | 10 | 98.71% | 3 | 5 |
| STS | 0.80 | 758 | 16 | 97.93% | 3 | 5 |
| STS | 0.85 | 738 | 36 | 95.35% | 5 | 5 |
| STS | 0.90 | 705 | 69 | 91.09% | 5 | 5 |
| STS | 0.95 | 623 | 151 | 80.49% | 6 | 5 |
| RoBERTa min entailment | 0.70 | 756 | 18 | 97.67% | 2 | 5 |
| RoBERTa min entailment | 0.80 | 749 | 25 | 96.77% | 2 | 5 |
| RoBERTa min entailment | 0.85 | 745 | 29 | 96.25% | 3 | 5 |
| RoBERTa min entailment | 0.90 | 735 | 39 | 94.96% | 3 | 5 |
| RoBERTa min entailment | 0.95 | 703 | 71 | 90.83% | 4 | 5 |

## Canary ranks among translated candidates

Percentile is the share of translated candidates scoring at or below the canary.

| source_index | STS percentile | RoBERTa percentile |
| --- | ---: | ---: |
| 148356 | 11.3024th | 36.9760th |
| 227619 | 37.3503th | 46.9311th |
| 512053 | 4.0419th | 36.6766th |

## Secondary disagreements

- Master review candidates scored: 406.
- Secondary disagreement candidates exported: 31 (capped at 150).
- review_secondary_disagreement.csv selects current-policy accepted candidates with current semantic min >= 0.95 and STS < 0.80 or RoBERTa min < 0.80.

## Limitations

- Secondary scores are analysis columns only; production filtering was not changed.
- BART MNLI was not run.
- The 13-anchor ranking is a sanity check, not a formal benchmark.
- STS uses one native CrossEncoder sigmoid activation for its one-logit regression head.
- No NLLB generation, full-SNLI augmentation, or threshold change was performed.
