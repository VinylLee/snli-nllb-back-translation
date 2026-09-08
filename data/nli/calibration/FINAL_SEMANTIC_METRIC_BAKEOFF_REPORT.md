# Final Semantic Metric Bake-off Report

Analysis-only. No NLLB, full-SNLI run, production-policy change, or threshold change was performed.

## Models

- Current DeBERTa semantic baseline: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, revision 6f5cf0a2b59cabb106aca4c287eed12e357e90eb; existing V2 scores reused.
- STS-base: cross-encoder/stsb-roberta-base; existing scores reused.
- STS-large: cross-encoder/stsb-roberta-large, revision 2b12c2c0088918e76151fd5937b7bba986ef1f98; CPU runtime 39.25s.
- BLEURT: Elron/bleurt-base-512, revision 4f4abeeba7c29ded45fc90b8a66eb49c8569f587; CPU raw-score runtime 12.71s.
- BLEURT primary fallback note: ValueError: The checkpoint you are trying to load has model type `bleurt` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date. You can upda.
- RoBERTa-MNLI: existing scores reused; BART-MNLI: not run.

## Expert-anchor performance

| metric | PASS > FAIL pairwise ranking |
| --- | ---: |
| Current DeBERTa | 0.7778 |
| STS-base | 0.5079 |
| STS-large | 0.7698 |
| BLEURT | 0.6508 |
| RoBERTa-MNLI | 0.4127 |

## Expert anchor scores

| anchor | verdict | Current DeBERTa min | STS-base | STS-large | BLEURT raw | RoBERTa MNLI min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| F1 | FAIL | 0.9404 | 0.8051 | 0.7982 | -0.7591 | 0.9818 |
| F2 | FAIL | 0.9717 | 0.9634 | 0.9269 | 0.6399 | 0.9887 |
| F3 | FAIL | 0.9956 | 0.5978 | 0.5132 | 0.3050 | 0.9814 |
| F4 | FAIL | 0.0002 | 0.3959 | 0.4292 | -0.5699 | 0.0006 |
| F5 | FAIL | 0.1976 | 0.8459 | 0.7425 | 0.4128 | 0.9682 |
| F6 | FAIL | 0.0005 | 0.2183 | 0.3803 | 0.3302 | 0.6746 |
| F7 | FAIL | 0.8687 | 0.9120 | 0.8959 | 0.5228 | 0.8293 |
| F8 | FAIL | 0.8193 | 0.9914 | 0.9433 | 0.5331 | 0.9037 |
| P1 | PASS | 0.9971 | 0.9966 | 0.9633 | 1.0287 | 0.9931 |
| P2 | PASS | 0.9951 | 0.9786 | 0.9670 | 0.7561 | 0.9926 |
| P3 | PASS | 0.9932 | 0.9967 | 0.9654 | 0.5912 | 0.9932 |
| P4 | PASS | 0.9966 | 0.9964 | 0.9680 | 1.0403 | 0.9936 |
| P5 | PASS | 0.9971 | 0.9962 | 0.9691 | 0.8037 | 0.9925 |

## Canary scores and ranks among all 1,336 translated candidates

| source_index | metric | score | percentile at-or-below |
| --- | --- | ---: | ---: |
| 148356 | Current DeBERTa | 0.9405 | 26.12th |
| 148356 | STS-base | 0.8051 | 11.30th |
| 148356 | STS-large | 0.7982 | 12.72th |
| 148356 | BLEURT | -0.7591 | 0.52th |
| 148356 | RoBERTa-MNLI | 0.9818 | 36.98th |
| 227619 | Current DeBERTa | 0.9716 | 29.72th |
| 227619 | STS-base | 0.9634 | 37.35th |
| 227619 | STS-large | 0.9269 | 29.94th |
| 227619 | BLEURT | 0.6399 | 43.79th |
| 227619 | RoBERTa-MNLI | 0.9887 | 46.93th |
| 512053 | Current DeBERTa | 0.9958 | 70.13th |
| 512053 | STS-base | 0.5978 | 4.04th |
| 512053 | STS-large | 0.5132 | 2.02th |
| 512053 | BLEURT | 0.3050 | 16.92th |
| 512053 | RoBERTa-MNLI | 0.9814 | 36.68th |

## Full translated-candidate distributions

| subset | metric | count | mean | median | p05 | p10 | p25 | p75 | p90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current accepted | current | 774 | 0.9896 | 0.9947 | 0.9604 | 0.9769 | 0.9901 | 0.9965 | 0.9973 |
| current accepted | sts_base | 774 | 0.9697 | 0.9961 | 0.8603 | 0.9058 | 0.9681 | 0.9967 | 0.9968 |
| current accepted | sts_large | 774 | 0.9498 | 0.9667 | 0.8625 | 0.9094 | 0.9563 | 0.9697 | 0.9714 |
| current accepted | bleurt | 774 | 0.6723 | 0.7264 | 0.1506 | 0.3396 | 0.5696 | 0.8573 | 0.9349 |
| current rejected | current | 562 | 0.5283 | 0.6441 | 0.0006 | 0.0012 | 0.0154 | 0.9952 | 0.9968 |
| current rejected | sts_base | 562 | 0.8721 | 0.9248 | 0.5420 | 0.6263 | 0.8143 | 0.9966 | 0.9968 |
| current rejected | sts_large | 562 | 0.8535 | 0.9171 | 0.5334 | 0.6328 | 0.7834 | 0.9674 | 0.9703 |
| current rejected | bleurt | 562 | 0.5542 | 0.5903 | -0.3811 | -0.1529 | 0.2662 | 1.0270 | 1.0777 |

## Semantic-drift rejection distributions

| subset | metric | count | mean | median | p75 | p90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| semantic_drift rejected | sts_base | 326 | 0.7936 | 0.8457 | 0.9116 | 0.9630 |
| semantic_drift rejected | sts_large | 326 | 0.7787 | 0.8166 | 0.8928 | 0.9443 |
| semantic_drift rejected | bleurt | 326 | 0.2915 | 0.3795 | 0.5879 | 0.7611 |

## STS threshold simulation on expert-reviewed disagreement set

| model | threshold | FAIL caught | FAIL recall | PASS falsely rejected | BORDERLINE rejected | PASS/FAIL rejection precision | accepted retained | newly rejected | retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STS-base | 0.60 | 2/18 | 11.11% | 1/7 | 0/6 | 66.67% | 771 | 3 | 99.61% |
| STS-base | 0.65 | 3/18 | 16.67% | 1/7 | 1/6 | 75.00% | 769 | 5 | 99.35% |
| STS-base | 0.70 | 3/18 | 16.67% | 2/7 | 2/6 | 60.00% | 767 | 7 | 99.10% |
| STS-base | 0.75 | 6/18 | 33.33% | 2/7 | 2/6 | 75.00% | 764 | 10 | 98.71% |
| STS-base | 0.80 | 8/18 | 44.44% | 5/7 | 3/6 | 61.54% | 758 | 16 | 97.93% |
| STS-base | 0.85 | 12/18 | 66.67% | 5/7 | 3/6 | 70.59% | 738 | 36 | 95.35% |
| STS-base | 0.90 | 15/18 | 83.33% | 5/7 | 3/6 | 75.00% | 705 | 69 | 91.09% |
| STS-base | 0.95 | 18/18 | 100.00% | 5/7 | 5/6 | 78.26% | 623 | 151 | 80.49% |
| STS-large | 0.60 | 2/18 | 11.11% | 0/7 | 0/6 | 100.00% | 772 | 2 | 99.74% |
| STS-large | 0.65 | 2/18 | 11.11% | 0/7 | 0/6 | 100.00% | 772 | 2 | 99.74% |
| STS-large | 0.70 | 3/18 | 16.67% | 1/7 | 0/6 | 75.00% | 768 | 6 | 99.22% |
| STS-large | 0.75 | 4/18 | 22.22% | 1/7 | 1/6 | 80.00% | 764 | 10 | 98.71% |
| STS-large | 0.80 | 8/18 | 44.44% | 2/7 | 2/6 | 80.00% | 755 | 19 | 97.55% |
| STS-large | 0.85 | 9/18 | 50.00% | 2/7 | 3/6 | 81.82% | 744 | 30 | 96.12% |
| STS-large | 0.90 | 17/18 | 94.44% | 2/7 | 3/6 | 89.47% | 707 | 67 | 91.34% |
| STS-large | 0.95 | 17/18 | 94.44% | 4/7 | 5/6 | 80.95% | 609 | 165 | 78.68% |

## BLEURT observed distributions

| subset | count | min | p05 | p10 | p25 | median | p75 | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expert PASS | 7 | -0.6688 | -0.3823 | -0.0959 | 0.3443 | 0.4155 | 0.6409 | 0.7655 | 0.8566 |
| expert FAIL | 18 | -0.2807 | -0.0557 | 0.0318 | 0.1045 | 0.2810 | 0.5044 | 0.6614 | 0.7647 |
| current V2 accepted | 774 | -0.8902 | 0.1506 | 0.3396 | 0.5696 | 0.7264 | 0.8573 | 0.9349 | 1.1670 |

BLEURT thresholds below are observed-score analysis points, not a fixed 01 grid.

| threshold | FAIL caught | FAIL recall | PASS falsely rejected | BORDERLINE rejected | accepted retained | newly rejected | retention |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -0.0160 | 1/18 | 5.56% | 1/7 | 1/6 | 759 | 15 | 98.06% |
| 0.1517 | 6/18 | 33.33% | 1/7 | 1/6 | 735 | 39 | 94.96% |
| 0.4025 | 12/18 | 66.67% | 3/7 | 1/6 | 666 | 108 | 86.05% |
| 0.6089 | 14/18 | 77.78% | 5/7 | 4/6 | 532 | 242 | 68.73% |
| 0.6919 | 17/18 | 94.44% | 5/7 | 6/6 | 423 | 351 | 54.65% |

## Combination-policy simulation

Combination rows are analysis-only. mode=either rejects when either metric is below threshold; mode=both rejects only when both are below threshold.
| policy | mode | STS threshold | BLEURT threshold | FAIL caught | PASS falsely rejected | accepted retained | retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| STS-base 0.75 + BLEURT observed PASS median | either | 0.75 | 0.4155 | 14/18 | 4/7 | 656/774 | 84.75% |
| STS-base 0.80 + BLEURT observed PASS median | either | 0.80 | 0.4155 | 14/18 | 6/7 | 653/774 | 84.37% |
| STS-base 0.75 + BLEURT observed PASS median | both | 0.75 | 0.4155 | 4/18 | 1/7 | 767/774 | 99.10% |
| STS-base 0.80 + BLEURT observed PASS median | both | 0.80 | 0.4155 | 6/18 | 2/7 | 764/774 | 98.71% |

## Current accepted model distributions

| metric | count | mean | median | p05 | p10 | p25 | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sts_base | 774 | 0.9697 | 0.9961 | 0.8603 | 0.9058 | 0.9681 | 0.9967 | 0.9968 |
| sts_large | 774 | 0.9498 | 0.9667 | 0.8625 | 0.9094 | 0.9563 | 0.9697 | 0.9714 |
| bleurt | 774 | 0.6723 | 0.7264 | 0.1506 | 0.3396 | 0.5696 | 0.8573 | 0.9349 |

## Known-risk examples

| source_index | candidate_id | original | candidate | STS-base | STS-large | BLEURT | current decision |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 517858 | 517858:premise:fra_Latn | Older female walking with cane next to flower wreaths. | An older female walks with a cane next to flower crowns. | 0.9477 | 0.8559 | 0.5326 | True |
| 517858 | 517858:hypothesis:fra_Latn | An older woman is walking with a cane. | An old woman walks with a cane. | 0.9967 | 0.9676 | 0.7383 | True |
| 277576 | 277576:premise:fra_Latn | A golfer wearing a striped red polo shirt and white pants watches his drive after swinging. | A golfer in a striped red polo and white pants looks at his car after swinging. | 0.7429 | 0.7732 | 0.6313 | True |
| 277576 | 277576:hypothesis:fra_Latn | The golfer is preparing food for dinner. | The golfer prepares food for dinner. | 0.9965 | 0.9668 | 0.8566 | True |
| 421310 | 421310:premise:fra_Latn | A person, whose face is blocked, is working on a large wooden loom. | A person with a blocked face works on a large wooden weaver. | 0.8344 | 0.8952 | 0.0522 | True |
| 421310 | 421310:hypothesis:fra_Latn | A person, whose face is blocked, is working on a wooden loom. | A person with a blocked face works on a wooden weaver. | 0.8438 | 0.8815 | 0.0943 | True |
| 215221 | 215221:premise:fra_Latn | A waveboarder skims through the surf at sunset. | A surfer crosses the wave at sunset. | 0.7624 | 0.8961 | 0.3403 | True |
| 215221 | 215221:hypothesis:fra_Latn | A person is in a waveboader competition. | A person is taking part in a waveboader competition. | 0.9950 | 0.9689 | 0.8538 | True |

## Decision-oriented analysis points

- S1 STS-base >= 0.75: 6/18 FAIL caught, 2/7 PASS lost, 764/774 current accepted retained.
- S2 STS-base >= 0.80: 8/18 FAIL caught, 5/7 PASS lost, 758/774 current accepted retained.
- Representative STS-large point 0.90: 17/18 FAIL caught, 2/7 PASS lost, 707/774 current accepted retained.
- Highest-FAIL-recall observed BLEURT point 0.6919: 17/18 FAIL caught, 5/7 PASS lost, 423/774 current accepted retained.
- Preservation-oriented STS-base+BLEURT representative (both mode, STS 0.75): 4/18 FAIL caught, 1/7 PASS lost, 767/774 current accepted retained.
- These are offline comparisons only and do not select or modify a production gate.

## Limitations and final recommendation

- The 31-row expert-reviewed set was sampled from secondary disagreement candidates and has selection bias; it is not a benchmark.
- BLEURT is reported as a raw regression score and is not comparable numerically to probability-like STS scores.
- No secondary metric was added to production in this round.
- Model C BART-MNLI was not run.
- Final recommendation: Option C — no additional general-purpose semantic metric is sufficient as a reliable production gate for the observed subtle drift; do not add STS-large or BLEURT to production in this round.
- Canary B is the decisive limitation: a general metric that keeps progressive `are drowning` → completed `drowned` high-confidence cannot provide the required guarantee. Next options are targeted tense/aspect or role guards, or explicit acceptance of residual noise.
- Final recommendation is based on targeted anchors, reviewed disagreement evidence, retention, and runtime; it is not threshold calibration.
