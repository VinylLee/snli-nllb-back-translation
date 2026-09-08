# Real Policy V2 Calibration Report

## Configuration

- source sample: `data/nli/calibration/snli_calibration_800_sources.jsonl` (800 records, seed 42)

- pivot: `fra_Latn`; augmentation mode: `separate`

- translation model: `facebook/nllb-200-distilled-600M`

- translation model revision: `f8d333a098d19b4fd9a8b18f94170487ad3f821d`

- verifier: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`

- verifier revision: `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`

- num_beams: 2; do_sample: false; max_input_tokens: 128; max_new_tokens: 64

- source NLI gate: on; source NLI threshold: 0.80; semantic threshold: 0.90; candidate NLI threshold: 0.80

- batch_size: 16; filter_batch_size: 32; chunk_size: 32; allow_truncation: false

- device: CPU; dtype: float32 (CUDA run was unavailable because the installed PyTorch CUDA runtime required a newer driver)


## Run summary

- source total: 800
- candidate total: 1600
- translation performed: 1336
- translation skipped: 264

- accepted: 774
- rejected: 826
- overall acceptance: 48.38%
- accepted / translated eligible: 57.93%


## Source gate

| status | source count | candidate count |
| --- | ---: | ---: |

| RELIABLE_GOLD | 668 | 1336 |

| GOLD_LOW_CONFIDENCE | 32 | 64 |

| GOLD_DISAGREEMENT | 100 | 200 |


## By label

| label | total | accepted | eligible | overall rate | eligible rate |
| --- | ---: | ---: | ---: | ---: | ---: |

| entailment | 534 | 287 | 470 | 53.75% | 61.06% |

| neutral | 532 | 226 | 414 | 42.48% | 54.59% |

| contradiction | 534 | 261 | 452 | 48.88% | 57.74% |


## By field

| field | total | accepted | eligible | overall rate | eligible rate |
| --- | ---: | ---: | ---: | ---: | ---: |

| premise | 800 | 383 | 668 | 47.88% | 57.34% |

| hypothesis | 800 | 391 | 668 | 48.88% | 58.53% |


## Rejection reasons

| reason | count |
| --- | ---: |

| generation_truncated | 4 |

| label_flip | 83 |

| low_nli_confidence | 26 |

| numeric_value_changed | 8 |

| semantic_drift | 326 |

| source_label_disagreement | 200 |

| source_nli_low_confidence | 64 |

| trivial_copy | 193 |


## Offline vs real

- offline V2 accepted: 774
- real V2 accepted: 774
- matching decisions: 1592
- different decisions: 8


Difference categories:

| category | count |
| --- | ---: |

| source-gate difference | 6 |

| translation difference | 2 |


## Canary examples

### Canary A (148356)
- original: A red-haired rides a go-cart.
- new BT: There's a red one in a go-kart.
- semantic forward/backward/min: 0.9921894073486328 / 0.9404593706130981 / 0.9404593706130981
- candidate NLI: contradiction (gold probability 0.9992294311523438)
- accepted: True
- reasons: []

### Canary B (227619)
- original: The two men have just been stuck in concrete boots and are drowning in the Hudson River.
- new BT: Both men are trapped in concrete boots and drowned in the Hudson River.
- semantic forward/backward/min: 0.9881734251976013 / 0.9715660214424133 / 0.9715660214424133
- candidate NLI: contradiction (gold probability 0.9988000392913818)
- accepted: True
- reasons: []

### Canary C (512053)
- original: Nobody is performing.
- new BT: No one's playing.
- semantic forward/backward/min: 0.9962934851646423 / 0.9958156943321228 / 0.9958156943321228
- candidate NLI: contradiction (gold probability 0.9997307658195496)
- accepted: True
- reasons: []


## Limitations

This report uses the real rerun outputs and the frozen source sample. The old offline simulation used the previous calibration translation/verifier scores and cannot observe generation termination; differences are therefore not expected to be zero. No full SNLI run was started.
