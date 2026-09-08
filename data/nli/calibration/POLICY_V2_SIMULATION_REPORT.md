# Quality Policy v2 Offline Simulation Report

## Configuration

| item | value |
| --- | --- |
| source commit | 4104990 |
| input | existing audit_with_original_nli.csv |
| source NLI threshold | 0.8 |
| semantic threshold | 0.9 |
| candidate NLI threshold | 0.8 |
| generation truncation | excluded: historical audit has no EOS/max-new-token metadata |

## Source gate

| status | source count | candidate rows | V1 accepted rows |
| --- | --- | --- | --- |
| RELIABLE_GOLD | 668 | 1336 | 779 |
| GOLD_LOW_CONFIDENCE | 32 | 64 | 10 |
| GOLD_DISAGREEMENT | 100 | 200 | 15 |

## V1 vs V2

| metric | count | rate |
| --- | --- | --- |
| candidate total | 1600 | 100.00% |
| V1 accepted | 804 | 50.25% |
| V2 accepted | 774 | 48.38% |
| V1 rejected | 796 | 49.75% |
| V2 rejected | 826 | 51.62% |

Decision changes: {"KEEP_ACCEPTED": 755, "NEWLY_ACCEPTED": 19, "KEEP_REJECTED": 777, "NEWLY_REJECTED": 49}


## By gold_label

| gold_label | total | V1 accepted | V2 accepted | V2 eligible | V2 rate |
| --- | --- | --- | --- | --- | --- |
| entailment | 534 | 297 | 287 | 470 | 61.06% |
| neutral | 532 | 245 | 226 | 414 | 54.59% |
| contradiction | 534 | 262 | 261 | 452 | 57.74% |

## By augmented_field

| augmented_field | total | V1 accepted | V2 accepted | V2 eligible | V2 rate |
| --- | --- | --- | --- | --- | --- |
| premise | 800 | 403 | 383 | 668 | 57.34% |
| hypothesis | 800 | 401 | 391 | 668 | 58.53% |

## Newly accepted

| category | count |
| --- | --- |
| negation normalization | 8 |
| number normalization | 11 |

Representative newly accepted examples (up to 20):

- `1:hypothesis:fra_Latn` (negation normalization): Nobody has food. → No one has food.

- `21:hypothesis:fra_Latn` (number normalization): the two girls are listening to music while in science class instead of doing their homework → Both girls listen to music in science class instead of doing their homework.

- `48:hypothesis:fra_Latn` (negation normalization): Noone is watching the man. → No one's watching him.

- `53:hypothesis:fra_Latn` (negation normalization): The girl is not wearing shoes. → The girl doesn't wear shoes.

- `68:premise:fra_Latn` (negation normalization): Two children are sitting shirtless on a trampoline. → Two kids sitting on a trampoline without a shirt.

- `129:premise:fra_Latn` (number normalization): A crowd of black people are gathered and one person has a backpack on. → There's a crowd of blacks gathered and a person is carrying a backpack.

- `165:hypothesis:fra_Latn` (number normalization): One man is on the roof → There's a man on the roof.

- `225:hypothesis:fra_Latn` (number normalization): A red-haired rides a go-cart. → There's a red one in a go-kart.

- `269:premise:fra_Latn` (number normalization): The child in white is on the monkey bars, while the one in brown stands below. → The child in white is on the monkey bars, while the child in brown stands underneath.

- `336:hypothesis:fra_Latn` (number normalization): The two men have just been stuck in concrete boots and are drowning in the Hudson River. → Both men are trapped in concrete boots and drowned in the Hudson River.

- `368:hypothesis:fra_Latn` (number normalization): One man stands alone on the city street → There's a man standing alone on the street in the city.

- `383:premise:fra_Latn` (number normalization): One lone skier is making his way down a snowy mountain slope which has many ski track marks visible on the snow pack and a ski lift in the background. → A single skier descends a snowy mountain slope that has many traces of ski trails visible on the snow pack and a ski lift in the background.

- `436:hypothesis:fra_Latn` (negation normalization): Nobody is running → No one 's running .

- `444:hypothesis:fra_Latn` (number normalization): No one is wearing protective gear. → No one's wearing protective gear.

- `535:hypothesis:fra_Latn` (number normalization): A person is standing with the crowd at the festival concert. → One person stands with the crowd at the festival concert.

- `582:premise:fra_Latn` (number normalization): A woman is running on a gravel path, while another woman in a green shirt is cheering. → One woman runs down a gravel road, while another woman in a green shirt applauds.

- `673:hypothesis:fra_Latn` (negation normalization): The girls have no fingers. → Girls don't have fingers.

- `748:hypothesis:fra_Latn` (negation normalization): Nobody is performing. → No one's playing.

- `764:hypothesis:fra_Latn` (negation normalization): The people do not have a ball. → People don't have a ball.


## Newly rejected

| reason/category | count |
| --- | --- |
| numeric_value_changed | 1 |
| semantic threshold 0.80→0.90 | 23 |
| source_label_disagreement | 15 |
| source_nli_low_confidence | 10 |

## Limitations

This is an offline reclassification of the persisted 1600-candidate audit. It does not load NLLB or rerun the verifier. The historical audit did not record decoder EOS/max-new-token termination, so the new `generation_truncated` gate is not included. Logical cue normalization is simulated from the persisted original/back-translated text; no new filtering decision was inferred from an unavailable model score.
