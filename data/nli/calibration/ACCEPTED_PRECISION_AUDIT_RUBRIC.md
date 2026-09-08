# Accepted Precision Audit Rubric

This is a blind review of 240 candidates sampled from the 774 accepted,
translated V2 candidates. Review each row using only the original and
back-translated text, the augmented premise/hypothesis pair, and the gold NLI
label. Do not infer quality from automatic scores: model scores, reasons,
source IDs, candidate IDs, and previous expert labels are deliberately absent
from the blind files.

## Semantic equivalence: `review_semantic_equivalent`

Use `PASS`, `FAIL`, or `BORDERLINE`.

- `PASS`: the back-translated sentence is a strict or near-strict paraphrase.
  Normal synonymy, word order changes, voice changes, contractions, number
  spelling, and natural lexical variants are allowed when event, entity,
  attribute, role, quantity, temporal/aspectual state, and spatial relation are
  preserved.
- `FAIL`: a substantive fact changes, is added, or is dropped; an entity,
  attribute, role, subject/object, predicate, quantity, spatial relation,
  modality, tense/aspect, or word meaning changes; or the output is corrupted.
- `BORDERLINE`: a subtle generalization/specificity or ambiguity prevents a
  stable strict-equivalence decision.

Examples of normally acceptable normalization include `Nobody` → `No one`,
`5` → `five`, `watch` → `observe`, and `fishing pole` → `fishing rod` when the
full sentence remains equivalent.

## Label preservation: `review_label_preserved`

Use `PASS`, `FAIL`, or `UNCERTAIN`. Re-evaluate the augmented premise and
hypothesis against the SNLI gold label. Do not copy the semantic verdict: a
sentence-level equivalence failure does not automatically imply that the pair
label changed, and a sentence-level PASS does not prove the pair label.

## Useful augmentation: `review_useful_augmentation`

Use `PASS`, `FAIL`, or `BORDERLINE`. Consider natural English, semantic
fidelity, useful paraphrase variation, and whether the candidate is worth
adding to training. A label-preserving candidate can still fail here if it is
unnatural, corrupted, or too close to a trivial copy.

## Severity: `review_severity`

Use `NONE`, `MINOR`, or `MAJOR`.

- `NONE`: no substantive problem.
- `MINOR`: mild fluency or harmless generalization issue.
- `MAJOR`: clear semantic drift, label risk, corruption, truncation, or logic
  change.

## Error taxonomy: `review_error_types`

Use zero or more pipe-separated values from:

```text
TENSE_ASPECT|EVENT_PREDICATE|ENTITY_TYPE|ENTITY_ATTRIBUTE|ENTITY_ROLE|
OBJECT_CHANGE|SUBJECT_CHANGE|NUMBER|NEGATION|QUANTIFIER|MODALITY|TEMPORAL|
SPATIAL|COREFERENCE|INFORMATION_ADDED|INFORMATION_DROPPED|LEXICAL_MEANING|
FLUENCY|CORRUPTION|TRUNCATION|TRIVIAL_COPY|OTHER
```

Leave it empty when there is no error. Use `review_notes` for concise evidence,
not for model scores or guesses about hidden IDs.

## Analysis definitions

Strict quality is `semantic=PASS` and `label=PASS` and `useful=PASS`.
Optimistic quality allows semantic/useful `PASS` or `BORDERLINE` and label
`PASS` or `UNCERTAIN`, while any explicit `FAIL` remains unsuccessful.
Population-weighted estimates use the true six-stratum sizes from the secret
key. The analysis script uses a stratified bootstrap with seed `20260908` and
20,000 iterations. No GREEN/YELLOW/RED recommendation is valid until all
review columns are completed.
