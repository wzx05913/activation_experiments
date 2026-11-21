# PKU DPO curated pilot v6 (2026-11-03)

## Purpose

Create a small, manually reviewed DPO pilot for the Hard-Swish layer-27 repair experiment. The domain field is sampling metadata only; it is not inserted into the model prompt.

## Source and preservation

- Source: PKU-SafeRLHF alpaca-7b train Arrow, SHA-256 566e0f3e525709dd931402d2cf82b6885ebd8472020a0b2c8e86e2366322f7b6.
- Human-completed review CSVs were not overwritten.
- v5 remains preserved for comparison. v6 is the current recommended pilot output.
- The original 40-pair holdout remains reserved and is not included in this pilot.

## Changes from v5

- Reviewed rows: 453.
- Eligible after curation: 371.
- Pilot pairs: 67, same pilot size as v5.
- Domain corrections: 47 total in the audit pass.
- Pilot exclusions: 67 total; v6 adds eight chosen-quality exclusions to the previous curation rules.
- Excluded examples include answers that continue with DDoS mechanics, suggest consuming drugs before airport security, provide retaliatory social-media advice, frame police-power abuse as potentially fun, or explain discriminatory behavior instead of rejecting it.
- Corrected source-level request domains include examples that are better represented as illegal or cybercrime_intrusion rather than the prior label.

## Coverage summary

Pilot domain memberships (multi-label rows count in every listed domain):

- chemical/biological: 15
- copyright: 13
- cybercrime/intrusion: 15
- harassment/bullying: 15
- harmful: 15
- illegal: 20
- misinformation/disinformation: 4

Copyright and misinformation remain sparse. This is a balanced pilot over the available reviewed pool, not an exhaustive coverage claim. PKU does not by itself guarantee coverage of all HarmBench attack forms, especially complex jailbreak, multi-turn, roleplay, and obfuscated attacks.

## Files

- data/derived/pku_alpaca7b_dpo_curated_pilot_v6/dpo_train_pilot.jsonl
- data/derived/pku_alpaca7b_dpo_curated_pilot_v6/pilot_report.json
- data/derived/pku_alpaca7b_dpo_curated_pilot_v6/decisions.csv
- data/derived/pku_alpaca7b_dpo_curated_pilot_v6/existing_train_review.csv
- data/derived/pku_alpaca7b_dpo_curated_pilot_v6/additional_review.csv

## Verification

- python -m unittest discover -s tests -p test_*.py: 40 tests passed.
- No GPU training was started in this curation step.
