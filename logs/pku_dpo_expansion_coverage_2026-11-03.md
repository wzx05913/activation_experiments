# PKU DPO expansion and coverage audit (2026-11-03)

## What we can establish now

- Current reviewed PKU train: 160 pairs; original PKU holdout: 40 pairs. These are separate from HarmBench/AdvBench/XSTest evaluation. The 160 were approved as harmful prompt plus safe refusal, but the rejected answers were not individually reviewed and no attack-form labels were recorded.
- HarmBench val contains 80 prompts: chemical/biological 14, copyright 20, cybercrime 15, harassment 4, general harm 4, illegal 12, misinformation 11. Its functional categories are standard 41, contextual 19 and copyright 20. Test has 320 prompts (42, 80, 52, 21, 18, 53, 54 respectively). These are evaluation **targets**, never training examples.
- PKU *response* labels on existing train suggest 5 copyright-only, 5 cybercrime-only, 5 chemical/biological-only, 2 misinformation-only and 43 illegal-only cases, with many multi-label examples. These are **not** validated prompt-domain counts; PKU categories do not describe prompt attack forms.
- New screening from the same PKU train Arrow found 293 additional distinct refusal-like candidate pairs after excluding old train/holdout, exact evaluation overlap and duplicates. Requested 400, short by 107. Initial response-label hints include 12 copyright-only, 12 cybercrime-only, 12 chemical/biological-only and 14 misinformation-only. None has been human-approved. See `data/derived/pku_alpaca7b_dpo_expansion_review_v1/coverage_screen_report.json` for the full breakdown and provenance hash.

**Conclusion:** all seven HarmBench *domain labels* have potential PKU candidates, but the old training set has not been audited by request domain or attack form. Coverage of contextual, authority, roleplay, hypothetical, obfuscated and multi-turn attacks is unknown. In particular PKU response label `Copyright Issues` does not imply coverage of requests for verbatim books or lyrics. Do not claim exhaustive coverage or train the additional candidates yet.

## Review workflow

Open `data/derived/pku_alpaca7b_dpo_expansion_review_v1/existing_train_review.csv` (160 rows) and `additional_review.csv` (293 rows). For every row mark `prompt_harmful`, `chosen_safe`, `rejected_unsafe` yes/no; the last field must be assessed independently of the PKU label. For approved rows also set one `domain` and one `attack_form` from the report's allowed lists. Use `other` when classification is uncertain rather than forcing a category. Reject answers that refuse initially but give actionable harmful detail later, or that merely claim to reproduce copyrighted text without doing so. Multi-label hints are only triage aids. Do not bulk-fill yes or infer attack form from the hint.

Suggested screening order: first audit the old 160, then prioritize categories with few validated examples and contextual/indirect prompts among the 293; inspect near-duplicates against all evaluation prompts before exporting. If reviewed PKU has no genuine examples for a category/form, collect independent **training-split** data from another source, document its provenance and review it separately. Do not adapt evaluation prompts as training data. Preserve the 40 old holdout pairs and use additional independent checks for new attack forms.

After completing both review CSVs, export on the local machine from the project root:

```bash
python -m src.expand_pku_dpo export \
  --arrow data/PKU-Alignment___pku-safe_rlhf/alpaca-7b/0.0.0/9421ffafec3fa40a1f1a7d567b4d525079477ecb/pku-safe_rlhf-train.arrow \
  --old-train data/derived/pku_alpaca7b_dpo_reviewed_v1/dpo_train.jsonl \
  --old-holdout data/derived/pku_alpaca7b_dpo_reviewed_v1/dpo_holdout.jsonl \
  --eval-root data \
  --review-dir data/derived/pku_alpaca7b_dpo_expansion_review_v1 \
  --output-dir data/derived/pku_alpaca7b_dpo_expanded_reviewed_v1
```

Review the exported `coverage_export_report.json`: report counts per domain/form, note absent or sparse cells, and compare them with the HarmBench category and functional distributions. The export checks exact prompt overlap but not semantic near-duplicates; inspect those manually before training. No adapter should be trained until this step is done.
