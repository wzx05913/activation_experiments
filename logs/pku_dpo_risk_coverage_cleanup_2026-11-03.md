# PKU DPO risk coverage and chosen cleanup (2026-11-03)

## Scope and taxonomy

The cited `2502.17424v6` paper is *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs*. Its Appendix B.3/Table 4 splits **evaluation questions** into vulnerable user, illegal recommendations, creative writing, problems with humans, talk back, medical advice, and other. These are not seven harmful-request classes: e.g. a user asking how to handle a fire or poisoning should receive helpful emergency guidance, not a blanket refusal. Do not use these seven labels as refusal targets.

For this experiment use the seven domains present in the local HarmBench behavior CSVs as a **coverage checklist**: `chemical_biological`, `copyright`, `cybercrime_intrusion`, `harassment_bullying`, `harmful`, `illegal`, `misinformation_disinformation`. `other` means unresolved/extra risk, not evidence that any one domain is covered. This is an operational mapping for comparison, not a claim that HarmBench taxonomy exhausts all real-world risk. Domain is separate from `attack_form`: `direct`, `contextual`, `roleplay`, `authority`, `hypothetical`, `obfuscated`, `multi_turn`, `other`. For copyright and misinformation, read the actual request and answer: do not assume all requests in the area warrant refusal. PKU `response_*_harm_category` describes the **response**, so `suggested_domain` is a triage hint only. Multi-label hints remain unassigned.

## Cleaning outcome

Original human-reviewed files remain at `data/derived/pku_alpaca7b_dpo_expansion_review_v1/`. A separate staging copy is at `data/derived/pku_alpaca7b_dpo_expansion_clean_v1/`:

| Source | Reviewed | Eligible after chosen check | Excluded |
| --- | ---: | ---: | ---: |
| Existing train | 160 | 157 | 3 |
| Additional candidates | 293 | 274 | 19 |
| Total | 453 | 431 | 22 |

All 453 prompts and rejected answers were marked `yes` for harmful and unsafe respectively. The 22 excluded pairs have `chosen_safe=no`, including one `no(...)` annotation stored verbatim in `excluded_review.csv` and its reason copied to `review_note` in the staging table. No chosen text was rewritten; excluded pairs will not be exported. Previously trained adapters used the old 160-pair data and cannot be retrospectively treated as trained on this cleaned set.

All 431 eligible rows still have blank `domain` and `attack_form`, so **verified domain/form counts are presently zero**. The single-label response hints among these rows are: illegal 86, harmful 52, harassment 18, chemical/biological 17, copyright 17, cybercrime 16, misinformation 15, other 8; 202 are ambiguous/multi-label. These numbers are *not request coverage*. See `cleaning_report.json` for machine-readable totals.

## Next review and experiment

1. Review the two staged CSVs, **only for rows with all three safety flags `yes`**. Start with about 15 candidate rows per suggested domain (roughly 105), then inspect ambiguous/multi-label rows as needed to fill genuine gaps. Fill `domain` from the request's risk and `attack_form` from how the request tries to elicit the answer. `suggested_domain` is not a validated label. Use `other` if uncertain and record ambiguity in `review_note`. Leave excluded and unselected rows' categories blank. Do not bulk-accept the hints. Spot-check chosen answers again in underrepresented domains, including whether the refusal avoids actionable detail.
2. Export using `src.expand_pku_dpo export` with `--allow-unclassified`, the same `--arrow`, `--old-train`, `--old-holdout`, and `--eval-root` arguments from `logs/pku_dpo_expansion_coverage_2026-11-03.md`, but set `--review-dir data/derived/pku_alpaca7b_dpo_expansion_clean_v1` and use a fresh `--output-dir`. Only safety-approved rows with **both** manual labels are exported. Partial labels fail validation; blank pairs are counted as `unclassified_skipped`. Without this flag, export remains strict. It checks source Arrow identity, unchanged exported pair text, exact train/holdout/evaluation overlap, and duplicate prompts. Inspect its **verified** domain/form counts. Also manually inspect semantic near-duplicates; exact-match checks do not catch paraphrases.
3. Decide a small balanced training subset only after reviewing category counts: initially roughly 15 independent refusals per domain where available, with several *genuine* request forms per domain. If one domain or form has no suitable PKU pairs, document the gap and acquire a separate training split; never reuse HarmBench val/test, AdvBench, XSTest or the 40 PKU holdout for training. Do not synthesize a chosen answer and mark it human-approved without another review.
4. Train the Layer-27 Hard-Swish DPO pilot on the cleaned subset, compare to unmodified and unrepaired baselines on held-out harmful requests **and** benign/XSTest utility to detect over-refusal. Keep dataset, random seed and generation/scoring rules fixed across adapters. Limited training coverage does not guarantee broad robustness or transfer to unseen attack forms.

For reproducibility, staging can be rerun into a new empty directory with:

```bash
python -m src.clean_pku_dpo_review \
  --review-dir data/derived/pku_alpaca7b_dpo_expansion_review_v1 \
  --output-dir data/derived/pku_alpaca7b_dpo_expansion_clean_v2
```
