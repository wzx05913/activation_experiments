"""Audit PKU preference coverage and stage additional pairs for human review."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.ipc as ipc

from .prepare_pku_sft import exclusions, key, write
from .scoring_rules import refusal_match

DOMAINS = ("chemical_biological", "copyright", "cybercrime_intrusion",
           "harassment_bullying", "harmful", "illegal", "misinformation_disinformation", "other")
FORMS = ("direct", "contextual", "roleplay", "authority", "hypothetical",
         "obfuscated", "multi_turn", "other", "direct_request",
         "personal_justification", "disguised_intent", "indirect_inquiry",
         "third_party_scenario", "provocation_offense", "hypothetical_research")
HINTS = {"Drugs": "chemical_biological", "Endangering Public Health": "chemical_biological",
         "Copyright Issues": "copyright", "Cybercrime": "cybercrime_intrusion",
         "Insulting Behavior": "harassment_bullying", "Discriminatory Behavior": "harassment_bullying",
         "Violence": "harmful", "Physical Harm": "harmful", "Psychological Harm": "harmful",
         "Economic Crime": "illegal", "White-Collar Crime": "illegal", "Human Trafficking": "illegal",
         "Disrupting Public Order": "illegal", "Privacy Violation": "illegal",
         "Mental Manipulation": "misinformation_disinformation"}
FIELDS = ("id", "prompt", "chosen", "rejected", "prompt_source", "pku_categories",
          "domain_hint", "prompt_harmful", "chosen_safe", "rejected_unsafe",
          "domain", "attack_form", "review_note")


def parse_domains(value):
    domains = [part.strip() for part in value.split(";")]
    if not domains or any(part not in DOMAINS for part in domains) or len(domains) != len(set(domains)):
        raise ValueError(f"Invalid or duplicate domain: {value!r}")
    if "other" in domains and len(domains) > 1:
        raise ValueError("other cannot be combined with a known domain")
    return domains


def validate_pairs(rows):
    if not rows:
        raise ValueError("Empty preference pairs")
    ids, prompts = set(), set()
    for row in rows:
        identifier, prompt = row.get("id"), key(row.get("prompt", ""))
        if (not identifier or not prompt or not row.get("chosen") or not row.get("rejected")
                or row["chosen"] == row["rejected"] or row.get("reviewed") is not True
                or identifier in ids or prompt in prompts):
            raise ValueError(f"Invalid, unreviewed or duplicate pair: {identifier}")
        ids.add(identifier)
        prompts.add(prompt)


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_source(path):
    if path.name != "pku-safe_rlhf-train.arrow":
        raise ValueError("Use the PKU train Arrow only")
    with path.open("rb") as stream:
        rows = ipc.open_stream(stream).read_all().to_pylist()
    result = {}
    for i, row in enumerate(rows):
        safe = [j for j in (0, 1) if row[f"is_response_{j}_safe"] is True]
        if len(safe) != 1 or row[f"is_response_{1-safe[0]}_safe"] is not False:
            continue
        prompt = str(row["prompt"] or "").strip()
        chosen = str(row[f"response_{safe[0]}"] or "").strip()
        rejected = str(row[f"response_{1-safe[0]}"] or "").strip()
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        cats = sorted(k for k, v in (row[f"response_{1-safe[0]}_harm_category"] or {}).items() if v)
        hints = sorted({HINTS[k] for k in cats if k in HINTS})
        identifier = f"pku_alpaca7b_train_{i:05d}"
        result[identifier] = {"id": identifier, "prompt": prompt, "chosen": chosen,
                              "rejected": rejected, "prompt_source": row.get("prompt_source") or "",
                              "pku_categories": "; ".join(cats), "domain_hint": "; ".join(hints) or "other"}
    return result


def review_row(pair):
    return {**pair, "prompt_harmful": "", "chosen_safe": "", "rejected_unsafe": "",
            "domain": "", "attack_form": "", "review_note": ""}


def save_review(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_review(path, allow_unclassified=False):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing review columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty review: {path}")
    for row in rows:
        flags = ("prompt_harmful", "chosen_safe", "rejected_unsafe")
        if any(row[f].strip().lower() not in {"yes", "no"} for f in flags):
            raise ValueError(f"Review all yes/no columns: {path} ({row['id']})")
        if all(row[f].strip().lower() == "yes" for f in flags):
            domain, form = row["domain"].strip(), row["attack_form"].strip()
            if allow_unclassified and not domain and not form:
                continue
            if not domain or form not in FORMS:
                raise ValueError(f"Approved row needs valid domain/attack_form: {path} ({row['id']})")
            try:
                parse_domains(domain)
            except ValueError as exc:
                raise ValueError(f"Approved row needs valid domain/attack_form: {path} ({row['id']})") from exc
    return rows


def approved(rows, source, allow_unclassified=False):
    result = []
    for row in rows:
        if not all(row[f].strip().lower() == "yes" for f in ("prompt_harmful", "chosen_safe", "rejected_unsafe")):
            continue
        if allow_unclassified and not row["domain"].strip() and not row["attack_form"].strip():
            continue
        pair = source.get(row["id"])
        if pair is None or any(pair[k] != row[k] for k in ("prompt", "chosen", "rejected")):
            raise ValueError(f"Review differs from Arrow: {row['id']}")
        result.append({"id": pair["id"], "prompt": pair["prompt"], "chosen": pair["chosen"],
                       "rejected": pair["rejected"], "reviewed": True,
                       "source": "PKU-Alignment/PKU-SafeRLHF:alpaca-7b:train",
                       "domain": row["domain"].strip(), "attack_form": row["attack_form"].strip()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("screen", "export"):
        p = sub.add_parser(action)
        for flag in ("arrow", "old-train", "old-holdout", "eval-root", "output-dir"):
            p.add_argument("--" + flag, type=Path, required=True)
        if action == "screen":
            p.add_argument("--candidates", type=int, default=400)
            p.add_argument("--seed", type=int, default=42)
        else:
            p.add_argument("--review-dir", type=Path, required=True)
            p.add_argument("--allow-unclassified", action="store_true",
                           help="Export only manually classified, safety-approved pairs")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    source = load_source(args.arrow)
    train, holdout = read_jsonl(args.old_train), read_jsonl(args.old_holdout)
    validate_pairs(train)
    validate_pairs(holdout)
    old_ids = {r["id"] for r in train + holdout}
    old_keys = {key(r["prompt"]) for r in train + holdout}
    if len(old_ids) != len(train) + len(holdout) or len(old_keys) != len(old_ids):
        raise ValueError("Old train/holdout overlap")
    blocked = exclusions(args.eval_root)
    if old_keys & blocked:
        raise ValueError("Existing pair overlaps evaluation")
    for row in train + holdout:
        pair = source.get(row["id"])
        if pair is None or any(pair[k] != row[k] for k in ("prompt", "chosen", "rejected")):
            raise ValueError(f"Existing pair differs from Arrow: {row['id']}")
    if args.action == "screen":
        if args.candidates <= 0:
            parser.error("Candidate count must be positive")
        pool, seen, skipped = [], set(), Counter()
        for pair in source.values():
            normalized = key(pair["prompt"])
            if pair["id"] in old_ids or normalized in old_keys:
                skipped["existing_split"] += 1
            elif normalized in blocked:
                skipped["eval_overlap"] += 1
            elif normalized in seen:
                skipped["duplicate"] += 1
            elif len(pair["chosen"]) > 1800 or not refusal_match(pair["chosen"][:240])[0]:
                skipped["not_refusal_candidate"] += 1
            else:
                seen.add(normalized)
                pool.append(pair)
        rng = random.Random(args.seed)
        groups = defaultdict(list)
        for pair in pool:
            groups[pair["domain_hint"].split("; ")[0]].append(pair)
        for rows in groups.values():
            rng.shuffle(rows)
        selected = []
        while len(selected) < args.candidates and any(groups.values()):
            for domain in sorted(groups):
                if groups[domain] and len(selected) < args.candidates:
                    selected.append(groups[domain].pop())
        save_review(args.output_dir / "existing_train_review.csv", [review_row(source[r["id"]]) for r in train])
        save_review(args.output_dir / "additional_review.csv", [review_row(r) for r in selected])
        with args.arrow.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report = {"source_sha256": digest, "old_train": len(train), "holdout_reserved": len(holdout),
                  "eligible_additional": len(pool), "additional_candidates": len(selected),
                  "unfilled_request": args.candidates - len(selected),
                  "candidate_domain_hints": dict(Counter(r["domain_hint"] for r in selected)),
                  "candidate_prompt_sources": dict(Counter(r["prompt_source"] for r in selected)),
                  "old_train_domain_hints": dict(Counter(source[r["id"]]["domain_hint"] for r in train)),
                  "review_candidate_ids": [r["id"] for r in selected],
                  "skipped": dict(skipped), "domains": DOMAINS, "attack_forms": FORMS,
                  "coverage_status": "UNVERIFIED. PKU categories label responses, not prompt intent or attack form. Review both CSVs."}
        name = "coverage_screen_report.json"
    else:
        screen_report = json.loads((args.review_dir / "coverage_screen_report.json").read_text(encoding="utf-8"))
        with args.arrow.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if screen_report["source_sha256"] != digest:
            raise ValueError("Review source Arrow has changed")
        old_review = read_review(args.review_dir / "existing_train_review.csv", args.allow_unclassified)
        extra_review = read_review(args.review_dir / "additional_review.csv", args.allow_unclassified)
        if len(old_review) != len(train) or {r["id"] for r in old_review} != {r["id"] for r in train}:
            raise ValueError("Existing review must match original train IDs")
        if len({r["id"] for r in extra_review}) != len(extra_review) or {r["id"] for r in extra_review} & old_ids:
            raise ValueError("Additional review contains duplicate/reserved IDs")
        if [r["id"] for r in extra_review] != screen_report["review_candidate_ids"]:
            raise ValueError("Additional review does not match screened candidates")
        old_approved = approved(old_review, source, args.allow_unclassified)
        extra_approved = approved(extra_review, source, args.allow_unclassified)
        if not old_approved and not extra_approved:
            raise ValueError("No approved classified pairs")
        combined = old_approved + extra_approved
        validate_pairs(combined)
        if {key(r["prompt"]) for r in combined} & (blocked | {key(r["prompt"]) for r in holdout}):
            raise ValueError("Train/evaluation/holdout overlap")
        write(args.output_dir / "dpo_train.jsonl", combined)
        report = {"old_approved": len(old_approved), "additional_approved": len(extra_approved),
                  "train_pairs": len(combined), "holdout_reserved": len(holdout),
                  "unclassified_skipped": sum(1 for r in old_review + extra_review
                                              if all(r[f].strip().lower() == "yes" for f in
                                                     ("prompt_harmful", "chosen_safe", "rejected_unsafe"))
                                              and not r["domain"].strip() and not r["attack_form"].strip()),
                  "domain_counts": dict(Counter(domain for r in combined for domain in parse_domains(r["domain"]))),
                  "attack_form_counts": dict(Counter(r["attack_form"] for r in combined)),
                  "note": "Coverage counts are not a claim of all attack types. Keep original holdout and check semantic near-duplicates."}
        name = "coverage_export_report.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / name).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
