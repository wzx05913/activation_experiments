"""Curate reviewed PKU pairs and select a small risk-domain pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .expand_pku_dpo import FIELDS, FORMS, load_source, parse_domains, read_jsonl, validate_pairs
from .prepare_pku_sft import exclusions, key, write

FILES = ("existing_train_review_completed.csv", "additional_review_completed.csv")
TARGETS = ("chemical_biological", "copyright", "cybercrime_intrusion", "harassment_bullying",
           "harmful", "illegal", "misinformation_disinformation")

# Request-level corrections; the original completed CSVs are never edited.
CORRECTIONS = {
    "23199": "illegal; harmful", "06484": "harmful", "10329": "chemical_biological; illegal",
    "01970": "chemical_biological; illegal", "26263": "illegal; harmful",
    "21720": "misinformation_disinformation; illegal", "10541": "illegal",
    "12622": "illegal", "24134": "illegal", "10210": "harmful", "21421": "harmful",
    "12521": "harmful", "02136": "harmful", "22939": "harmful", "03427": "harmful",
    "22768": "harmful", "04134": "illegal; harmful", "04161": "harassment_bullying",
    "25026": "harassment_bullying", "12774": "harassment_bullying", "02552": "harassment_bullying",
    "09682": "harassment_bullying", "01489": "illegal", "03733": "illegal; harmful",
    "19407": "illegal", "08755": "illegal", "02353": "illegal", "24767": "illegal",
    "15294": "illegal", "18174": "illegal", "07551": "illegal", "26232": "illegal",
    "23814": "illegal", "06196": "illegal", "04890": "illegal", "23995": "illegal",
    "26871": "illegal", "25929": "illegal", "15566": "cybercrime_intrusion; illegal",
    "15207": "other", "10058": "other", "24736": "harmful",
    "20412": "other", "16835": "illegal", "10016": "other",
    "10967": "copyright", "18680": "illegal",
    "18548": "illegal", "22605": "illegal",
}

# Excluded from this harmful-request/refusal pilot, including ambiguous chosen responses.
EXCLUDE = {
    "25561": "Concern about terrorism, not an attack request; chosen gives ordinary advice.",
    "04574": "Chosen cites media-generation limits, not safety refusal.",
    "20412": "Exam cheating is outside target domains; chosen does not refuse.",
    "25928": "Lawful resale ambiguous; chosen overstates copyright restriction.",
    "23995": "Fraud request, but chosen asserts improbability rather than refusing.",
    "15207": "Infidelity is not public misinformation.",
    "10058": "Private relationship deception is not public misinformation.",
    "14915": "Lawful resale/license transfer ambiguous; chosen blanket-prohibits it.",
    "01507": "Household cleaning request is not clearly malicious.",
    "23509": "No concrete harmful action requested.",
    "24344": "Chosen supplies a joke rather than refusing.",
    "00540": "Consensual relationship request; chosen gives advice.",
    "11897": "Pet rehoming can be benign; chosen gives advice.",
    "09915": "Consensual private recording; chosen gives advice.",
    "15423": "Chosen offers humane training alternatives, not a target refusal.",
    "05424": "Locating a widow lacks explicit harmful intent.",
    "20193": "Chosen reiterates stereotypes.",
    "20544": "Chosen provides jokes, not refusal.",
    "01339": "Online privacy can be legitimate; chosen gives VPN guidance.",
    "24015": "Anonymous hosting can be legitimate; chosen gives privacy guidance.",
    "04177": "Benign pest control interpretation; chosen gives advice.",
    "26128": "Pet care can be benign; chosen gives instructions.",
    "07044": "Chosen critiques stereotype, not a refusal example.",
    "15652": "Friend may have authorized account use; chosen discusses permission.",
    "10016": "Contacting ex-spouse lacks explicit harmful intent; chosen over-refuses.",
    "24884": "Chosen confusingly suggests alternative illicit methods.",
    "17746": "Political gathering may be legitimate.",
    "11728": "Disruptive protest is not inherently harmful.",
    "20205": "Authorized free music access ambiguous; chosen overstates rule.",
    "06835": "Copyright use depends on license; needs closer review.",
    "12713": "Pregnancy/abortion needs supportive guidance, not blanket refusal.",
    "07507": "Gift request for private address is not an explicit attack.",
    "10950": "Gift request for private address is not an explicit attack.",
    "25780": "Species population control may be legitimate conservation.",
    "11505": "Medical privacy request lacks clear malicious intent.",
    "00059": "Historical review chosen is truncated and differs from original PKU Arrow.",
    "08293": "Chosen begins by endorsing disciplinary actions rather than clearly refusing emotional abuse.",
    "11654": "Chosen describes a sabotage effect rather than refusing.",
    "08574": "Medical safety: chosen incorrectly says brake fluid will not kill you.",
    "13123": "Chosen contains inaccurate claim about cough medicine and is not a clean refusal.",
    "25989": "Chosen cites advice limits rather than rejecting drugging and theft.",
    "10969": "Chosen denies model capability instead of refusing extortion.",
    "24653": "Chosen terminates conversation without explaining the misinformation risk.",
    "20495": "Free access may be licensed; chosen blanket-prohibits it.",
    "12352": "Free authorized music access exists; chosen blanket-prohibits it.",
    "10008": "Chosen repeats attack vectors; too much harmful detail for a refusal pilot.",
    "19203": "Chosen makes false categorical claims about crypto and illicit transactions.",
    "12853": "Chosen proposes relocation incentives after rejecting religious expulsion.",
    "17879": "Chosen claims incapability rather than refusing drug procurement.",
    "19032": "Chosen claims incapability rather than refusing unauthorized deletion.",
    "13081": "Chosen implies racial intelligence is subjective rather than rejecting the premise.",
    "04729": "Chosen makes questionable claims about drug-test traceability.",
    "02552": "Chosen gives publicly searchable targeting advice; not a clean refusal.",
    "23148": "Joining political communities is ambiguous absent explicit violent intent.",
    "06688": "Chosen reinforces gender stereotypes rather than challenging them.",
    "01279": "Chosen suggests alternate routes to dismiss a pregnant employee.",
    "12427": "Chosen discusses DDoS capability requirements rather than refusing the attack.",
    "05635": "Chosen cites computer limitations rather than a safety refusal.",
    "00559": "Chosen provides supposed algorithm details rather than refusing intrusion.",
    "26587": "Chosen claims password cracking is impossible instead of a clean refusal.",
    "04933": "Cannabis purchase request is ambiguous and chosen offers purchase advice.",
    "11069": "Chosen elaborates a racial-intelligence comparison rather than a clean rejection.",
    "02151": "Chosen treats discriminatory action as impossible rather than rejecting it.",
    "21421": "Chosen redirects to DIY instructions for fireworks after refusing explosives.",
    "10759": "Chosen explains the DDoS mechanism after the attack request.",
    "17943": "Chosen recommends consuming drugs before arriving at airport security.",
    "10180": "Chosen suggests retaliatory social-media pranks.",
    "16015": "Chosen frames abuse of police power as potentially fun.",
}

# Preserve the meaning of the human review columns: a bad chosen is not a benign prompt.
BAD_CHOSEN = {"04574", "23995", "24344", "20193", "20544", "24884", "08293", "11654",
              "08574", "13123", "25989", "10969", "24653", "10008", "19203", "12853",
              "17879", "19032", "13081", "04729", "02552", "06688", "01279",
              "12427", "05635", "00559", "26587", "04933", "11069", "02151",
              "21421", "10759", "17943", "10180", "16015"}

assert BAD_CHOSEN <= EXCLUDE.keys()
assert len(CORRECTIONS) == len(set(CORRECTIONS))


def curate(review_dir, arrow, old_train, old_holdout, eval_root, output_dir, per_domain=15, seed=42):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty")
    screen = json.loads((review_dir / "coverage_screen_report.json").read_text(encoding="utf-8"))
    with arrow.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != screen["source_sha256"]:
        raise ValueError("PKU source hash differs from screened review")
    source = load_source(arrow)
    train, holdout = read_jsonl(old_train), read_jsonl(old_holdout)
    validate_pairs(train)
    validate_pairs(holdout)
    blocked = exclusions(eval_root) | {key(row["prompt"]) for row in holdout}
    seen_ids, seen_prompts, decisions, curated, eligible = set(), set(), [], {}, []
    for filename in FILES:
        with (review_dir / filename).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if filename == FILES[0] and {r["id"] for r in rows} != {r["id"] for r in train}:
            raise ValueError("Existing train IDs changed")
        if filename == FILES[1] and [r["id"] for r in rows] != screen["review_candidate_ids"]:
            raise ValueError("Additional review IDs/order changed")
        curated[filename] = []
        for row in rows:
            identifier = row["id"]
            suffix = identifier.rsplit("_", 1)[-1]
            src = source.get(identifier)
            if not src or (suffix != "00059" and any(row[f] != src[f] for f in ("prompt", "chosen", "rejected"))):
                raise ValueError(f"Pair differs from PKU Arrow: {identifier}")
            if suffix == "00059" and (row["prompt"] != src["prompt"] or row["rejected"] != src["rejected"]
                                      or row["chosen"] == src["chosen"] or not src["chosen"].startswith(row["chosen"])):
                raise ValueError("Expected historical truncation changed: 00059")
            normalized = key(row["prompt"])
            if identifier in seen_ids or normalized in seen_prompts or normalized in blocked:
                raise ValueError(f"Duplicate or evaluation/holdout overlap: {identifier}")
            seen_ids.add(identifier)
            seen_prompts.add(normalized)
            result = {f: row[f] for f in FIELDS}
            if suffix in CORRECTIONS and all(row[f].strip().lower() == "yes" for f in ("prompt_harmful", "chosen_safe", "rejected_unsafe")):
                result["domain"] = CORRECTIONS[suffix]
                if result["domain"] != row["domain"]:
                    decisions.append({"id": identifier, "action": "domain_corrected", "before": row["domain"],
                                      "after": result["domain"], "reason": "Request-level risk; see original prompt."})
            if suffix in EXCLUDE:
                field = "chosen_safe" if suffix in BAD_CHOSEN else "prompt_harmful"
                result[field] = "no"
                result["review_note"] = (result["review_note"].strip() + " | Pilot exclusion: " + EXCLUDE[suffix]).strip(" |")
                decisions.append({"id": identifier, "action": "pilot_excluded", "before": row["domain"],
                                  "after": result["domain"], "reason": EXCLUDE[suffix]})
            curated[filename].append(result)
            if all(result[f].strip().lower() == "yes" for f in ("prompt_harmful", "chosen_safe", "rejected_unsafe")):
                domains = parse_domains(result["domain"])
                if result["attack_form"] not in FORMS:
                    raise ValueError(f"Invalid attack form: {identifier}")
                eligible.append({"id": identifier, "prompt": src["prompt"], "chosen": src["chosen"],
                                 "rejected": src["rejected"], "reviewed": True,
                                 "source": "PKU-Alignment/PKU-SafeRLHF:alpaca-7b:train",
                                 "domain": "; ".join(domains), "attack_form": result["attack_form"]})
    validate_pairs(eligible)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen, counts, forms = [], Counter(), Counter()
    remaining = eligible[:]
    for domain in sorted(TARGETS, key=lambda d: sum(d in parse_domains(r["domain"]) for r in eligible)):
        while counts[domain] < per_domain:
            options = [r for r in remaining if domain in parse_domains(r["domain"])]
            if not options:
                break
            row = max(options, key=lambda r: (sum(counts[d] < per_domain for d in parse_domains(r["domain"])),
                                               -forms[(domain, r["attack_form"])],
                                               -sum(counts[d] for d in parse_domains(r["domain"]))))
            chosen.append(row)
            remaining.remove(row)
            for d in parse_domains(row["domain"]):
                counts[d] += 1
                forms[(d, row["attack_form"])] += 1
    rng.shuffle(chosen)
    validate_pairs(chosen)
    output_dir.mkdir(parents=True)
    (output_dir / "coverage_screen_report.json").write_text(
        json.dumps(screen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for filename in FILES:
        out = output_dir / filename.replace("_completed", "")
        with out.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, FIELDS)
            writer.writeheader()
            writer.writerows(curated[filename])
    with (output_dir / "decisions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, ("id", "action", "before", "after", "reason"))
        writer.writeheader()
        writer.writerows(decisions)
    write(output_dir / "dpo_train_pilot.jsonl", chosen)
    report = {"source_sha256": digest, "seed": seed, "per_domain_target": per_domain,
              "reviewed_rows": len(seen_ids), "eligible_after_curating": len(eligible),
              "excluded_original_or_new": len(seen_ids) - len(eligible),
              "corrections": sum(d["action"] == "domain_corrected" for d in decisions),
              "pilot_exclusions": sum(d["action"] == "pilot_excluded" for d in decisions),
              "pilot_rows": len(chosen),
              "eligible_domain_memberships": dict(Counter(d for r in eligible for d in parse_domains(r["domain"]))),
              "pilot_domain_memberships": dict(counts),
              "pilot_form_counts": dict(Counter(r["attack_form"] for r in chosen)),
              "pilot_domain_form_counts": {d: dict(Counter(r["attack_form"] for r in chosen if d in parse_domains(r["domain"]))) for d in TARGETS},
              "missing_target_domains": [d for d in TARGETS if counts[d] < per_domain],
              "note": "Domain is metadata only; prompt/chosen/rejected unchanged. Multi-label counts overlap. Spot-check before GPU training; no exhaustive coverage claim."}
    (output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("review-dir", "arrow", "old-train", "old-holdout", "eval-root", "output-dir"):
        p.add_argument("--" + flag, required=True, type=Path)
    p.add_argument("--per-domain", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.per_domain < 1:
        p.error("per-domain must be positive")
    print(json.dumps(curate(args.review_dir, args.arrow, args.old_train, args.old_holdout,
                            args.eval_root, args.output_dir, args.per_domain, args.seed), indent=2))


if __name__ == "__main__":
    main()
