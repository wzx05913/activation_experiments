"""Measure function-level approximation error for SiLU candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .approximation import ApproximateSiLU


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain", type=float, default=4.0)
    parser.add_argument("--num-bins", type=int, default=16)
    parser.add_argument("--samples", type=int, default=20001)
    args = parser.parse_args()
    x = torch.linspace(-8.0, 8.0, args.samples, dtype=torch.float32)
    reference = torch.nn.functional.silu(x)
    records = []
    for kind in ("hard_swish", "piecewise_linear", "piecewise_polynomial"):
        approx = ApproximateSiLU(kind, num_bins=args.num_bins, domain=args.domain)(x)
        err = (approx - reference).abs()
        records.append({
            "method": kind,
            "domain": args.domain,
            "num_bins": args.num_bins,
            "samples": args.samples,
            "max_abs_error": float(err.max()),
            "mean_abs_error": float(err.mean()),
            "rmse": float(torch.mean((approx-reference)**2).sqrt()),
            "cosine_similarity": float(torch.nn.functional.cosine_similarity(approx, reference, dim=0)),
        })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
