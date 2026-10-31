"""P split-half ceiling (plan 3.5, risk 1), with the parameters fixed in D69.

Reads the P patching results, splits the objects into random halves 1,000
times, runs the identical discovery pipeline on each half, and writes the
result to results/.

    .venv/bin/python scripts/analyze_split_half.py \\
        results/replication_patching_qwen25vl3b.json \\
        results/split_half_ceiling_p_qwen25vl3b.json

CPU only, about a minute. Every number it prints goes into RESULTS.md as is.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuit_overlap.ceiling import split_half_ceiling  # noqa: E402
from circuit_overlap.provenance import result_envelope, write_result  # noqa: E402

# D69. Changing any of these after the result exists breaks the pre-registration.
N_SPLITS = 1000
SEED = 0
SIGMA = 2.0


def fmt(s: dict) -> str:
    return (f"median {s['median']:.3f}   95% interval [{s['p2_5']:.3f}, {s['p97_5']:.3f}]"
            f"   range [{s['min']:.3f}, {s['max']:.3f}]")


def main(src: str, dst: str | None) -> dict:
    payload = json.loads(Path(src).read_text())
    records = payload["result"]["records"]
    out = split_half_ceiling(records, n_splits=N_SPLITS, seed=SEED, sigma=SIGMA)

    print(f"source   {src}  (commit {payload.get('commit')}, n = {payload.get('n')})")
    print(f"splits   {out['n_splits']} x halves of {out['half_sizes']} objects, "
          f"seed {out['seed']}, sigma {out['sigma']}")
    print(f"full-sample promoting ({len(out['full_sample']['promoting'])}): "
          f"{', '.join(out['full_sample']['promoting'])}")

    print("\nSELF-AGREEMENT BETWEEN HALVES")
    print(f"  Jaccard, promoting    {fmt(out['jaccard_promoting'])}")
    print(f"  Jaccard, suppressing  {fmt(out['jaccard_suppressing'])}")
    print(f"  PC1 rank corr (576)   {fmt(out['pc1_spearman'])}")
    print(f"     Spearman-Brown to full length: {out['pc1_spearman_brown_full_length']:.3f}")
    print(f"  identical promoting sets in {out['share_of_splits_with_identical_promoting_sets']:.1%} of splits")
    print(f"  promoting set size per half: {fmt(out['promoting_set_size_per_half'])}")

    cf = out["chance_floor"]
    print(f"\nCHANCE FLOOR  two random {cf['set_size']}-head sets of {out['n_heads_total']}: "
          f"mean J {cf['mean']:.4f}, 95th percentile {cf['p95']:.4f}")

    print("\nHOW OFTEN EACH HEAD IS SELECTED (share of the 2,000 half-runs)")
    for head, freq in list(out["selection_frequency_promoting"].items())[:16]:
        marker = "" if head in out["full_sample"]["promoting"] else "   <- not in full-sample set"
        print(f"  {head:8s} {freq:6.1%}{marker}")

    if dst:
        write_result(dst, result_envelope(
            experiment="split_half_ceiling_p",
            payload=out,
            model=payload.get("model"), checkpoint=payload.get("checkpoint"),
            seed=SEED, prompt=payload.get("prompt"), n=out["n_objects"],
            repo_dir=ROOT,
            source_file=src, source_commit=payload.get("commit"),
            decision="D69",
        ))
        print(f"\nwrote {dst}")
    return out


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "results/replication_patching_qwen25vl3b.json",
        sys.argv[2] if len(sys.argv) > 2 else None,
    )
