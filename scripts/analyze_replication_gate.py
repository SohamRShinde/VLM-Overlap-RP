"""Apply the D40 replication-gate thresholds to saved results.

Reads only from ``results/`` and recomputes every number from the per-object
records in the files (CLAUDE.md: analysis never trusts a summary it can
recompute). Either file may be missing; its criteria are then reported as
NOT RUN.

    .venv/bin/python scripts/analyze_replication_gate.py \
        results/replication_knockout_qwen25vl3b.json \
        results/replication_patching_qwen25vl3b.json

The thresholds are D40's, written on 2026-10-27 before either run. Change
them only through DECISIONS.md, never after looking at the output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from circuit_overlap.heads import format_head_label  # noqa: E402
from circuit_overlap.metrics import containment, jaccard_permutation_test  # noqa: E402
from circuit_overlap.patching import classify_heads, mean_restoration_grids, rank_heads  # noqa: E402
from circuit_overlap.replication import knockout_flip_counts  # noqa: E402

# D40 (DECISIONS.md), pre-registered 2026-10-27.
MIN_PRIOR_FLIP = 2 / 3          # (a)
MAX_VISUAL_FLIP = 0.10          # (b)
MIN_FOUND_OF_THEIR_9 = 5        # (c)
MAX_PERM_P = 0.001              # (d)
MAX_MEDIAN_RANK = 29            # (e) top 5% of 576
TOP_K_SINGLE = 3                # (f)
SINGLE_HEADS = ((26, 5), (31, 3))
N_PERM = 10_000


def verdict(ok: bool | None) -> str:
    return "NOT RUN" if ok is None else ("pass" if ok else "FAIL")


def knockout_criteria(path: Path) -> dict:
    payload = json.loads(path.read_text())
    result = payload["result"]
    records = result["records"]
    configs = {k: [tuple(h) for h in v] for k, v in result["configs"].items()}
    none = knockout_flip_counts(records, "none")
    if none["prior_flips"] or none["visual_flips"]:
        print(f"  WARNING: {none['prior_flips']} prior / {none['visual_flips']} visual 'flips' "
              "with no knockout -- the baseline answers and the cached path disagree")
    grp = knockout_flip_counts(records, "grp_promoting")
    print(f"knockout  {path}  (commit {payload['commit']}, n = {grp['n']})")
    print(f"  group promoting: prior flip {grp['prior_flips']}/{grp['n']} = {grp['prior_flip_rate']:.1%}, "
          f"visual flip {grp['visual_flips']}/{grp['n']} = {grp['visual_flip_rate']:.1%}")
    sup = knockout_flip_counts(records, "grp_suppressing")
    print(f"  group suppressing: prior flip {sup['prior_flip_rate']:.1%}, "
          f"visual flip {sup['visual_flip_rate']:.1%}")

    promoting = configs["grp_promoting"]
    rates = {h: knockout_flip_counts(records, f"ind_{format_head_label(h)}")["prior_flip_rate"]
             for h in promoting}
    ordered = sorted(promoting, key=lambda h: (-rates[h], h))
    print("  single promoting heads by prior flip: " + ", ".join(
        f"{format_head_label(h)} {rates[h]:.1%}" for h in ordered))
    single_ranks = {h: ordered.index(h) + 1 for h in SINGLE_HEADS if h in ordered}
    return {
        "a": grp["prior_flip_rate"] >= MIN_PRIOR_FLIP,
        "b": grp["visual_flip_rate"] <= MAX_VISUAL_FLIP,
        "f": len(single_ranks) == len(SINGLE_HEADS)
        and all(rank <= TOP_K_SINGLE for rank in single_ranks.values()),
        "_single_ranks": single_ranks,
    }


def patching_criteria(path: Path) -> dict:
    payload = json.loads(path.read_text())
    result = payload["result"]
    p2v, v2p, n = mean_restoration_grids(result["records"])
    cls = classify_heads(p2v, v2p, sigma=result.get("sigma", 2.0))
    ours = set(cls["promoting"])
    theirs = {tuple(h) for h in result["reference_promoting"]}
    ranks = rank_heads(cls["pc1_scores"])
    n_total = p2v.size
    perm = jaccard_permutation_test(ours, theirs, n_heads=n_total, n_perm=N_PERM, seed=0)
    their_ranks = sorted(ranks[h] for h in theirs)
    median_rank = float(np.median(their_ranks))
    stored = {tuple(h) for h in result["promoting"]}
    print(f"patching  {path}  (commit {payload['commit']}, {n} objects scored)")
    if stored != ours:
        print("  WARNING: promoting set recomputed from records differs from the stored one")
    print(f"  our promoting ({len(ours)}): " + ", ".join(format_head_label(h) for h in sorted(ours)))
    print(f"  our suppressing ({len(cls['suppressing'])}): "
          + ", ".join(format_head_label(h) for h in sorted(cls["suppressing"])))
    print(f"  their 9 found: {len(ours & theirs)}  containment {containment(theirs, ours):.2f}")
    print(f"  Jaccard {perm['jaccard']:.3f}, chance {perm['null_mean']:.3f}, p = {perm['p_value']:.4f}")
    print(f"  PC1 ranks of their promoting heads: {their_ranks}  median {median_rank:g}")
    return {
        "c": len(ours & theirs) >= MIN_FOUND_OF_THEIR_9,
        "d": perm["p_value"] < MAX_PERM_P,
        "e": median_rank <= MAX_MEDIAN_RANK,
    }


def main(knockout_path: str | None, patching_path: str | None) -> None:
    results: dict = {}
    for path, fn in ((knockout_path, knockout_criteria), (patching_path, patching_criteria)):
        if path and Path(path).exists():
            results.update(fn(Path(path)))
            print()
        elif path:
            print(f"missing: {path}\n")

    labels = {
        "a": f"(a) group prior flip >= {MIN_PRIOR_FLIP:.1%}",
        "b": f"(b) group visual flip <= {MAX_VISUAL_FLIP:.0%}",
        "c": f"(c) >= {MIN_FOUND_OF_THEIR_9} of their 9 promoting heads found",
        "d": f"(d) Jaccard beats chance, p < {MAX_PERM_P}",
        "e": f"(e) median PC1 rank of their 9 <= {MAX_MEDIAN_RANK}",
        "f": f"(f) L26H5 and L31H3 in the top {TOP_K_SINGLE} single heads",
    }
    print("D40 criteria")
    for key, label in labels.items():
        print(f"  {label:48s} {verdict(results.get(key))}")
    if all(results.get(k) is not None for k in labels):
        if all(results[k] for k in labels):
            overall = "PASS"
        elif not (results["a"] and results["b"] and results["d"]):
            overall = "FAIL -- stop and debug (plan section 4)"
        else:
            overall = "PARTIAL -- proceed with our own P set (D68), report every deviation"
        print(f"\nD40 verdict: {overall}")
    else:
        print("\nD40 verdict: incomplete (not every criterion has run)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: analyze_replication_gate.py <knockout.json> <patching.json>")
    main(sys.argv[1], sys.argv[2])
