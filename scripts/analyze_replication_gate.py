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

D40 = {"label": "D40", "min_prior_flip": MIN_PRIOR_FLIP, "max_visual_flip": MAX_VISUAL_FLIP,
       "min_found": MIN_FOUND_OF_THEIR_9, "max_perm_p": MAX_PERM_P,
       "max_median_rank": MAX_MEDIAN_RANK, "top_k_single": TOP_K_SINGLE,
       "single_heads": SINGLE_HEADS, "n_perm": N_PERM}


def verdict(ok: bool | None) -> str:
    return "NOT RUN" if ok is None else ("pass" if ok else "FAIL")


def knockout_criteria(path: Path, gate: dict = D40) -> dict:
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
    single_ranks = {h: ordered.index(h) + 1 for h in gate["single_heads"] if h in ordered}
    return {
        "a": grp["prior_flip_rate"] >= gate["min_prior_flip"],
        "b": grp["visual_flip_rate"] <= gate["max_visual_flip"],
        "f": len(single_ranks) == len(gate["single_heads"])
        and all(rank <= gate["top_k_single"] for rank in single_ranks.values()),
        "_single_ranks": single_ranks,
    }


def patching_criteria(path: Path, gate: dict = D40) -> dict:
    payload = json.loads(path.read_text())
    result = payload["result"]
    p2v, v2p, n = mean_restoration_grids(result["records"])
    cls = classify_heads(p2v, v2p, sigma=result.get("sigma", 2.0))
    ours = set(cls["promoting"])
    theirs = {tuple(h) for h in result["reference_promoting"]}
    ranks = rank_heads(cls["pc1_scores"])
    n_total = p2v.size
    perm = jaccard_permutation_test(ours, theirs, n_heads=n_total, n_perm=gate["n_perm"], seed=0)
    their_ranks = sorted(ranks[h] for h in theirs)
    median_rank = float(np.median(their_ranks))
    stored = {tuple(h) for h in result["promoting"]}
    print(f"patching  {path}  (commit {payload['commit']}, {n} objects scored)")
    if stored != ours:
        print("  WARNING: promoting set recomputed from records differs from the stored one")
    print(f"  our promoting ({len(ours)}): " + ", ".join(format_head_label(h) for h in sorted(ours)))
    print(f"  our suppressing ({len(cls['suppressing'])}): "
          + ", ".join(format_head_label(h) for h in sorted(cls["suppressing"])))
    print(f"  their {len(theirs)} found: {len(ours & theirs)}  containment {containment(theirs, ours):.2f}")
    print(f"  Jaccard {perm['jaccard']:.3f}, chance {perm['null_mean']:.3f}, p = {perm['p_value']:.4f}")
    print(f"  PC1 ranks of their promoting heads: {their_ranks}  median {median_rank:g}")
    return {
        "c": len(ours & theirs) >= gate["min_found"],
        "d": perm["p_value"] < gate["max_perm_p"],
        "e": median_rank <= gate["max_median_rank"],
        "_n_theirs": len(theirs),
    }


def main(knockout_path: str | None, patching_path: str | None, gate: dict = D40) -> dict:
    results: dict = {}
    for path, fn in ((knockout_path, knockout_criteria), (patching_path, patching_criteria)):
        if path and Path(path).exists():
            results.update(fn(Path(path), gate))
            print()
        elif path:
            print(f"missing: {path}\n")

    n = results.get("_n_theirs", 9)
    singles = " and ".join(format_head_label(h) for h in gate["single_heads"])
    labels = {
        "a": f"(a) group prior flip >= {gate['min_prior_flip']:.1%}",
        "b": f"(b) group visual flip <= {gate['max_visual_flip']:.0%}",
        "c": f"(c) >= {gate['min_found']} of their {n} promoting heads found",
        "d": f"(d) Jaccard beats chance, p < {gate['max_perm_p']}",
        "e": f"(e) median PC1 rank of their {n} <= {gate['max_median_rank']}",
        "f": f"(f) {singles} in the top {gate['top_k_single']} single heads",
    }
    print(f"{gate['label']} criteria")
    for key, label in labels.items():
        print(f"  {label:48s} {verdict(results.get(key))}")
    if all(results.get(k) is not None for k in labels):
        if all(results[k] for k in labels):
            overall = "PASS"
        elif not (results["a"] and results["b"] and results["d"]):
            overall = "FAIL -- stop and debug (plan section 4)"
        else:
            overall = "PARTIAL -- proceed with our own P set (D68), report every deviation"
        print(f"\n{gate['label']} verdict: {overall}")
    else:
        overall = "incomplete (not every criterion has run)"
        print(f"\n{gate['label']} verdict: {overall}")
    return {"criteria": {k: results.get(k) for k in labels}, "verdict": overall}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: analyze_replication_gate.py <knockout.json> <patching.json>")
    main(sys.argv[1], sys.argv[2])
