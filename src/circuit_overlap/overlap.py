"""P-T circuit overlap, read by D70's rule (implementation details: D72).

The headline of the paper. Written and committed **before** any T-circuit
result existed, so the bootstrap, the chance floor and the wording cannot have
been tuned to the answer.

What it computes, from per-object patching records of the two conditions:

* **J_PT** (primary, D70): Jaccard of the P and T promoting sets, with a
  bootstrap 95% CI over objects, P and T resampled independently -- they are
  different object samples.
* **The D70 verdict**, one of four fixed phrases, from where that CI sits
  between the chance floor and the P split-half ceiling.
* **Stable-core J** (D70): Jaccard of the heads each condition selects in at
  least 90% of its own split-half runs. Guards against a head sitting on the
  2-sigma cut (L31H1 does, in P).
* **All-head PC1 rank correlation** (secondary, D70), against P's own ceiling.
  Reported as *landscape* similarity, not circuit overlap, and without a
  bootstrap interval: on known-answer controls built from the P data
  (D72) it stayed at 0.974 with 7 of 9 circuit heads moved, so it barely
  responds to circuit identity, and its resampling distribution is biased
  low (bootstrap mean ~0.8 against a true 1.0).
* Descriptive extras from plan 3.5 that do not enter the verdict: suppressing
  J, containment both ways (comparable with Lietzow's 61%), random-normalised
  J, a shared-object check, and a per-layer-band profile.

numpy only; runs on the CPU from saved results.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .ceiling import spearman, split_half_ceiling
from .heads import format_head_label, parse_head_label
from .metrics import containment, jaccard, jaccard_permutation_test
from .patching import classify_heads, mean_restoration_grids

__all__ = [
    "VERDICTS",
    "d70_verdict",
    "classify_records",
    "stable_core",
    "layer_profile",
    "analyse_overlap",
]

# D70's four phrases, verbatim. Do not edit after T results exist.
VERDICTS = {
    "chance": "not distinguishable from chance at this n",
    "ceiling": "not distinguishable from the P circuit's own self-agreement",
    "inconclusive": "inconclusive at this n",
    "between": "above chance, below self-agreement",
}


def d70_verdict(ci_low: float, ci_high: float, floor_p95: float, ceiling_p2_5: float) -> str:
    """D70's wording rule, as a pure function of four numbers.

    The CI "reaches" the floor when its lower end is at or below the floor's
    95th percentile, and "reaches" the ceiling when its upper end is at or
    above the ceiling's 2.5th percentile. Touching both claims neither.
    """
    if not ci_low <= ci_high:
        raise ValueError(f"CI is not ordered: [{ci_low}, {ci_high}]")
    if not floor_p95 < ceiling_p2_5:
        raise ValueError(
            f"floor p95 {floor_p95} is not below ceiling p2.5 {ceiling_p2_5}: "
            "the ruler has no length, so no reading is possible"
        )
    at_floor = ci_low <= floor_p95
    at_ceiling = ci_high >= ceiling_p2_5
    if at_floor and at_ceiling:
        return "inconclusive"
    if at_floor:
        return "chance"
    if at_ceiling:
        return "ceiling"
    return "between"


def classify_records(records: Sequence[Mapping], sigma: float = 2.0) -> dict:
    """The discovery pipeline on a list of records: mean grids, then PCA cut."""
    p2v, v2p, _ = mean_restoration_grids(records)
    return classify_heads(p2v, v2p, sigma=sigma)


def stable_core(selection_frequency: Mapping[str, float], threshold: float = 0.9) -> set:
    """Heads selected in at least ``threshold`` of split-half runs."""
    return {parse_head_label(h) for h, f in selection_frequency.items() if f >= threshold}


def layer_profile(p_set, t_set, n_layers: int, band: int = 6) -> list[dict]:
    """Promoting heads per band of layers, and how many the two share."""
    p_set, t_set = set(p_set), set(t_set)
    out = []
    for start in range(0, n_layers, band):
        layers = range(start, min(start + band, n_layers))
        p = {h for h in p_set if h[0] in layers}
        t = {h for h in t_set if h[0] in layers}
        out.append({"layers": f"{layers[0]}-{layers[-1]}", "p": len(p), "t": len(t),
                    "shared": len(p & t)})
    return out


def _ci(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _resample(records: Sequence[Mapping], rng) -> list[Mapping]:
    return [records[k] for k in rng.integers(0, len(records), len(records))]


def analyse_overlap(
    p_records: Sequence[Mapping],
    t_records: Sequence[Mapping],
    p_ceiling: Mapping,
    n_boot: int = 1000,
    seed: int = 0,
    sigma: float = 2.0,
    t_splits: int = 1000,
    n_perm: int = 10_000,
    core_threshold: float = 0.9,
) -> dict:
    """Everything D70 asks for, plus plan 3.5's descriptive extras.

    p_ceiling:
        The ``result`` block of ``results/split_half_ceiling_p_*.json`` (D69).
        Its 2.5th percentile is read from the file, never typed in.
    """
    if len(p_records) < 4 or len(t_records) < 4:
        raise ValueError("need at least 4 records per condition")

    p_cls, t_cls = classify_records(p_records, sigma), classify_records(t_records, sigma)
    p_prom, t_prom = set(p_cls["promoting"]), set(t_cls["promoting"])
    p_supp, t_supp = set(p_cls["suppressing"]), set(t_cls["suppressing"])
    n_layers, n_heads = np.asarray(p_cls["pc1_scores"]).shape
    n_total = n_layers * n_heads

    # --- primary: J_PT with an independent-resampling bootstrap ------------
    rng = np.random.default_rng(seed)
    boot_j = []
    for _ in range(n_boot):
        bp = classify_records(_resample(p_records, rng), sigma)
        bt = classify_records(_resample(t_records, rng), sigma)
        boot_j.append(jaccard(bp["promoting"], bt["promoting"]))
    j_lo, j_hi = _ci(boot_j)

    # Chance floor at the OBSERVED set sizes (plan 3.5; D72).
    perm = jaccard_permutation_test(p_prom, t_prom, n_heads=n_total, n_perm=n_perm, seed=seed)
    floor = np.sort(np.asarray(_null_values(len(p_prom), len(t_prom), n_total, n_perm, seed)))
    floor_p95 = float(np.percentile(floor, 95))
    ceiling_p2_5 = float(p_ceiling["jaccard_promoting"]["p2_5"])
    verdict = d70_verdict(j_lo, j_hi, floor_p95, ceiling_p2_5)

    # --- stable cores --------------------------------------------------------
    t_half = split_half_ceiling(t_records, n_splits=t_splits, seed=seed, sigma=sigma,
                                n_perm=min(n_perm, 1000))
    p_core = stable_core(p_ceiling["selection_frequency_promoting"], core_threshold)
    t_core = stable_core(t_half["selection_frequency_promoting"], core_threshold)

    # --- secondary: all-head rank correlation --------------------------------
    rho = spearman(p_cls["pc1_scores"], t_cls["pc1_scores"])
    p_rel = float(p_ceiling["pc1_spearman_brown_full_length"])
    t_rel = float(t_half["pc1_spearman_brown_full_length"])

    # --- shared objects: is sample selection driving J? ----------------------
    shared_ids = sorted({r["i"] for r in p_records} & {r["i"] for r in t_records})
    shared = None
    if len(shared_ids) >= 4:
        ps = [r for r in p_records if r["i"] in set(shared_ids)]
        ts = [r for r in t_records if r["i"] in set(shared_ids)]
        sp, st = classify_records(ps, sigma), classify_records(ts, sigma)
        j_shared = jaccard(sp["promoting"], st["promoting"])
        # Paired bootstrap: the same objects in both conditions.
        by_p = {r["i"]: r for r in ps}
        by_t = {r["i"]: r for r in ts}
        boot_shared = []
        for _ in range(n_boot):
            pick = [shared_ids[k] for k in rng.integers(0, len(shared_ids), len(shared_ids))]
            boot_shared.append(jaccard(
                classify_records([by_p[i] for i in pick], sigma)["promoting"],
                classify_records([by_t[i] for i in pick], sigma)["promoting"],
            ))
        s_lo, s_hi = _ci(boot_shared)
        shared = {
            "n_objects": len(shared_ids),
            "j_pt": j_shared,
            "ci95": [s_lo, s_hi],
            "p_promoting": sorted(format_head_label(h) for h in sp["promoting"]),
            "t_promoting": sorted(format_head_label(h) for h in st["promoting"]),
            # D72: flag when the full-sample J lies outside the shared-object CI.
            "selection_effect_flag": not (s_lo <= perm["jaccard"] <= s_hi),
        }

    def labels(s):
        return sorted(format_head_label(h) for h in s)

    return {
        "n_p_objects": len(p_records),
        "n_t_objects": len(t_records),
        "n_heads_total": n_total,
        "p_promoting": labels(p_prom),
        "t_promoting": labels(t_prom),
        "shared_promoting": labels(p_prom & t_prom),
        "primary": {
            "j_pt": perm["jaccard"],
            "ci95": [j_lo, j_hi],
            "chance_floor_p95_at_observed_sizes": floor_p95,
            "chance_floor_mean": perm["null_mean"],
            "set_sizes": [len(p_prom), len(t_prom)],
            "ceiling_p2_5": ceiling_p2_5,
            "ceiling_median": float(p_ceiling["jaccard_promoting"]["median"]),
            "verdict": verdict,
            "wording": VERDICTS[verdict],
            "p_value_vs_chance": perm["p_value"],
            "normalised_j": perm["normalised_jaccard"],
            # Bootstrap diagnostics (D72). On the known-answer controls the
            # bias was within +/-0.01 except at J = 1, a boundary.
            "bootstrap_mean": float(np.mean(boot_j)),
            "bootstrap_bias": float(np.mean(boot_j) - perm["jaccard"]),
            "point_inside_ci": bool(j_lo <= perm["jaccard"] <= j_hi),
        },
        "stable_core": {
            "threshold": core_threshold,
            "p_core": labels(p_core),
            "t_core": labels(t_core),
            "j_core": jaccard(p_core, t_core),
            "t_self_agreement_jaccard": t_half["jaccard_promoting"],
            "t_selection_frequency": t_half["selection_frequency_promoting"],
        },
        "rank_correlation": {
            "rho": rho,
            "reading": "whole-landscape similarity of all 576 heads' PC1 scores; "
                       "insensitive to circuit identity (D72 controls), not an overlap measure",
            "p_ceiling_half_length": float(p_ceiling["pc1_spearman"]["median"]),
            "p_ceiling_full_length": p_rel,
            "t_reliability_full_length": t_rel,
            # Classical attenuation bound for a correlation between two noisy
            # measures; descriptive only, not part of D70.
            "attenuation_ceiling": float(np.sqrt(max(p_rel, 0) * max(t_rel, 0))),
        },
        "descriptive": {
            "j_suppressing": jaccard(p_supp, t_supp),
            "p_suppressing": labels(p_supp),
            "t_suppressing": labels(t_supp),
            "containment_p_in_t": containment(p_prom, t_prom),
            "containment_t_in_p": containment(t_prom, p_prom),
            "layer_profile": layer_profile(p_prom, t_prom, n_layers),
        },
        "shared_objects": shared,
        "parameters": {"n_boot": n_boot, "seed": seed, "sigma": sigma,
                       "t_splits": t_splits, "n_perm": n_perm},
        "bootstrap": {"j_pt": [float(x) for x in boot_j]},
    }


def _null_values(size_a: int, size_b: int, n_total: int, n_perm: int, seed: int):
    from .metrics import permutation_null_jaccard

    return permutation_null_jaccard(size_a, size_b, n_total, n_perm=n_perm, seed=seed)
