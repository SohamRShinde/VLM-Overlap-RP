"""V-circuit discovery and the V overlaps: the P pipeline on Lietzow's visual contrast.

Design and reading rule pre-registered in DECISIONS.md D75 before this code or
any V run existed. The GPU work reuses ``patching.LastTokenRunner`` /
``patch_example`` / ``classify_heads`` **unchanged** (D30, D68), so P, T and V
come out of one pipeline and only the pair of inputs differs:

    P: Prior prompt vs Visual prompt, on the counterfactual image
    T: incongruent-note vs neutral-note image, both with the Visual prompt
    V: original image vs counterfactual image, both with the Visual prompt

V is Lietzow's ``VISUAL_CIRCUIT_CONTRAST`` (their ``src/vdpo/types/contrast.py``,
commit 145b453): source = (Visual, original), target = (Visual, counterfactual).

**Roles, not names.** ``patch_example`` calls its runs "prior" (source) and
"visual" (target). For V those field names are kept, so the records feed the
same scoring and classification code, but they mean:

    source  "prior"  = original image       -- answers the real colour
    target  "visual" = counterfactual image -- answers the counterfactual colour

so a promoting V head carries the original image's colour into the
counterfactual run. The token roles match P's (source answers the real
colour) and are swapped relative to T's (source answers the note's colour),
while the image-variation is shared with T (D75 (2)).

The analysis half (``analyse_v``) computes D75 (6)-(9) from saved records on
the CPU; torch is never imported here.
"""

from __future__ import annotations

import copy
from typing import Iterable, Mapping, Sequence

import numpy as np

from .ceiling import split_half_ceiling
from .cross_ablation import head_sets
from .heads import format_head_label
from .metrics import containment, jaccard
from .overlap import VERDICTS, analyse_overlap, classify_records
from .replication import color_token_sets
from .t_patching import sweep_order

__all__ = [
    "V_ROLES",
    "V_CAP",
    "V_VERDICTS",
    "DJ_WORDING",
    "is_v_conflicting",
    "v_sweep_order",
    "binding_ceiling",
    "delta_j_bootstrap",
    "dj_verdict",
    "analyse_v",
]

V_ROLES = {
    "source": "original image (answers the real colour); record fields *_prior",
    "target": "counterfactual image (answers the counterfactual colour); record fields *_visual",
    "prompt": "Visual prompt for both runs",
}

# D75 (4): T's n, so V and T rest on equally many objects.
V_CAP = 244

# D75 (7): D70's phrases, with the ceiling phrase no longer naming P.
V_VERDICTS = {**VERDICTS, "ceiling": "not distinguishable from self-agreement"}

# D75 (8), verbatim.
DJ_WORDING = {
    "greater": "J_VT > J_PT",
    "less": "J_VT < J_PT",
    "none": "no difference between J_VT and J_PT detectable at this n",
}


def is_v_conflicting(row: Mapping, top1_original: int, top1_counterfactual: int,
                     color_ids: Mapping[str, Sequence[int]],
                     ambiguous: Iterable[str] = ()) -> bool:
    """V's selection (D75 (3)): each image's answer follows the image.

    The original image's top-1 is one of the real colours and the
    counterfactual image's top-1 is the counterfactual colour, both as
    first-token ids. Objects using a colour that shares a first token with
    another (D26) are excluded. Mirrors Lietzow's ``visual_original_ok`` and
    ``visual_cf_ok`` in ``load_intersection_example_ids``.
    """
    if set(ambiguous) & ({row["attack"]} | set(row["correct"])):
        return False
    attack, real = color_token_sets(row, color_ids)
    return top1_original in real and top1_counterfactual in attack


def v_sweep_order(selected: Sequence[int], p_ids: Iterable[int], seed: int = 0,
                  cap: int = V_CAP) -> list[int]:
    """D75 (4): objects shared with P first, the rest in seeded order, then the cap.

    The objects shared with P are Lietzow's own V subset (theirs also requires
    the Prior prompt to answer the real colour), so they lead: a cut-off run
    still holds them, followed by an unbiased random sample.
    """
    if cap < 1:
        raise ValueError("cap must be positive")
    return sweep_order(selected, p_ids, seed=seed)[:cap]


def binding_ceiling(first: Mapping, second: Mapping) -> dict:
    """``first``'s ceiling with its promoting-J summary replaced by the lower one.

    D75 (7): J_PV and J_VT are read against the lower of the two conditions'
    ceiling 2.5th percentiles, since the noisier circuit limits how high the
    overlap can go. Everything else ``analyse_overlap`` reads from its ceiling
    argument (the first condition's stable core and rank reliability) stays
    the first condition's own.
    """
    out = copy.deepcopy(dict(first))
    a, b = first["jaccard_promoting"], second["jaccard_promoting"]
    out["jaccard_promoting"] = dict(b if b["p2_5"] < a["p2_5"] else a)
    out["binding"] = "second" if b["p2_5"] < a["p2_5"] else "first"
    return out


def delta_j_bootstrap(p_records: Sequence[Mapping], v_records: Sequence[Mapping],
                      t_records: Sequence[Mapping], n_boot: int = 1000, seed: int = 0,
                      sigma: float = 2.0) -> dict:
    """D75 (8): dJ = J_VT - J_PT, with T resampled once per replicate.

    T appears in both terms, so each replicate draws one T resample and uses
    it for both Jaccards; P and V are drawn independently. Percentile 95% CI.
    """
    p_prom = set(classify_records(p_records, sigma)["promoting"])
    v_prom = set(classify_records(v_records, sigma)["promoting"])
    t_prom = set(classify_records(t_records, sigma)["promoting"])
    point = jaccard(v_prom, t_prom) - jaccard(p_prom, t_prom)

    rng = np.random.default_rng(seed)

    def resample(records):
        return [records[k] for k in rng.integers(0, len(records), len(records))]

    boot = []
    for _ in range(n_boot):
        bt = set(classify_records(resample(t_records), sigma)["promoting"])
        bp = set(classify_records(resample(p_records), sigma)["promoting"])
        bv = set(classify_records(resample(v_records), sigma)["promoting"])
        boot.append(jaccard(bv, bt) - jaccard(bp, bt))
    lo, hi = (float(np.percentile(boot, q)) for q in (2.5, 97.5))
    verdict = dj_verdict(lo, hi)
    return {
        "delta_j": point,
        "j_vt": jaccard(v_prom, t_prom),
        "j_pt": jaccard(p_prom, t_prom),
        "ci95": [lo, hi],
        "verdict": verdict,
        "wording": DJ_WORDING[verdict],
        "bootstrap_mean": float(np.mean(boot)),
        "point_inside_ci": bool(lo <= point <= hi),
        "n_boot": n_boot,
        "seed": seed,
        "bootstrap": [float(x) for x in boot],
    }


def dj_verdict(ci_low: float, ci_high: float) -> str:
    """D75 (8)'s rule: the CI's position relative to 0."""
    if not ci_low <= ci_high:
        raise ValueError(f"CI is not ordered: [{ci_low}, {ci_high}]")
    if ci_low > 0:
        return "greater"
    if ci_high < 0:
        return "less"
    return "none"


def _labels(heads) -> list[str]:
    return sorted(format_head_label(h) for h in heads)


def _reword(block: dict) -> dict:
    """Swap D70's P-specific ceiling phrase for D75's general one."""
    block = dict(block)
    block["wording"] = V_VERDICTS[block["verdict"]]
    return block


def analyse_v(p_records: Sequence[Mapping], v_records: Sequence[Mapping],
              t_records: Sequence[Mapping], p_ceiling: Mapping,
              p_result: Mapping, t_result: Mapping, n_boot: int = 1000, seed: int = 0,
              sigma: float = 2.0, n_splits: int = 1000, n_perm: int = 10_000,
              core_threshold: float = 0.9) -> dict:
    """Everything D75 (6)-(9) asks for.

    p_ceiling:
        The ``result`` block of ``results/split_half_ceiling_p_*.json`` (D69).
    p_result, t_result:
        The ``result`` blocks of the P and T discovery files, for the shared
        and T-only head sets (``cross_ablation.head_sets``, as D73/D74 use).
    """
    if min(len(p_records), len(v_records), len(t_records)) < 4:
        raise ValueError("need at least 4 records per condition")

    # (6) V's own split-half ceiling, D69's parameters; T's, for the binding rule.
    v_ceiling = split_half_ceiling(v_records, n_splits=n_splits, seed=seed, sigma=sigma,
                                   n_perm=n_perm)
    t_ceiling = split_half_ceiling(t_records, n_splits=n_splits, seed=seed, sigma=sigma,
                                   n_perm=n_perm)

    kw = dict(n_boot=n_boot, seed=seed, sigma=sigma, t_splits=n_splits, n_perm=n_perm,
              core_threshold=core_threshold)
    # (7) J_PV: P first (so the shared-object check is Lietzow's intersection),
    # J_VT: V first. Each read against the binding ceiling.
    pv_ceiling = binding_ceiling(p_ceiling, v_ceiling)
    vt_ceiling = binding_ceiling(v_ceiling, t_ceiling)
    pv = analyse_overlap(p_records, v_records, pv_ceiling, **kw)
    vt = analyse_overlap(v_records, t_records, vt_ceiling, **kw)
    pv["primary"], vt["primary"] = _reword(pv["primary"]), _reword(vt["primary"])

    # (8) the primary test of H1-specific.
    dj = delta_j_bootstrap(p_records, v_records, t_records, n_boot=n_boot, seed=seed,
                           sigma=sigma)

    # (9) descriptive.
    p_cls, v_cls = classify_records(p_records, sigma), classify_records(v_records, sigma)
    p_prom, v_prom = set(p_cls["promoting"]), set(v_cls["promoting"])
    p_all = p_prom | set(p_cls["suppressing"])
    v_all = v_prom | set(v_cls["suppressing"])
    p_ids = {r["i"] for r in p_records}
    inter = [r for r in v_records if r["i"] in p_ids]
    inter_block = None
    if len(inter) >= 4:
        vi = classify_records(inter, sigma)
        vi_prom = set(vi["promoting"])
        vi_all = vi_prom | set(vi["suppressing"])
        inter_block = {
            "n_objects": len(inter),
            "v_promoting": _labels(vi_prom),
            "containment_v_in_p_promoting": containment(vi_prom, p_prom),
            "containment_v_in_p_classified": containment(vi_all, p_all),
        }
    sets = head_sets(p_result["promoting"], t_result["promoting"])
    shared, t_only = set(map(tuple, sets["shared"])), set(map(tuple, sets["T_only"]))

    return {
        "n_objects": {"p": len(p_records), "v": len(v_records), "t": len(t_records)},
        "roles": V_ROLES,
        "v_promoting": _labels(v_prom),
        "v_suppressing": _labels(v_cls["suppressing"]),
        "v_ceiling": v_ceiling,
        "t_ceiling_summary": {
            "jaccard_promoting": t_ceiling["jaccard_promoting"],
            "half_sizes": t_ceiling["half_sizes"],
        },
        "j_pv": {**pv, "ceiling_binding": "P" if pv_ceiling["binding"] == "first" else "V"},
        "j_vt": {**vt, "ceiling_binding": "V" if vt_ceiling["binding"] == "first" else "T"},
        "delta_j": dj,
        "descriptive": {
            "containment_v_in_p_promoting": containment(v_prom, p_prom),
            "containment_v_in_p_classified": containment(v_all, p_all),
            "lietzow_intersection_subset": inter_block,
            "shared_heads_in_v": {"n": len(shared & v_prom), "of": len(shared),
                                  "heads": _labels(shared & v_prom)},
            "t_only_heads_in_v": {"n": len(t_only & v_prom), "of": len(t_only),
                                  "heads": _labels(t_only & v_prom)},
        },
        "parameters": {"n_boot": n_boot, "seed": seed, "sigma": sigma,
                       "n_splits": n_splits, "n_perm": n_perm,
                       "core_threshold": core_threshold, "cap": V_CAP},
    }
