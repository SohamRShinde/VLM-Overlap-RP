"""Split-half ceiling: how well does the discovery pipeline agree with itself?

Plan section 3.5 and risk 1, parameters fixed in DECISIONS.md D69.

A circuit found from ~70 examples is an *estimate*. Run the same discovery on
a different sample of the same condition and the head set will not come back
identical, so even P compared with P does not give J = 1. Without knowing how
far it falls short, a low J_PT cannot be read: it might mean "different
circuits", or only "same circuit, noisy estimate".

The ceiling measures that self-agreement directly. Split the objects into two
random halves, run the **identical** pipeline on each half
(``patching.mean_restoration_grids`` + ``patching.classify_heads``), and
compare the two results. Repeat over many splits. J_PT is then read on the
scale from the chance floor (random head sets) to this ceiling (D70).

A half holds about half the objects, so the ceiling is **conservative**: the
full sample agrees with itself better than two halves do. For the all-head
rank correlation the Spearman-Brown formula gives the full-length estimate;
for a Jaccard of thresholded sets there is no such correction, so it is
reported as is.

numpy only, so it runs on the CPU from saved results.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from .metrics import jaccard, permutation_null_jaccard
from .patching import classify_heads, head_restoration_scores, mean_restoration_grids

__all__ = [
    "rankdata",
    "spearman",
    "spearman_brown",
    "split_half_ceiling",
]


def rankdata(values) -> np.ndarray:
    """1-based ranks, ties given their average rank (as scipy's default)."""
    x = np.asarray(values, dtype=float).ravel()
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman rank correlation of two equal-length arrays (any shape)."""
    ra, rb = rankdata(a), rankdata(b)
    if ra.size != rb.size:
        raise ValueError(f"arrays differ in size: {ra.size} vs {rb.size}")
    if ra.size < 2:
        raise ValueError("need at least 2 values")
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom == 0.0:
        return float("nan")  # a constant array has no rank order
    return float((ra * rb).sum() / denom)


def spearman_brown(r: float, factor: float = 2.0) -> float:
    """Reliability of a measure ``factor`` times as long as the one giving ``r``.

    Split-half correlations describe half-length samples; ``factor=2`` gives the
    estimate for the full sample. Only meaningful for a correlation, not for a
    Jaccard of thresholded sets.
    """
    if not np.isfinite(r):
        return float("nan")
    return float(factor * r / (1.0 + (factor - 1.0) * r))


def _summary(values: Sequence[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "p2_5": float(np.percentile(v, 2.5)),
        "p97_5": float(np.percentile(v, 97.5)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def _label(head: tuple[int, int]) -> str:
    return f"L{head[0]}H{head[1]}"


def split_half_ceiling(
    records: Sequence[Mapping],
    n_splits: int = 1000,
    seed: int = 0,
    sigma: float = 2.0,
    n_perm: int = 10_000,
) -> dict:
    """Self-agreement of head discovery across random halves of the objects.

    records:
        Per-object patching records, as stored in
        ``results/replication_patching_*.json`` (the fields
        ``head_restoration_scores`` reads). Degenerate examples are dropped
        exactly as the full pipeline drops them.

    Returns a JSON-ready dict: per-statistic summaries (median and central 95%
    interval), the raw per-split values, how often each head is selected, the
    full-sample classification for reference, and the permutation chance floor
    at the observed half-set sizes.
    """
    usable = [r for r in records if head_restoration_scores(r) is not None]
    n = len(usable)
    if n < 4:
        raise ValueError(f"need at least 4 non-degenerate records, got {n}")
    if n_splits < 1:
        raise ValueError("n_splits must be positive")

    full = classify_heads(*mean_restoration_grids(usable)[:2], sigma=sigma)
    n_layers, n_heads = np.asarray(full["pc1_scores"]).shape

    rng = np.random.default_rng(seed)
    half = n // 2
    j_prom, j_supp, rho, sizes = [], [], [], []
    selected: Counter = Counter()
    for _ in range(n_splits):
        perm = rng.permutation(n)
        halves = [
            classify_heads(
                *mean_restoration_grids([usable[i] for i in idx])[:2], sigma=sigma
            )
            for idx in (perm[:half], perm[half:])
        ]
        a, b = halves
        j_prom.append(jaccard(a["promoting"], b["promoting"]))
        j_supp.append(jaccard(a["suppressing"], b["suppressing"]))
        rho.append(spearman(a["pc1_scores"], b["pc1_scores"]))
        for h in halves:
            sizes.append(len(h["promoting"]))
            selected.update(h["promoting"])

    half_runs = 2 * n_splits
    frequency = {
        _label(head): count / half_runs
        for head, count in sorted(selected.items(), key=lambda kv: (-kv[1], kv[0]))
    }

    # Chance floor at the half-set sizes actually observed.
    typical = int(round(float(np.median(sizes))))
    floor = permutation_null_jaccard(
        typical, typical, n_layers * n_heads, n_perm=n_perm, seed=seed
    )

    rho_summary = _summary(rho)
    return {
        "n_objects": n,
        "n_dropped_degenerate": len(records) - n,
        "half_sizes": [half, n - half],
        "n_splits": n_splits,
        "seed": seed,
        "sigma": sigma,
        "n_heads_total": n_layers * n_heads,
        "full_sample": {
            "promoting": [_label(h) for h in full["promoting"]],
            "suppressing": [_label(h) for h in full["suppressing"]],
        },
        "jaccard_promoting": _summary(j_prom),
        "jaccard_suppressing": _summary(j_supp),
        "pc1_spearman": rho_summary,
        "pc1_spearman_brown_full_length": spearman_brown(rho_summary["median"]),
        "promoting_set_size_per_half": _summary(sizes),
        "share_of_splits_with_identical_promoting_sets": float(
            np.mean(np.isclose(j_prom, 1.0))
        ),
        "selection_frequency_promoting": frequency,
        "chance_floor": {
            "set_size": typical,
            "mean": float(floor.mean()),
            "p95": float(np.percentile(floor, 95)),
            "n_perm": n_perm,
        },
        "per_split": {
            "jaccard_promoting": [float(x) for x in j_prom],
            "jaccard_suppressing": [float(x) for x in j_supp],
            "pc1_spearman": [float(x) for x in rho],
        },
    }
