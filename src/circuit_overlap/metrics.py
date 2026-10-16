"""Metrics for the circuit-overlap study.

Implements the quantities defined in docs/plan_v2.md section 3:

* ``logit_difference``        -- Delta = logit(t_co) - logit(t_ccf)        (3.4)
* ``restoration_score``       -- R_d(c)                                    (3.4)
* ``flip_rate``               -- group/single-head ablation flip rate      (3.4)
* ``jaccard``                 -- J_AB on promoting head sets               (3.5)
* ``containment``             -- |A & B| / |A|, to compare with Lietzow's 61%  (2, 3.5)
* ``permutation_null_jaccard``/``jaccard_permutation_test``
                              -- chance floor, p-value, normalised J       (3.5)

Pure NumPy on purpose: these run on CPU in unit tests and on Kaggle alike, and
every function takes array-likes, so torch tensors work via ``np.asarray``
(detach and move to CPU first).

Head sets are collections of hashable head identifiers. Use ``(layer, head)``
tuples, 0-based, matching the convention recorded in docs/DECISIONS.md (D30).
Never hard-code head indices here; load them from configs/heads_*.json.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping

import numpy as np

__all__ = [
    "logit_difference",
    "restoration_score",
    "flip_rate",
    "jaccard",
    "containment",
    "permutation_null_jaccard",
    "jaccard_permutation_test",
]


def logit_difference(logits, idx_co: int, idx_ccf: int) -> np.ndarray:
    """Delta = logit(t_co) - logit(t_ccf) at the last token position.

    Parameters
    ----------
    logits:
        Array-like of shape ``(..., vocab_size)``. Raw logits, not softmaxed:
        the restoration score is defined on logits (plan 3.4).
    idx_co, idx_ccf:
        Vocabulary ids of the original-colour and counterfactual-colour tokens.
        Both must be single tokens (plan 3.1: colour words are restricted to
        those that tokenise to a single first token).

    Returns
    -------
    ``np.ndarray`` of shape ``logits.shape[:-1]``; a 0-d array for 1-d input.
    """
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("logits must have a vocabulary axis")
    vocab = arr.shape[-1]
    for name, idx in (("idx_co", idx_co), ("idx_ccf", idx_ccf)):
        if not (-vocab <= int(idx) < vocab):
            raise IndexError(f"{name}={idx} out of range for vocab_size={vocab}")
    return arr[..., int(idx_co)] - arr[..., int(idx_ccf)]


def restoration_score(delta_patched, delta_target, delta_source, eps: float = 1e-6):
    """Restoration score R_d(c) (plan 3.4).

        R = (Delta_patched - Delta_target) / (Delta_source - Delta_target)

    R = 0 means the patch left the run at the target (unpatched) behaviour;
    R = 1 means it fully restored the source behaviour. Values outside [0, 1]
    are meaningful (over- and under-shoot) and are not clipped.

    The denominator is the contrast size for that example. Where it is smaller
    than ``eps`` the example carries no signal and R is undefined; those entries
    are returned as ``nan`` rather than raising, so a batch is not lost to a few
    degenerate examples. Filter them with ``np.isnan`` before averaging.

    Broadcasting follows NumPy rules, so scalars mix with arrays.
    """
    patched = np.asarray(delta_patched, dtype=np.float64)
    target = np.asarray(delta_target, dtype=np.float64)
    source = np.asarray(delta_source, dtype=np.float64)
    if eps < 0:
        raise ValueError("eps must be non-negative")

    denom = source - target
    degenerate = np.abs(denom) < eps
    safe_denom = np.where(degenerate, 1.0, denom)
    result = (patched - target) / safe_denom
    return np.where(degenerate, np.nan, result)


def flip_rate(pred_before, pred_after, mask=None) -> float:
    """Fraction of examples whose prediction changed (plan 3.4).

    Used for group and single-head zero-ablation flip rates. ``pred_before`` and
    ``pred_after`` are per-example answers of any comparable dtype (token ids,
    colour-word strings, booleans).

    ``mask`` restricts the denominator to the examples the plan says count --
    for P, the "correctly conflicting" ones; for T, objects the model gets right
    with a neutral note and wrong with the incongruent note (plan 3.4, D27).

    To get a sign-flip rate on the logit difference instead, pass
    ``np.sign(delta_before)`` and ``np.sign(delta_after)``.

    Returns ``nan`` when no example is selected -- an empty selection has no
    rate, and 0.0 would silently read as "nothing flipped".
    """
    before = np.asarray(pred_before)
    after = np.asarray(pred_after)
    if before.shape != after.shape:
        raise ValueError(
            f"pred_before and pred_after must have the same shape, "
            f"got {before.shape} and {after.shape}"
        )

    changed = before != after
    if mask is not None:
        sel = np.asarray(mask, dtype=bool)
        if sel.shape != before.shape:
            raise ValueError(
                f"mask shape {sel.shape} does not match predictions {before.shape}"
            )
        changed = changed[sel]

    if changed.size == 0:
        return float("nan")
    return float(np.mean(changed))


def _as_head_set(heads: Iterable[Hashable], name: str) -> frozenset:
    if isinstance(heads, (str, bytes)):
        raise TypeError(f"{name} must be a collection of head ids, not a string")
    try:
        return frozenset(heads)
    except TypeError as exc:  # unhashable head ids, e.g. lists
        raise TypeError(
            f"{name} must contain hashable head ids such as (layer, head) tuples"
        ) from exc


def jaccard(set_a: Iterable[Hashable], set_b: Iterable[Hashable]) -> float:
    """J_AB = |A & B| / |A | B| on promoting head sets (plan 3.5).

    Duplicates and ordering are irrelevant; the inputs are treated as sets.

    Convention: two empty sets give 0.0, not 1.0. Discovery returning no heads
    in either condition is not evidence of a shared circuit.
    """
    a = _as_head_set(set_a, "set_a")
    b = _as_head_set(set_b, "set_b")
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def containment(set_a: Iterable[Hashable], set_b: Iterable[Hashable]) -> float:
    """|A & B| / |A|: the fraction of A's heads that are also in B (plan 2, 3.5).

    This is the quantity behind Lietzow et al.'s "61% of V-classified heads were
    also classified under P" -- a containment fraction, not a Jaccard. Report it
    alongside ``jaccard`` whenever that number is cited (D34).

    Asymmetric: ``containment(a, b) != containment(b, a)`` in general.
    Returns ``nan`` for empty A, which has no fraction to report.
    """
    a = _as_head_set(set_a, "set_a")
    b = _as_head_set(set_b, "set_b")
    if not a:
        return float("nan")
    return len(a & b) / len(a)


def _check_perm_args(size_a: int, size_b: int, n_heads: int, n_perm: int) -> None:
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    if n_perm <= 0:
        raise ValueError("n_perm must be positive")
    for name, size in (("size_a", size_a), ("size_b", size_b)):
        if size < 0:
            raise ValueError(f"{name} must be non-negative")
        if size > n_heads:
            raise ValueError(f"{name}={size} exceeds n_heads={n_heads}")


def permutation_null_jaccard(
    size_a: int,
    size_b: int,
    n_heads: int,
    n_perm: int = 10_000,
    seed: int | None = 0,
) -> np.ndarray:
    """Null distribution of J for two random head sets of fixed sizes (plan 3.5).

    The chance floor depends on both set sizes and on the number of heads in the
    model (Qwen2.5-VL-3B: 36 x 16 = 576, to be verified against the config), so
    it must be recomputed per comparison rather than quoted once.

    Parameters
    ----------
    size_a, size_b:
        Sizes of the two head sets, normally ``len(H_P)`` and ``len(H_T)``.
    n_heads:
        Total heads available for selection (layers x heads per layer).
    n_perm:
        Number of random draws.
    seed:
        Seed for ``np.random.default_rng``; fix it so results are reproducible
        and record it in the result JSON (plan 7).

    Returns
    -------
    ``np.ndarray`` of shape ``(n_perm,)`` with one J per draw.
    """
    _check_perm_args(size_a, size_b, n_heads, n_perm)
    if size_a == 0 or size_b == 0:
        # Every draw has an empty intersection; J is 0 by the convention above.
        return np.zeros(n_perm, dtype=np.float64)

    rng = np.random.default_rng(seed)

    def draw_masks(size: int) -> np.ndarray:
        masks = np.zeros((n_perm, n_heads), dtype=bool)
        if size == n_heads:
            masks[:] = True
            return masks
        scores = rng.random((n_perm, n_heads), dtype=np.float32)
        idx = np.argpartition(scores, size - 1, axis=1)[:, :size]
        np.put_along_axis(masks, idx, True, axis=1)
        return masks

    mask_a = draw_masks(int(size_a))
    mask_b = draw_masks(int(size_b))

    intersection = np.count_nonzero(mask_a & mask_b, axis=1)
    union = size_a + size_b - intersection
    return intersection / union


def jaccard_permutation_test(
    set_a: Iterable[Hashable],
    set_b: Iterable[Hashable],
    n_heads: int,
    n_perm: int = 10_000,
    seed: int | None = 0,
) -> Mapping[str, float]:
    """Observed J against its permutation null (plan 3.5).

    Returns a dict with:

    ``jaccard``
        Observed J.
    ``null_mean``, ``null_std``
        Mean and standard deviation of the chance floor.
    ``p_value``
        One-sided ``(1 + #{null >= observed}) / (1 + n_perm)``. The +1 keeps the
        p-value away from exactly 0, which no finite permutation test can support.
    ``normalised_jaccard``
        Observed / null_mean, the random-normalised overlap used by Nikankin et
        al. for circuit IoU (D33). ``nan`` when the null mean is 0.
    ``n_a``, ``n_b``, ``n_intersection``, ``n_heads``, ``n_perm``
        Bookkeeping, so a result JSON can record what was compared.

    Note this tests J against chance only. A J that beats chance may still sit
    far below the split-half ceiling; both bounds are required (plan 3.5, 6.1).
    """
    a = _as_head_set(set_a, "set_a")
    b = _as_head_set(set_b, "set_b")
    _check_perm_args(len(a), len(b), n_heads, n_perm)

    observed = jaccard(a, b)
    null = permutation_null_jaccard(len(a), len(b), n_heads, n_perm=n_perm, seed=seed)

    null_mean = float(np.mean(null))
    p_value = float((1 + np.count_nonzero(null >= observed)) / (1 + n_perm))
    normalised = observed / null_mean if null_mean > 0 else float("nan")

    return {
        "jaccard": observed,
        "null_mean": null_mean,
        "null_std": float(np.std(null)),
        "p_value": p_value,
        "normalised_jaccard": normalised,
        "n_a": len(a),
        "n_b": len(b),
        "n_intersection": len(a & b),
        "n_heads": int(n_heads),
        "n_perm": int(n_perm),
    }
