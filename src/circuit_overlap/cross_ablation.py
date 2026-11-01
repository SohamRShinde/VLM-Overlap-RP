"""H2 cross-ablation (plan 3.6), designed and read by D73.

Does ablating the P circuit defeat the typographic override, and does
ablating the T circuit defeat the prompted prior? Each effect is judged
against random head sets with the same number of heads in each layer.

Written and committed before any H2 run, like ``overlap.py``.

* ``build_plan``: the head sets (loaded from the discovery results), the
  layer-matched random draws, and which configurations run on which inputs.
* ``ablate_heads_last_token``: the intervention, stacked
  ``patching.patch_head_last_token`` hooks with a zero (or mean) value.
* ``analyse_cross_ablation``: D73's tests, intervals and wording.

Record format, one per object and condition, as in Kush's knockout
(``replication_knockout_qwen25vl3b.json``): ``record[f"{config}/prior"]`` and
``record[f"{config}/visual"]`` are ``[logit of the source run's argmax token,
logit of the target run's argmax token]`` with that configuration ablated, on
the source and target run respectively. Read "prior" as the source run (the
competing signal wins: Prior prompt in P, incongruent note in T) and
"visual" as the target run (the image wins), per ``t_patching.T_ROLES``.

torch is imported lazily, so everything but the hook runs on a CPU box.
"""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, contextmanager
from typing import Iterable, Mapping, Sequence

import numpy as np

from .heads import format_head_label
from .patching import patch_head_last_token

__all__ = [
    "SETS",
    "ALPHA",
    "H2_VERDICTS",
    "TESTS",
    "head_sets",
    "excluded_heads",
    "layer_matched_draws",
    "build_plan",
    "plan_cost",
    "ablate_heads_last_token",
    "source_flips",
    "target_flips",
    "null_test",
    "analyse_cross_ablation",
]

# The five fixed sets, and the prefix of the random draws matched to each.
SETS = {"P": "rP", "T": "rT", "shared": "rS", "P_only": "rPo", "T_only": "rTo"}
ALPHA = 0.025                    # Bonferroni over the two primary tests (D73)
N_DRAWS, N_MEAN_DRAWS, N_CLEAN_DRAWS = 100, 50, 20

# D73's wording, verbatim. Do not edit after H2 results exist.
H2_VERDICTS = {
    "transfer": "ablating H_A defeats the B override more than layer-matched random heads",
    "null": "not distinguishable from layer-matched random heads at this n",
}

# (test name, ablated set, condition whose objects it runs on, tier)
TESTS = [
    ("P->T", "P", "T", "primary"),
    ("T->P", "T", "P", "primary"),
    ("P->P", "P", "P", "diagonal"),
    ("T->T", "T", "T", "diagonal"),
    ("shared->P", "shared", "P", "decomposition"),
    ("shared->T", "shared", "T", "decomposition"),
    ("P_only->T", "P_only", "T", "decomposition"),
    ("T_only->P", "T_only", "P", "decomposition"),
]


# --- the plan ----------------------------------------------------------------


def _heads(raw: Iterable) -> set:
    return {tuple(int(x) for x in h) for h in raw}


def head_sets(p_promoting: Iterable, t_promoting: Iterable) -> dict[str, list]:
    """H_P, H_T and their decomposition, each sorted."""
    p, t = _heads(p_promoting), _heads(t_promoting)
    return {"P": sorted(p), "T": sorted(t), "shared": sorted(p & t),
            "P_only": sorted(p - t), "T_only": sorted(t - p)}


def excluded_heads(*results: Mapping) -> set:
    """Every head classified, promoting or suppressing, in any of ``results``."""
    out = set()
    for r in results:
        out |= _heads(r["promoting"]) | _heads(r["suppressing"])
    return out


def layer_matched_draws(heads: Sequence, exclude: Iterable, n_heads_per_layer: int,
                        n_draws: int, seed: int, stream: int) -> list[list[tuple]]:
    """Random head sets with as many heads in each layer as ``heads`` has.

    Within a layer, heads are drawn uniformly without replacement from those
    not in ``exclude``. ``stream`` gives each matched set its own random
    sequence, so adding a set never changes another set's draws.
    """
    exclude = _heads(exclude)
    counts = Counter(l for l, _ in _heads(heads))
    pools = {l: [h for h in range(n_heads_per_layer) if (l, h) not in exclude] for l in counts}
    for l, k in counts.items():
        if len(pools[l]) < k:
            raise ValueError(f"layer {l}: need {k} unclassified heads, only {len(pools[l])} exist")
    rng = np.random.default_rng([seed, stream])
    draws = []
    for _ in range(n_draws):
        draw = []
        for l in sorted(counts):
            picks = rng.choice(len(pools[l]), counts[l], replace=False)
            draw += [(l, pools[l][k]) for k in sorted(picks)]
        draws.append(draw)
    return draws


def _draw_names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{k:03d}" for k in range(n)]


def build_plan(p_result: Mapping, t_result: Mapping, n_heads_per_layer: int = 16,
               n_draws: int = N_DRAWS, n_mean_draws: int = N_MEAN_DRAWS,
               n_clean_draws: int = N_CLEAN_DRAWS, seed: int = 0) -> dict:
    """Everything the notebook runs, as data (D73).

    ``p_result`` / ``t_result``: the ``result`` blocks of the P and T
    discovery files (they carry ``promoting`` and ``suppressing``).

    Returns ``configs`` (name -> heads, JSON-ready) and, per phase, the list
    of ``[config, runs]`` to run, ``runs`` being ``"both"`` or ``"source"``:

    * ``zero[cond]``: zero ablation on condition ``cond``'s objects;
    * ``mean[cond]``: mean ablation, for the primary tests;
    * ``clean``: zero ablation on the clean image, one run per object.
    """
    if not n_mean_draws <= n_draws:
        raise ValueError("mean-ablation draws are the first n_mean_draws of the zero draws")
    sets = head_sets(p_result["promoting"], t_result["promoting"])
    exclude = excluded_heads(p_result, t_result)
    configs: dict[str, list] = {"none": []}
    configs.update({name: heads for name, heads in sets.items()})
    for stream, (name, prefix) in enumerate(SETS.items()):
        if not sets[name]:
            continue
        for draw_name, draw in zip(_draw_names(prefix, n_draws),
                                   layer_matched_draws(sets[name], exclude, n_heads_per_layer,
                                                       n_draws, seed, stream)):
            configs[draw_name] = draw

    zero = {"P": {}, "T": {}}
    for cond in zero:
        for name in ["none", *SETS]:
            zero[cond][name] = "both"
        for name in ("P", "T"):                  # primary and diagonal nulls, plus collateral
            for d in _draw_names(SETS[name], n_draws):
                zero[cond][d] = "both"
    for _, s, cond, tier in TESTS:
        if tier == "decomposition" and sets[s]:
            for d in _draw_names(SETS[s], n_draws):
                zero[cond].setdefault(d, "source")

    mean = {"P": {}, "T": {}}
    for test, s, cond, tier in TESTS:
        if tier != "primary":
            continue
        for name in SETS:
            mean[cond][name] = "both"
        for d in _draw_names(SETS[s], n_mean_draws):
            mean[cond][d] = "source"

    clean = ["none", *SETS] + _draw_names("rP", n_clean_draws) + _draw_names("rT", n_clean_draws)

    def ready(heads):
        return [list(h) for h in heads]

    return {
        "decision": "D73",
        "seed": seed,
        "n_draws": n_draws, "n_mean_draws": n_mean_draws, "n_clean_draws": n_clean_draws,
        "sets": {k: [format_head_label(h) for h in v] for k, v in sets.items()},
        "excluded_from_draws": sorted(format_head_label(h) for h in exclude),
        "configs": {k: ready(v) for k, v in configs.items()},
        "zero": {c: [[k, v] for k, v in d.items() if k in configs] for c, d in zero.items()},
        "mean": {c: [[k, v] for k, v in d.items() if k in configs] for c, d in mean.items()},
        "clean": [k for k in clean if k in configs],
    }


def plan_cost(plan: Mapping, n_objects: Mapping[str, int], n_clean: int) -> dict:
    """Last-token runs per phase, for the notebook's runtime estimate."""
    def runs(entries):
        return sum(2 if r == "both" else 1 for _, r in entries)
    return {
        "zero": sum(n_objects[c] * runs(plan["zero"][c]) for c in n_objects),
        "mean": sum(n_objects[c] * runs(plan["mean"][c]) for c in n_objects),
        "clean": n_clean * len(plan["clean"]),
        "prefills": 2 * 2 * sum(n_objects.values()) + n_clean,
    }


# --- the intervention (GPU side) -------------------------------------------


@contextmanager
def ablate_heads_last_token(model, heads: Iterable, head_dim: int,
                            values: Mapping | None = None):
    """Replace each head's ``o_proj`` input slice at the last position.

    With ``values=None`` the slice is zeroed (CLAUDE.md's ablation rule);
    otherwise ``values[(layer, head)]`` is written in (mean ablation). Hooks
    on one layer chain, so several heads in a layer are all replaced.
    """
    heads = sorted(_heads(heads))
    with ExitStack() as stack:
        if heads and values is None:
            import torch
            zero = torch.zeros(head_dim)
        for layer, head in heads:
            value = zero if values is None else values[(layer, head)]
            stack.enter_context(patch_head_last_token(model, layer, head, head_dim, value))
        yield


# --- analysis (CPU side) ------------------------------------------------------


def source_flips(records: Sequence[Mapping], config: str) -> np.ndarray:
    """1 where the source run stops being source-grounded (D67; ties stay)."""
    return np.array([float(r[f"{config}/prior"][1] > r[f"{config}/prior"][0]) for r in records])


def target_flips(records: Sequence[Mapping], config: str) -> np.ndarray:
    """1 where the target run stops being target-grounded (D67)."""
    return np.array([float(not r[f"{config}/visual"][1] > r[f"{config}/visual"][0])
                     for r in records])


def logit_diff_change(records: Sequence[Mapping], config: str) -> np.ndarray:
    """Per object: (visual - prior answer logit) on the source run, ablated minus none.

    Positive = the ablation moved the source run toward the image's answer.
    """
    def diff(r, c):
        lp, lv = r[f"{c}/prior"]
        return lv - lp
    return np.array([diff(r, config) - diff(r, "none") for r in records])


def null_test(observed: np.ndarray, null: np.ndarray, n_boot: int, rng,
              alpha: float = ALPHA) -> dict:
    """Observed per-object outcomes against a matrix of draws (D73 (5), (8)).

    ``observed``: shape (n,); ``null``: shape (R, n), one row per draw, on
    the same objects. p = (1 + #draws >= observed) / (R + 1).
    """
    observed, null = np.asarray(observed, float), np.asarray(null, float)
    n, r = observed.size, null.shape[0]
    obs = float(observed.mean())
    null_rates = null.mean(axis=1)
    p = (1 + int((null_rates >= obs).sum())) / (r + 1)
    boot_obs, boot_excess = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        o = observed[idx].mean()
        boot_obs.append(o)
        boot_excess.append(o - null[:, idx].mean())
    return {
        "n": n,
        "n_draws": r,
        "observed": obs,
        "count": int(observed.sum()),
        "ci95": [float(np.percentile(boot_obs, 2.5)), float(np.percentile(boot_obs, 97.5))],
        "null_mean": float(null_rates.mean()),
        "null_p95": float(np.percentile(null_rates, 95)),
        "null_max": float(null_rates.max()),
        "p": p,
        "excess": obs - float(null_rates.mean()),
        "excess_ci95": [float(np.percentile(boot_excess, 2.5)),
                        float(np.percentile(boot_excess, 97.5))],
        "passes": p < alpha,
        # 1 / (R + 1): when this is not below alpha the test cannot pass.
        "min_attainable_p": 1 / (r + 1),
    }


def _wording(test: str, passes: bool) -> str:
    a, b = test.split("->")
    if not passes:
        return f"ablating H_{a}: {H2_VERDICTS['null']}"
    return H2_VERDICTS["transfer"].replace("H_A", f"H_{a}").replace("the B", f"the {b}")


def _ok(records: Iterable[Mapping]) -> list[Mapping]:
    return [r for r in records if r.get("status") == "ok"]


def analyse_cross_ablation(plan: Mapping, records: Mapping[str, Sequence[Mapping]],
                           clean_records: Sequence[Mapping] = (), n_boot: int = 1000,
                           seed: int = 0, alpha: float = ALPHA) -> dict:
    """D73's tests on the saved records.

    ``records``: ``{"P": [...], "T": [...]}``, one record per object, phase
    results merged (zero-ablation keys ``"P/prior"``, mean-ablation keys
    ``"mean:P/prior"``). Only records with ``status == "ok"`` count.
    """
    rng = np.random.default_rng(seed)
    recs = {c: _ok(v) for c, v in records.items()}
    configs = plan["configs"]

    def draws(prefix, n, cond, key_prefix=""):
        names = [d for d in _draw_names(prefix, n) if d in configs]
        return [d for d in names if f"{key_prefix}{d}/prior" in recs[cond][0]] if recs[cond] else []

    def run_test(test, s, cond, tier, key_prefix="", n_null=None):
        rs = recs[cond]
        if not rs or not configs.get(s):
            return None
        names = draws(SETS[s], n_null or plan["n_draws"], cond, key_prefix)
        if not names:
            return None
        null = np.stack([source_flips(rs, key_prefix + d) for d in names])
        out = null_test(source_flips(rs, key_prefix + s), null, n_boot, rng, alpha)
        out.update(tier=tier, ablated=s, condition=cond, heads=plan["sets"][s],
                   wording=_wording(test, out["passes"]))
        ld = logit_diff_change(rs, key_prefix + s)
        ld_null = np.array([logit_diff_change(rs, key_prefix + d).mean() for d in names])
        out["logit_diff_change"] = {"mean": float(ld.mean()), "null_mean": float(ld_null.mean()),
                                    "null_p95": float(np.percentile(ld_null, 95))}
        return out

    tests = {name: run_test(name, s, c, tier) for name, s, c, tier in TESTS}

    # Mean ablation: the primary tests only, against the first n_mean_draws.
    mean_tests = {name: run_test(name, s, c, tier, key_prefix="mean:", n_null=plan["n_mean_draws"])
                  for name, s, c, tier in TESTS if tier == "primary"}
    robust = None
    if all(mean_tests.values()) and all(tests[n] for n in mean_tests):
        robust = all(mean_tests[n]["passes"] == tests[n]["passes"] for n in mean_tests)

    # Descriptive: the condition-specific sets on their own condition (no null).
    own = {}
    for name, s, cond in (("P_only->P", "P_only", "P"), ("T_only->T", "T_only", "T")):
        if recs[cond] and configs.get(s):
            f = source_flips(recs[cond], s)
            own[name] = {"n": f.size, "observed": float(f.mean()), "count": int(f.sum())}

    # Collateral: target-run flips under every fixed set, with both primary nulls.
    collateral = {}
    for cond, rs in recs.items():
        if not rs:
            continue
        null = {p: np.array([target_flips(rs, d).mean() for d in draws(p, plan["n_draws"], cond)
                             if f"{d}/visual" in rs[0]]) for p in ("rP", "rT")}
        collateral[cond] = {
            s: {"target_flip_rate": float(target_flips(rs, s).mean()),
                "null_mean_rP": float(null["rP"].mean()) if null["rP"].size else None,
                "null_mean_rT": float(null["rT"].mean()) if null["rT"].size else None}
            for s in ["none", *SETS] if configs.get(s) is not None and f"{s}/visual" in rs[0]
        }

    clean = None
    cl = [r for r in clean_records if r.get("status") == "ok" and r.get("none/correct")]
    if cl:
        names = [k for k in plan["clean"] if f"{k}/correct" in cl[0]]
        acc = {k: float(np.mean([r[f"{k}/correct"] for r in cl])) for k in names}
        clean = {
            "n_correct_unablated": len(cl),
            "n_objects": sum(1 for r in clean_records if r.get("status") == "ok"),
            "accuracy_kept": {k: acc[k] for k in ["none", *SETS] if k in acc},
            "random_mean_rP": float(np.mean([acc[k] for k in acc if k.startswith("rP_")] or [np.nan])),
            "random_mean_rT": float(np.mean([acc[k] for k in acc if k.startswith("rT_")] or [np.nan])),
        }

    transfer_ratio = {}
    for cross, diag in (("P->T", "T->T"), ("T->P", "P->P")):
        if tests[cross] and tests[diag] and tests[diag]["observed"] > 0:
            transfer_ratio[cross] = tests[cross]["observed"] / tests[diag]["observed"]

    return {
        "alpha": alpha,
        "n_objects": {c: len(v) for c, v in recs.items()},
        "n_dropped": {c: sum(1 for r in v if r.get("status") != "ok") for c, v in records.items()},
        "sets": plan["sets"],
        "primary": {n: tests[n] for n, *_, tier in TESTS if tier == "primary"},
        "diagonal": {n: tests[n] for n, *_, tier in TESTS if tier == "diagonal"},
        "decomposition": {n: tests[n] for n, *_, tier in TESTS if tier == "decomposition"},
        "own_condition_subsets": own,
        "transfer_ratio": transfer_ratio,
        "mean_ablation": {"tests": mean_tests, "same_wording_as_zero": robust},
        "collateral_target_flips": collateral,
        "clean_accuracy": clean,
        "parameters": {"n_boot": n_boot, "seed": seed},
    }
