"""H3 mechanism (plan 3.7), designed and read by D74.

How do the T circuit's heads make the note win? Two measurements per head,
both at the last token, on the cached last-token path:

* **attention mass** of the last-token query on the note's image tokens, the
  object's, and all image tokens (eager attention: sdpa returns no weights);
* **direct effect**: the head's own output pushed through the final norm and
  ``lm_head`` (logit lens), as the logit of the source answer minus the
  target answer (T: note colour minus real colour).

Each set's value is judged against D73's layer-matched random draws, reused
from ``cross_ablation.build_plan`` so H2 and H3 compare against the same heads.

Written and committed before any H3 run, like ``cross_ablation.py``.

Record format, one per object and condition: ``record["runs"][run]`` holds,
per run, the run's top-1 id, ``pair`` (logits of the two answer tokens),
token counts, and ``(n_layers_stored, n_heads)`` grids ``m_image``,
``m_note`` (``None`` on P runs, which have no note), ``m_object`` and ``dla``,
for layers ``FIRST_LAYER`` onward.

torch is imported lazily, so everything but the GPU helpers runs on a CPU box.
The GPU helpers are checked on the real model by the notebook's section 2.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Mapping, Sequence

import numpy as np

from .cross_ablation import H2_VERDICTS, SETS, _draw_names, _heads
from .tokens import coverage_on_grid

__all__ = [
    "ALPHA",
    "FIRST_LAYER",
    "T_RUNS",
    "P_RUNS",
    "SMALL_INC",
    "SMALL_NEU",
    "H3_TESTS",
    "H3_VERDICTS",
    "MECHANISM",
    "image_token_layout",
    "token_sets",
    "attention_masses",
    "round_grid",
    "capture_last_token_attention",
    "capture_final_residual",
    "cached_values",
    "head_direct_effects",
    "measure_run",
    "check_attention_reconstructs_heads",
    "check_direct_effect_formula",
    "set_values",
    "directed_null_test",
    "analyse_h3",
]

ALPHA = 0.025                  # Bonferroni over the two primary tests (D74 (6))
FIRST_LAYER = 20               # layers 20-35 are stored (D74 (3))

LAYOUTS = ("on_object_small", "on_object_medium", "beside_object")
T_RUNS = tuple(f"{l}/{v}" for l in LAYOUTS for v in ("incongruent", "neutral")) + ("clean",)
P_RUNS = ("prior", "visual")
SMALL_INC, SMALL_NEU = "on_object_small/incongruent", "on_object_small/neutral"

# (name, set, condition, measure, run, baseline run or None, direction, tier).
# direction "greater": p counts draws >= observed; "less": draws <= observed.
H3_TESTS = [
    ("A", "T_only", "T", "m_note", SMALL_INC, None, "greater", "primary"),
    ("W", "T_only", "T", "dla", SMALL_INC, None, "greater", "primary"),
    ("A:shared", "shared", "T", "m_note", SMALL_INC, None, "greater", "secondary"),
    ("W:shared", "shared", "T", "dla", SMALL_INC, None, "greater", "secondary"),
    ("A:P_only", "P_only", "T", "m_note", SMALL_INC, None, "greater", "secondary"),
    ("W:P_only", "P_only", "T", "dla", SMALL_INC, None, "greater", "secondary"),
    ("A:T", "T", "T", "m_note", SMALL_INC, None, "greater", "secondary"),
    ("W:T", "T", "T", "dla", SMALL_INC, None, "greater", "secondary"),
    ("A_rise:T_only", "T_only", "T", "m_note", SMALL_INC, SMALL_NEU, "greater", "secondary"),
    ("A_rise:shared", "shared", "T", "m_note", SMALL_INC, SMALL_NEU, "greater", "secondary"),
    ("object:clean", "T_only", "T", "m_object", "clean", None, "greater", "secondary"),
    ("image_colour:clean", "T_only", "T", "dla", "clean", None, "less", "secondary"),
    ("object:P_visual", "T_only", "P", "m_object", "visual", None, "greater", "secondary"),
    ("image_colour:P_visual", "T_only", "P", "dla", "visual", None, "less", "secondary"),
    ("routing:P", "P", "P", "m_image", "visual", "prior", "greater", "validation"),
]

# D74's wording, verbatim. Do not edit after H3 results exist.
_CLAIMS = {
    "A": "attend to the note",
    "W": "write the note's colour",
    "A_rise": "attend to the note more when it names a colour",
    "object": "attend to the object",
    "image_colour": "write the image's colour",
    "routing": "attend to the image more under the Visual than the Prior prompt",
}
H3_VERDICTS = {
    "pass": "{set} heads {claim} more than layer-matched random heads",
    "null": H2_VERDICTS["null"],
}
MECHANISM = {
    (True, True): "the T-only heads read the note and write its colour",
    (False, True): "the T-only heads write the note's colour without attending to the note "
                   "more than random heads",
    (True, False): "the T-only heads attend to the note but do not directly write its colour",
    (False, False): "the T-only heads neither attend to the note nor directly write its "
                    "colour more than random heads",
}

_SET_LABEL = {"T_only": "T-only", "shared": "shared", "P_only": "P-only", "T": "H_T", "P": "H_P"}


# --- token positions (CPU) -----------------------------------------------------


def image_token_layout(input_ids, position_ids, image_token_id: int, rows: int, cols: int) -> int:
    """Check the image block and return the position of its first token.

    ``input_ids``: the prompt's ids, shape ``(n,)``. ``position_ids``: the
    prefill's M-RoPE positions, shape ``(3 or 4, n')`` with the last three
    rows (temporal, height, width), covering at least the image block.
    Raises unless the image tokens are contiguous, there are ``rows * cols``
    of them, and token ``k`` sits at grid row ``k // cols``, column
    ``k % cols`` -- the row-major convention ``tokens.py`` assumes.
    """
    ids = np.asarray(input_ids).reshape(-1)
    where = np.flatnonzero(ids == image_token_id)
    if where.size != rows * cols:
        raise ValueError(f"{where.size} image tokens, grid {rows}x{cols} = {rows * cols}")
    if where.size == 0 or (np.diff(where) != 1).any():
        raise ValueError("image tokens are not one contiguous block")
    pos = np.asarray(position_ids)
    pos = pos.reshape(pos.shape[0], -1)
    if pos.shape[0] not in (3, 4) or pos.shape[1] <= where[-1]:
        raise ValueError(f"position ids of shape {pos.shape} do not cover the image block")
    h, w = pos[-2, where], pos[-1, where]
    k = np.arange(where.size)
    if not ((h - h.min() == k // cols).all() and (w - w.min() == k % cols).all()):
        raise ValueError("image tokens are not in row-major order of the merged grid")
    return int(where[0])


def token_sets(note_mask, object_mask, rows: int, cols: int, offset: int,
               threshold: float = 0.5) -> dict[str, np.ndarray]:
    """Sequence positions of the image, note and object tokens (D74 (3)).

    *note*: note coverage >= threshold. *object*: object coverage >=
    threshold and **no** note coverage, so tokens under the note's edge
    belong to neither. ``note_mask`` may be ``None`` (no note): the note set
    is then empty and the object set is the object tokens.
    """
    n = rows * cols
    image = offset + np.arange(n)
    note_cov = (np.zeros((rows, cols)) if note_mask is None
                else coverage_on_grid(note_mask, rows, cols))
    obj_cov = (np.zeros((rows, cols)) if object_mask is None
               else coverage_on_grid(object_mask, rows, cols))
    note = np.flatnonzero(note_cov.reshape(-1) >= threshold)
    obj = np.flatnonzero((obj_cov.reshape(-1) >= threshold) & (note_cov.reshape(-1) == 0))
    return {"image": image, "note": offset + note, "object": offset + obj}


def attention_masses(weights, sets: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Summed attention per head on each token set.

    ``weights``: ``(n_heads, n_keys)`` post-softmax weights of the last-token
    query. Returns ``{set: (n_heads,)}``.
    """
    w = np.asarray(weights, dtype=np.float64)
    return {k: w[:, idx].sum(axis=1) for k, idx in sets.items()}


def round_grid(grid, digits: int = 4) -> list:
    """A grid as nested lists, ``digits`` significant figures, for JSON."""
    return [[float(f"{v:.{digits}g}") for v in row] for row in np.asarray(grid, float)]


# --- GPU side --------------------------------------------------------------


def _layers(model):
    return model.model.language_model.layers


@contextmanager
def capture_last_token_attention(model, layers: Iterable[int]):
    """``{layer: tensor(n_heads, n_keys)}``: the last query's attention weights.

    Needs ``attn_implementation="eager"``: sdpa returns no weights, which is
    caught here rather than recorded as zeros.
    """
    weights: dict[int, object] = {}
    handles = []

    def make_hook(layer):
        def hook(module, args, output):
            w = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if w is None:
                raise RuntimeError("attention weights are None: load the model with "
                                   "attn_implementation='eager' (D74 (3))")
            weights[layer] = w[0, :, -1, :].detach().float()
        return hook

    try:
        for layer in layers:
            handles.append(_layers(model)[layer].self_attn.register_forward_hook(make_hook(layer)))
        yield weights
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_final_residual(model):
    """``{"x": tensor(hidden)}``: the last position's residual entering the final norm."""
    out: dict[str, object] = {}

    def hook(module, args):
        out["x"] = args[0][0, -1].detach().clone()

    handle = model.model.language_model.norm.register_forward_pre_hook(hook)
    try:
        yield out
    finally:
        handle.remove()


def cached_values(cache, layer: int):
    """Value states of one layer in a KV cache, ``(batch, kv_heads, n, head_dim)``."""
    if hasattr(cache, "layers"):
        return cache.layers[layer].values
    if hasattr(cache, "value_cache"):
        return cache.value_cache[layer]
    return cache[layer][1]


def _rms_scale(model, x):
    import torch

    norm = model.model.language_model.norm
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    return torch.rsqrt(x.float().pow(2).mean() + eps)


def head_direct_effects(model, acts: Mapping[int, object], x_final, tok_a: int, tok_b: int,
                        layers: Sequence[int], head_dim: int) -> np.ndarray:
    """Each head's direct effect on logit(tok_a) - logit(tok_b), ``(len(layers), n_heads)``.

    A head's output is W_O applied to its slice alone (o_proj has no bias in
    Qwen2), scaled as the final RMSNorm scales the actual residual, then put
    through ``lm_head``. Applying the modules, not their weights, keeps this
    correct whether or not they are quantised.
    """
    import torch

    norm = model.model.language_model.norm
    head_out = model.get_output_embeddings()
    scale = _rms_scale(model, x_final)
    rows = []
    with torch.no_grad():
        for layer in layers:
            z = acts[layer]
            n_heads = z.shape[0] // head_dim
            blocks = torch.zeros(n_heads, z.shape[0], dtype=z.dtype, device=z.device)
            for h in range(n_heads):
                blocks[h, h * head_dim:(h + 1) * head_dim] = z[h * head_dim:(h + 1) * head_dim]
            contrib = _layers(model)[layer].self_attn.o_proj(blocks)        # (n_heads, hidden)
            # As Qwen2RMSNorm: scale in fp32, cast to the residual's dtype, times weight.
            normed = norm.weight * (contrib.float() * scale).to(x_final.dtype)
            logits = head_out(normed).float()
            rows.append((logits[:, tok_a] - logits[:, tok_b]).cpu().numpy())
    return np.stack(rows)


def measure_run(model, runner, sets: Mapping[str, np.ndarray], tok_a: int, tok_b: int,
                layers: Sequence[int], head_dim: int, with_note: bool = True) -> dict:
    """Everything D74 records for one run: answer, masses and direct effects."""
    import torch

    from .patching import capture_last_token_o_proj_inputs

    with capture_last_token_attention(model, layers) as weights, \
            capture_last_token_o_proj_inputs(model) as acts, \
            capture_final_residual(model) as final:
        logits = runner.logits()
    names = ["image", "object"] + (["note"] if with_note else [])
    idx = {k: torch.as_tensor(np.asarray(sets[k]), device=logits.device, dtype=torch.long)
           for k in names}
    grids = {k: [] for k in names}
    for layer in layers:
        w = weights[layer]
        for k in names:
            grids[k].append(w[:, idx[k].to(w.device)].sum(dim=1).cpu().numpy())
    dla = head_direct_effects(model, acts, final["x"], tok_a, tok_b, layers, head_dim)
    return {
        "top1": int(logits.argmax()),
        "pair": [float(logits[tok_a]), float(logits[tok_b])],
        "finite": bool(torch.isfinite(logits).all()) and all(
            bool(np.isfinite(np.stack(g)).all()) for g in grids.values()) and bool(np.isfinite(dla).all()),
        "n_tokens": {k: int(len(sets[k])) for k in names},
        "m_image": round_grid(np.stack(grids["image"])),
        "m_object": round_grid(np.stack(grids["object"])),
        "m_note": round_grid(np.stack(grids["note"])) if with_note else None,
        "dla": round_grid(dla),
    }


def check_attention_reconstructs_heads(model, runner, layers: Sequence[int], head_dim: int,
                                       rtol: float = 2e-2) -> float:
    """Weights x values must give back each head's o_proj input slice.

    Proves the captured weights are the ones the model used, and that head
    ``h`` reads key-value group ``h // (n_heads / n_kv_heads)``. Returns the
    worst relative error; raises above ``rtol``.
    """
    import torch

    from .patching import capture_last_token_o_proj_inputs

    v_last: dict[int, object] = {}
    handles = []
    for layer in layers:
        def hook(module, args, output, layer=layer):
            v_last[layer] = output[0, -1].detach().float()
        handles.append(_layers(model)[layer].self_attn.v_proj.register_forward_hook(hook))
    try:
        with capture_last_token_attention(model, layers) as weights, \
                capture_last_token_o_proj_inputs(model) as acts:
            runner.logits()
    finally:
        for handle in handles:
            handle.remove()
    worst = 0.0
    for layer in layers:
        prefix = cached_values(runner.cache, layer)[0].float()             # (kv, n-1, hd)
        n_kv = prefix.shape[0]
        values = torch.cat([prefix, v_last[layer].view(n_kv, 1, head_dim)], dim=1)
        w = weights[layer]                                                  # (H, n)
        n_heads = w.shape[0]
        rep = n_heads // n_kv
        for h in range(n_heads):
            z = w[h] @ values[h // rep]
            ref = acts[layer][h * head_dim:(h + 1) * head_dim].float()
            err = float((z - ref).norm() / ref.norm().clamp_min(1e-6))
            worst = max(worst, err)
    if worst > rtol:
        raise RuntimeError(f"attention x values does not reconstruct the heads: relative error "
                           f"{worst:.3g} > {rtol}. Do not run H3.")
    return worst


def check_direct_effect_formula(model, runner, tok_a: int, tok_b: int, layer: int,
                                head_dim: int, atol: float = 0.1) -> dict:
    """The logit-lens pieces must reproduce the model.

    (a) ``lm_head(norm.weight * x_final * rms_scale)`` gives the model's own
    logit difference; (b) the per-head outputs of one layer sum to o_proj's
    output on the whole vector. Raises above ``atol``.
    """
    import torch

    from .patching import capture_last_token_o_proj_inputs

    with capture_last_token_o_proj_inputs(model) as acts, capture_final_residual(model) as final:
        logits = runner.logits()
    norm = model.model.language_model.norm
    x = final["x"]
    with torch.no_grad():
        normed = norm.weight * (x.float() * _rms_scale(model, x)).to(x.dtype)
        lens = model.get_output_embeddings()(normed[None]).float()[0]
        o_proj = _layers(model)[layer].self_attn.o_proj
        z = acts[layer]
        n_heads = z.shape[0] // head_dim
        blocks = torch.zeros(n_heads, z.shape[0], dtype=z.dtype, device=z.device)
        for h in range(n_heads):
            blocks[h, h * head_dim:(h + 1) * head_dim] = z[h * head_dim:(h + 1) * head_dim]
        parts = o_proj(blocks).float().sum(dim=0)
        whole = o_proj(z[None]).float()[0]
    out = {
        "logit_diff_model": float(logits[tok_a] - logits[tok_b]),
        "logit_diff_lens": float(lens[tok_a] - lens[tok_b]),
        "head_sum_vs_o_proj": float((parts - whole).abs().max() / whole.abs().max().clamp_min(1e-6)),
    }
    out["lens_error"] = abs(out["logit_diff_model"] - out["logit_diff_lens"])
    if out["lens_error"] > atol or out["head_sum_vs_o_proj"] > 2e-2:
        raise RuntimeError(f"direct-effect pieces disagree with the model: {out}. Do not run H3.")
    return out


# --- analysis (CPU side) ------------------------------------------------------


def _ok(records: Iterable[Mapping]) -> list[Mapping]:
    return [r for r in records if r.get("status") == "ok"]


def set_values(records: Sequence[Mapping], run: str, measure: str, heads: Iterable,
               first_layer: int = FIRST_LAYER) -> np.ndarray:
    """Per object: the set's mean over heads (masses) or sum (``dla``), D74 (5)."""
    heads = sorted(_heads(heads))
    reduce = np.sum if measure == "dla" else np.mean
    out = []
    for r in records:
        grid = r["runs"][run][measure]
        out.append(reduce([grid[l - first_layer][h] for l, h in heads]))
    return np.array(out, dtype=float)


def directed_null_test(observed: np.ndarray, null: np.ndarray, direction: str, n_boot: int,
                       rng, alpha: float = ALPHA) -> dict:
    """``cross_ablation.null_test`` for continuous values, either direction.

    ``observed``: (n,); ``null``: (R, n). ``direction="greater"``:
    p = (1 + #draws >= observed) / (R + 1); ``"less"``: draws <= observed.
    """
    observed, null = np.asarray(observed, float), np.asarray(null, float)
    n, r = observed.size, null.shape[0]
    obs = float(observed.mean())
    null_means = null.mean(axis=1)
    beats = null_means >= obs if direction == "greater" else null_means <= obs
    p = (1 + int(beats.sum())) / (r + 1)
    boot_obs, boot_excess = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        o = observed[idx].mean()
        boot_obs.append(o)
        boot_excess.append(o - null[:, idx].mean())
    return {
        "n": n,
        "n_draws": r,
        "direction": direction,
        "observed": obs,
        "ci95": [float(np.percentile(boot_obs, 2.5)), float(np.percentile(boot_obs, 97.5))],
        "null_mean": float(null_means.mean()),
        "null_p5": float(np.percentile(null_means, 5)),
        "null_p95": float(np.percentile(null_means, 95)),
        "p": p,
        "excess": obs - float(null_means.mean()),
        "excess_ci95": [float(np.percentile(boot_excess, 2.5)),
                        float(np.percentile(boot_excess, 97.5))],
        "passes": p < alpha,
        "min_attainable_p": 1 / (r + 1),
    }


def _wording(name: str, s: str, passes: bool) -> str:
    if not passes:
        return f"{_SET_LABEL[s]} heads: {H3_VERDICTS['null']}"
    claim = _CLAIMS[name.split(":")[0]]
    return H3_VERDICTS["pass"].format(set=_SET_LABEL[s], claim=claim)


def _per_head(recs: Sequence[Mapping], runs: Sequence[str], heads: Iterable,
              first_layer: int) -> dict:
    out = {}
    for l, h in sorted(_heads(heads)):
        row = {}
        for run in runs:
            for m in ("m_note", "m_object", "m_image", "dla"):
                if recs and recs[0]["runs"][run].get(m) is not None:
                    row[f"{run}:{m}"] = float(np.mean(
                        [r["runs"][run][m][l - first_layer][h] for r in recs]))
        out[f"L{l}H{h}"] = row
    return out


def _boot_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int, rng) -> dict:
    out = {"n": [int(a.size), int(b.size)], "a": float(a.mean()) if a.size else None,
           "b": float(b.mean()) if b.size else None}
    if a.size and b.size:
        boots = [a[rng.integers(0, a.size, a.size)].mean() - b[rng.integers(0, b.size, b.size)].mean()
                 for _ in range(n_boot)]
        out.update(diff=float(a.mean() - b.mean()),
                   ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
    return out


def analyse_h3(plan: Mapping, records: Mapping[str, Sequence[Mapping]], n_boot: int = 1000,
               seed: int = 0, alpha: float = ALPHA, first_layer: int = FIRST_LAYER) -> dict:
    """D74's tests and descriptives on the saved records.

    ``plan``: ``cross_ablation.build_plan``'s output (sets and draws).
    ``records``: ``{"P": [...], "T": [...]}``; only ``status == "ok"`` counts.
    """
    rng = np.random.default_rng(seed)
    recs = {c: _ok(v) for c, v in records.items()}
    configs = plan["configs"]

    def values(rs, s_heads, measure, run, base):
        v = set_values(rs, run, measure, s_heads, first_layer)
        if base is not None:
            v = v - set_values(rs, base, measure, s_heads, first_layer)
        return v

    tests = {}
    for name, s, cond, measure, run, base, direction, tier in H3_TESTS:
        rs = recs.get(cond, [])
        if not rs or not configs.get(s):
            tests[name] = None
            continue
        draws = [d for d in _draw_names(SETS[s], plan["n_draws"]) if d in configs]
        null = np.stack([values(rs, configs[d], measure, run, base) for d in draws])
        out = directed_null_test(values(rs, configs[s], measure, run, base), null, direction,
                                 n_boot, rng, alpha)
        out.update(tier=tier, set=s, condition=cond, measure=measure, run=run, baseline=base,
                   heads=plan["sets"][s], wording=_wording(name, s, out["passes"]))
        tests[name] = out

    mechanism = None
    if tests["A"] and tests["W"]:
        mechanism = MECHANISM[(tests["A"]["passes"], tests["W"]["passes"])]

    descriptive: dict = {}
    t = recs.get("T", [])
    if t:
        # Note enrichment: share of image attention on the note / note's share of image tokens.
        frac = np.mean([r["runs"][SMALL_INC]["n_tokens"]["note"] / r["runs"][SMALL_INC]["n_tokens"]["image"]
                        for r in t])
        descriptive["note_token_share"] = float(frac)
        descriptive["note_enrichment"] = {}
        for s in ("T_only", "shared", "P_only", "T"):
            if configs.get(s):
                for run in (SMALL_INC, SMALL_NEU, "clean"):
                    note = set_values(t, run, "m_note", configs[s], first_layer).mean()
                    img = set_values(t, run, "m_image", configs[s], first_layer).mean()
                    descriptive["note_enrichment"][f"{s}:{run}"] = float(note / img / frac) if img > 0 else None
        # Spatial (plan 3.7): T-only note mass by layout and variant, and fooled vs not.
        spatial = {}
        for layout in LAYOUTS:
            inc, neu = f"{layout}/incongruent", f"{layout}/neutral"
            fooled = np.array([bool(r["fooled"].get(layout)) for r in t])
            m_inc = set_values(t, inc, "m_note", configs["T_only"], first_layer)
            m_neu = set_values(t, neu, "m_note", configs["T_only"], first_layer)
            spatial[layout] = {
                "fool_rate": float(fooled.mean()),
                "note_tokens_mean": float(np.mean([r["runs"][inc]["n_tokens"]["note"] for r in t])),
                "n_objects_without_note_tokens": int(sum(r["runs"][inc]["n_tokens"]["note"] == 0 for r in t)),
                "T_only_note_mass_incongruent": float(m_inc.mean()),
                "T_only_note_mass_neutral": float(m_neu.mean()),
                "fooled_minus_not_incongruent": _boot_mean_diff(m_inc[fooled], m_inc[~fooled], n_boot, rng),
            }
        descriptive["spatial"] = spatial
        # Layer profile on the primary runs: classified vs unclassified heads, per layer.
        classified = {tuple(h) for k in ("P", "T") for h in configs.get(k, [])}
        n_layers = len(t[0]["runs"][SMALL_INC]["m_note"])
        n_heads = len(t[0]["runs"][SMALL_INC]["m_note"][0])
        prof = {}
        for li in range(n_layers):
            l = li + first_layer
            mean_by_head = np.mean([r["runs"][SMALL_INC]["m_note"][li] for r in t], axis=0)
            cls = [h for h in range(n_heads) if (l, h) in classified]
            rest = [h for h in range(n_heads) if (l, h) not in classified]
            prof[str(l)] = {"unclassified_mean": float(mean_by_head[rest].mean()) if rest else None,
                            "classified": {f"L{l}H{h}": float(mean_by_head[h]) for h in cls}}
        descriptive["layer_profile_note_mass"] = prof
        descriptive["per_head_T"] = _per_head(t, (SMALL_INC, SMALL_NEU, "clean"),
                                              set(map(tuple, configs["T"])) | set(map(tuple, configs["P"])),
                                              first_layer)
        drift = [r for r in t if "discovery_pair_diff" in r]
        if drift:
            descriptive["eager_vs_discovery_max_abs_logit_T"] = float(max(r["discovery_pair_diff"] for r in drift))
    p = recs.get("P", [])
    if p:
        descriptive["per_head_P"] = _per_head(p, P_RUNS, set(map(tuple, configs["T"])) | set(map(tuple, configs["P"])),
                                              first_layer)
        drift = [r for r in p if "discovery_pair_diff" in r]
        if drift:
            descriptive["eager_vs_discovery_max_abs_logit_P"] = float(max(r["discovery_pair_diff"] for r in drift))

    return {
        "decision": "D74",
        "alpha": alpha,
        "n_objects": {c: len(v) for c, v in recs.items()},
        "n_dropped": {c: sum(1 for r in v if r.get("status") != "ok") for c, v in records.items()},
        "dropped_status": {c: sorted({r.get("status") for r in v if r.get("status") != "ok"})
                           for c, v in records.items()},
        "sets": plan["sets"],
        "primary": {n: tests[n] for n, *_, tier in H3_TESTS if tier == "primary"},
        "mechanism": mechanism,
        "secondary": {n: tests[n] for n, *_, tier in H3_TESTS if tier == "secondary"},
        "validation": {n: tests[n] for n, *_, tier in H3_TESTS if tier == "validation"},
        "descriptive": descriptive,
        "parameters": {"n_boot": n_boot, "seed": seed, "first_layer": first_layer},
    }
