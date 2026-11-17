"""PaliGemma H3 (D80): attention and direct effects on Gemma-2, full forward passes.

D74's measurements, ported to PaliGemma. What changes from ``h3.py``:

* **Full forward passes** (``paligemma.FullForwardRunner``), not the cached
  last-token path: PaliGemma's prefix attends both ways (D76 (2)). So the
  reconstruction check takes the values from the same pass (``v_proj`` at
  every position) instead of a KV cache.
* **Direct effect through Gemma-2's extra norm.** A Gemma-2 layer normalises
  the attention output (``post_attention_layernorm``) before adding it to the
  residual, and Gemma's RMSNorm multiplies by ``1 + weight``. A head's
  contribution is therefore its ``o_proj`` output times that norm's scale on
  the *whole* attention output (captured) times ``1 + weight``; then the final
  norm's scale on the actual residual times ``1 + weight``; then ``lm_head``.
  With both scales held at their actual values this is linear, so per-head
  contributions sum to the norm's output, which the notebook checks.
* **Image-token order.** PaliGemma has no 2-d position ids, so D74's M-RoPE
  row-major check is replaced by ``check_patch_order``: changing one pixel
  patch must change the vision tower's token at row-major index the most.

The attention capture, the final-residual capture and the record format are
``h3``'s, unchanged, so ``h3.analyse_h3`` reads these records as they are
(with ``first_layer=0``, all 26 layers stored, D80 (3)).

**Untested on the GPU** until the notebook's section 3 runs; torch is
imported lazily, so the module imports on a CPU box.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Mapping, Sequence

import numpy as np

from .h3 import capture_final_residual, capture_last_token_attention, round_grid

__all__ = [
    "FIRST_LAYER_PG",
    "PROBE_CELL",
    "capture_post_attention_norm",
    "gemma_scale",
    "head_direct_effects_gemma",
    "measure_run_pg",
    "check_attention_reconstructs_heads_full",
    "check_direct_effect_formula_gemma",
    "check_patch_order",
]

FIRST_LAYER_PG = 0          # D80 (3): every layer is stored
PROBE_CELL = (5, 21)        # D80 (4): row != column, so a transposed grid is caught


def _layers(model):
    return model.model.language_model.layers


@contextmanager
def capture_post_attention_norm(model, layers: Iterable[int]):
    """``{layer: (input, output)}`` of ``post_attention_layernorm`` at the last position."""
    out: dict[int, tuple] = {}
    handles = []

    def make_hook(layer):
        def hook(module, args, output):
            out[layer] = (args[0][0, -1].detach().float(), output[0, -1].detach().float())
        return hook

    try:
        for layer in layers:
            handles.append(_layers(model)[layer].post_attention_layernorm.register_forward_hook(make_hook(layer)))
        yield out
    finally:
        for handle in handles:
            handle.remove()


def gemma_scale(norm, x):
    """The RMS scale a Gemma RMSNorm applies to ``x`` (computed in fp32, as Gemma does)."""
    import torch

    eps = getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6))
    return torch.rsqrt(x.float().pow(2).mean() + eps)


def _head_blocks(z, head_dim: int):
    import torch

    n_heads = z.shape[0] // head_dim
    blocks = torch.zeros(n_heads, z.shape[0], dtype=z.dtype, device=z.device)
    for h in range(n_heads):
        blocks[h, h * head_dim:(h + 1) * head_dim] = z[h * head_dim:(h + 1) * head_dim]
    return blocks


def _head_residuals(model, z, post_in, layer: int, head_dim: int):
    """Each head's contribution to the residual stream, ``(n_heads, hidden)``, fp32."""
    blk = _layers(model)[layer]
    contrib = blk.self_attn.o_proj(_head_blocks(z, head_dim)).float()
    pn = blk.post_attention_layernorm
    return contrib * gemma_scale(pn, post_in) * (1.0 + pn.weight.float())


def _lens(model, resid, x_final):
    """``lm_head`` of residual pieces, scaled as the final norm scales the real residual."""
    norm = model.model.language_model.norm
    normed = resid * gemma_scale(norm, x_final) * (1.0 + norm.weight.float())
    return model.get_output_embeddings()(normed.to(x_final.dtype)).float()


def head_direct_effects_gemma(model, acts: Mapping[int, object], post: Mapping[int, tuple], x_final,
                              tok_a: int, tok_b: int, layers: Sequence[int], head_dim: int) -> np.ndarray:
    """Each head's direct effect on logit(tok_a) - logit(tok_b), ``(len(layers), n_heads)``."""
    import torch

    rows = []
    with torch.no_grad():
        for layer in layers:
            logits = _lens(model, _head_residuals(model, acts[layer], post[layer][0], layer, head_dim), x_final)
            rows.append((logits[:, tok_a] - logits[:, tok_b]).cpu().numpy())
    return np.stack(rows)


def measure_run_pg(model, runner, sets: Mapping[str, np.ndarray], tok_a: int, tok_b: int,
                   layers: Sequence[int], head_dim: int, with_note: bool = True) -> dict:
    """``h3.measure_run`` for PaliGemma: same record, Gemma-2 direct effects."""
    import torch

    from .patching import capture_last_token_o_proj_inputs

    with capture_last_token_attention(model, layers) as weights, \
            capture_last_token_o_proj_inputs(model) as acts, \
            capture_post_attention_norm(model, layers) as post, \
            capture_final_residual(model) as final:
        logits = runner.logits()
    names = ["image", "object"] + (["note"] if with_note else [])
    grids = {k: [] for k in names}
    for layer in layers:
        w = weights[layer]
        for k in names:
            idx = torch.as_tensor(np.asarray(sets[k]), device=w.device, dtype=torch.long)
            grids[k].append(w[:, idx].sum(dim=1).cpu().numpy())
    dla = head_direct_effects_gemma(model, acts, post, final["x"], tok_a, tok_b, layers, head_dim)
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


def check_attention_reconstructs_heads_full(model, runner, layers: Sequence[int], head_dim: int,
                                            rtol: float = 2e-2) -> float:
    """Weights x values (same pass, every position) must rebuild each head's o_proj input.

    Proves the captured weights are the ones the model used and that head h
    reads KV group ``h // (n_heads / n_kv_heads)``. Returns the worst relative
    error; raises above ``rtol`` (D80 (6)).
    """
    from .patching import capture_last_token_o_proj_inputs

    values: dict[int, object] = {}
    handles = []
    for layer in layers:
        def hook(module, args, output, layer=layer):
            values[layer] = output[0].detach().float()                      # (n, kv * hd)
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
        v = values[layer]
        n_kv = v.shape[-1] // head_dim
        v = v.view(v.shape[0], n_kv, head_dim)
        w = weights[layer]                                                  # (H, n)
        rep = w.shape[0] // n_kv
        for h in range(w.shape[0]):
            z = w[h] @ v[:, h // rep]
            ref = acts[layer][h * head_dim:(h + 1) * head_dim].float()
            worst = max(worst, float((z - ref).norm() / ref.norm().clamp_min(1e-6)))
    if worst > rtol:
        raise RuntimeError(f"attention x values does not reconstruct the heads: relative error "
                           f"{worst:.3g} > {rtol}. Do not run H3.")
    return worst


def check_direct_effect_formula_gemma(model, runner, tok_a: int, tok_b: int, layer: int,
                                      head_dim: int, atol: float = 0.1) -> dict:
    """The logit-lens pieces must reproduce the model (D80 (5)-(6)).

    (a) the lens on the final residual gives the model's own logit difference,
    with no final soft-capping; (b) one layer's per-head residual contributions
    sum to ``post_attention_layernorm``'s output. Raises if either fails.
    """
    import torch

    from .patching import capture_last_token_o_proj_inputs

    with capture_last_token_o_proj_inputs(model) as acts, \
            capture_post_attention_norm(model, [layer]) as post, \
            capture_final_residual(model) as final:
        logits = runner.logits()
    x = final["x"]
    with torch.no_grad():
        lens = _lens(model, x.float()[None], x)[0]
        parts = _head_residuals(model, acts[layer], post[layer][0], layer, head_dim).sum(dim=0)
    whole = post[layer][1]
    out = {
        "logit_diff_model": float(logits[tok_a] - logits[tok_b]),
        "logit_diff_lens": float(lens[tok_a] - lens[tok_b]),
        "head_sum_vs_post_norm": float((parts - whole).abs().max() / whole.abs().max().clamp_min(1e-6)),
    }
    cap = getattr(model.config.text_config, "final_logit_softcapping", None)
    if cap:
        capped = torch.tanh(lens / cap) * cap
        out["logit_diff_lens_if_capped"] = float(capped[tok_a] - capped[tok_b])
    out["lens_error"] = abs(out["logit_diff_model"] - out["logit_diff_lens"])
    if out["lens_error"] > atol or out["head_sum_vs_post_norm"] > 2e-2:
        raise RuntimeError(f"direct-effect pieces disagree with the model: {out}. Do not run H3.")
    return out


def check_patch_order(model, pixel_values, side: int, cell: tuple[int, int] = PROBE_CELL, dtype=None) -> dict:
    """Changing one pixel patch must move image token ``row * side + col`` the most (D80 (4)).

    Runs the vision tower twice; the patch is replaced by its sign-flipped
    opposite (+1/-1 per pixel), so it changes whatever it held. Raises if the
    most-changed token is anything else, e.g. the transposed index.
    """
    import torch

    r, c = cell
    tower = model.model.vision_tower
    # The patch embedding (a conv, never quantised) sets the dtype the tower takes.
    emb = getattr(getattr(tower, "vision_model", tower), "embeddings", None)
    weight = getattr(getattr(emb, "patch_embedding", None), "weight", None)
    dtype = weight.dtype if weight is not None else dtype
    pv = pixel_values.to(dtype) if dtype is not None else pixel_values
    p = pv.shape[-1] // side
    changed = pv.clone()
    block = changed[..., r * p:(r + 1) * p, c * p:(c + 1) * p]
    changed[..., r * p:(r + 1) * p, c * p:(c + 1) * p] = torch.where(block > 0, -1.0, 1.0).to(block.dtype)
    with torch.no_grad():
        a = tower(pv).last_hidden_state[0].float()
        b = tower(changed).last_hidden_state[0].float()
    delta = (b - a).norm(dim=-1)
    if delta.shape[0] != side * side:
        raise RuntimeError(f"vision tower gives {delta.shape[0]} tokens, grid {side} x {side}")
    order = torch.argsort(delta, descending=True)
    out = {"cell": [r, c], "patch_px": p, "expected": r * side + c, "most_changed": int(order[0]),
           "transposed_rank": int((order == c * side + r).nonzero()[0, 0]),
           "ratio_to_second": float(delta[order[0]] / delta[order[1]].clamp_min(1e-9))}
    if out["most_changed"] != out["expected"]:
        raise RuntimeError(f"image tokens are not row-major: {out}. Do not run H3.")
    return out
