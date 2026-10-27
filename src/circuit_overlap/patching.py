"""Last-token activation patching of attention heads, and head classification.

Ports the method of Lietzow et al.'s repo (MIT, Copyright 2026 Niclas Lietzow,
commit 145b453, read 2026-10-27), so our P circuit is found the way theirs was:

* ``runner/patching_last_token_attn_heads.py``: run the source (Prior prompt)
  and target (Visual prompt) on the counterfactual image, cache every layer's
  ``o_proj`` input at the last token, then for every head patch the source
  slice into the target run (P2V) and the target slice into the source run
  (V2P), recording the logits of the two runs' argmax tokens. Examples whose
  argmax tokens coincide are skipped; source and target inputs must have the
  same shape.
* ``scripts/analysis/_patching_common.py``: the per-head restoration score.
* ``scripts/analysis/classify_attn_heads.py``: PCA on the per-head mean
  (P2V, V2P) scores, PC1 oriented so its components sum positive, heads beyond
  +/- 2 sigma of the PC1 scores are promoting / suppressing.

One deliberate difference: they run a full forward pass per patched head. A
last-token patch cannot change any earlier position (attention is causal), so
here the prompt minus its last token is run once and cached, and each patch
re-runs only the last token on top of that cache (plan 3.4's speed-up).
``LastTokenRunner.check_against_full_forward`` verifies the equivalence on
the real model before a sweep.

torch is imported lazily, so the numpy half of this module (scores and
classification) stays importable and testable on a CPU box without it.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "o_proj_modules",
    "capture_last_token_o_proj_inputs",
    "patch_head_last_token",
    "LastTokenRunner",
    "patch_example",
    "head_restoration_scores",
    "mean_restoration_grids",
    "classify_heads",
    "rank_heads",
]


# --- GPU side --------------------------------------------------------------


def o_proj_modules(model) -> list:
    """The ``o_proj`` of every language-model layer (Qwen2.5-VL layout)."""
    return [layer.self_attn.o_proj for layer in model.model.language_model.layers]


@contextmanager
def capture_last_token_o_proj_inputs(model):
    """Collect each layer's ``o_proj`` input at the last position.

    Yields a dict ``{layer: tensor(hidden)}`` filled by the next forward pass.
    This is the per-head attention output z, before W_O mixes the heads.
    """
    acts: dict[int, object] = {}
    handles = []

    def make_hook(layer):
        def hook(module, args):
            acts[layer] = args[0][0, -1].detach().clone()
        return hook

    try:
        for layer, module in enumerate(o_proj_modules(model)):
            handles.append(module.register_forward_pre_hook(make_hook(layer)))
        yield acts
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_head_last_token(model, layer: int, head: int, head_dim: int, value):
    """Overwrite one head's ``o_proj`` input slice at the last position."""
    start, end = head * head_dim, (head + 1) * head_dim
    module = o_proj_modules(model)[layer]

    def hook(module, args):
        x = args[0].clone()
        x[:, -1, start:end] = value.to(dtype=x.dtype, device=x.device)
        return (x, *args[1:])

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class LastTokenRunner:
    """Run a prompt's final token repeatedly over a cached prefix.

    The prefix (every token but the last, image included) is run once. Each
    call to :meth:`logits` copies that cache and runs the last token alone, so
    a patch at the last position costs one token instead of ~1,400.

    Qwen2.5-VL keeps its multimodal RoPE offset on the model
    (``model.model.rope_deltas``) between calls; it is recorded after the
    prefill and restored before every step, so interleaving two runners
    (source and target) stays correct.
    """

    def __init__(self, model, inputs):
        import torch

        self.model = model
        self.inputs = inputs
        ids = inputs["input_ids"]
        n = ids.shape[1]
        prefix = {}
        for key, value in inputs.items():
            sliceable = (
                isinstance(value, torch.Tensor) and value.dim() >= 2
                and value.shape[0] == ids.shape[0] and value.shape[1] == n
            )
            prefix[key] = value[:, :-1] if sliceable else value
        with torch.no_grad():
            out = model(**prefix, use_cache=True)
        self.cache = out.past_key_values
        self.rope_deltas = getattr(model.model, "rope_deltas", None)
        self.cache_position = torch.tensor([n - 1], device=ids.device)

    def logits(self):
        """Last-position logits (float32) under whatever hooks are active."""
        import torch

        if self.rope_deltas is not None:
            self.model.model.rope_deltas = self.rope_deltas
        with torch.no_grad():
            out = self.model(
                input_ids=self.inputs["input_ids"][:, -1:],
                attention_mask=self.inputs["attention_mask"],
                past_key_values=copy.deepcopy(self.cache),
                cache_position=self.cache_position,
                use_cache=True,
            )
        return out.logits[0, -1].float()

    def check_against_full_forward(self, atol: float) -> float:
        """Max |logit difference| between this path and one full forward pass.

        Raises if it exceeds ``atol`` or the argmax differs: a sweep on a
        broken cache path would be silently wrong for every head.
        """
        import torch

        with torch.no_grad():
            full = self.model(**self.inputs, use_cache=False).logits[0, -1].float()
        fast = self.logits()
        diff = float((full - fast).abs().max())
        if int(full.argmax()) != int(fast.argmax()) or diff > atol:
            raise RuntimeError(
                f"cached last-token path disagrees with a full forward pass: "
                f"max |dlogit| = {diff:.4g}, argmax {int(full.argmax())} vs "
                f"{int(fast.argmax())}. Do not run the sweep."
            )
        return diff


def patch_example(model, source: LastTokenRunner, target: LastTokenRunner,
                  head_dim: int, layers: Sequence[int] | None = None) -> dict | None:
    """Both patching directions for one example, in Lietzow's record format.

    ``source`` is the Prior-prompt run, ``target`` the Visual-prompt run, both
    on the counterfactual image. Returns ``None`` when the two runs' argmax
    tokens coincide (they skip those too). Logit grids are
    ``[layer][head]`` lists of floats.
    """
    with capture_last_token_o_proj_inputs(model) as acts_source:
        out_source = source.logits()
    with capture_last_token_o_proj_inputs(model) as acts_target:
        out_target = target.logits()
    a_prior, a_visual = int(out_source.argmax()), int(out_target.argmax())
    if a_prior == a_visual:
        return None

    n_layers = len(o_proj_modules(model))
    layers = list(range(n_layers)) if layers is None else list(layers)
    n_heads = acts_source[0].shape[0] // head_dim

    def sweep(run: LastTokenRunner, acts: Mapping[int, object]):
        logit_prior = [[float("nan")] * n_heads for _ in range(n_layers)]
        logit_visual = [[float("nan")] * n_heads for _ in range(n_layers)]
        for layer in layers:
            for head in range(n_heads):
                value = acts[layer][head * head_dim:(head + 1) * head_dim]
                with patch_head_last_token(model, layer, head, head_dim, value):
                    out = run.logits()
                logit_prior[layer][head] = float(out[a_prior])
                logit_visual[layer][head] = float(out[a_visual])
        return logit_prior, logit_visual

    p2v_prior, p2v_visual = sweep(target, acts_source)
    v2p_prior, v2p_visual = sweep(source, acts_target)
    return {
        "next_token_idx_prior": a_prior,
        "next_token_idx_visual": a_visual,
        "prior_token_logit_prior_run": float(out_source[a_prior]),
        "visual_token_logit_prior_run": float(out_source[a_visual]),
        "prior_token_logit_visual_run": float(out_target[a_prior]),
        "visual_token_logit_visual_run": float(out_target[a_visual]),
        "p2v_logit_prior": p2v_prior,
        "p2v_logit_visual": p2v_visual,
        "v2p_logit_prior": v2p_prior,
        "v2p_logit_visual": v2p_visual,
    }


# --- CPU side: scores and classification ---------------------------------


def head_restoration_scores(record: Mapping) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-head (P2V, V2P) restoration scores for one example.

    Lietzow's ``_compute_head_restoration_scores_prior_circuit``. With
    Delta = logit(prior token) - logit(visual token):

        P2V = (Delta_patched - Delta_visual_run) / (Delta_prior_run - Delta_visual_run)
        V2P = (Delta_patched - Delta_prior_run) / (Delta_visual_run - Delta_prior_run)

    so 1 means the patch fully moved the run to the other run's behaviour.
    Returns ``None`` when the contrast is degenerate (|denominator| < 1e-8),
    which drops the example, as they do.
    """
    prior_baseline = record["prior_token_logit_prior_run"] - record["visual_token_logit_prior_run"]
    visual_baseline = record["prior_token_logit_visual_run"] - record["visual_token_logit_visual_run"]
    denom = prior_baseline - visual_baseline
    if abs(denom) < 1e-8:
        return None
    p2v_diff = np.asarray(record["p2v_logit_prior"], float) - np.asarray(record["p2v_logit_visual"], float)
    v2p_diff = np.asarray(record["v2p_logit_prior"], float) - np.asarray(record["v2p_logit_visual"], float)
    p2v = (p2v_diff - visual_baseline) / denom
    v2p = (v2p_diff - prior_baseline) / -denom
    return p2v, v2p


def mean_restoration_grids(records: Iterable[Mapping]) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean (P2V, V2P) grids over examples, and how many examples counted."""
    p2v_sum = v2p_sum = None
    n = 0
    for record in records:
        scores = head_restoration_scores(record)
        if scores is None:
            continue
        p2v, v2p = scores
        p2v_sum = p2v if p2v_sum is None else p2v_sum + p2v
        v2p_sum = v2p if v2p_sum is None else v2p_sum + v2p
        n += 1
    if n == 0:
        raise ValueError("no example has a non-degenerate contrast")
    return p2v_sum / n, v2p_sum / n, n


def classify_heads(p2v_grid, v2p_grid, sigma: float = 2.0) -> dict:
    """Promoting / suppressing heads by PCA on per-head (P2V, V2P) means.

    Port of Lietzow's ``classify_attn_heads``: points are centred, PC1 is the
    top eigenvector of their covariance, flipped so its components sum
    positive, and a head is promoting above mean + sigma * std of the PC1
    scores (population std) and suppressing below mean - sigma * std.
    """
    p2v = np.asarray(p2v_grid, float)
    v2p = np.asarray(v2p_grid, float)
    if p2v.shape != v2p.shape or p2v.ndim != 2:
        raise ValueError(f"grids must be the same 2-D shape, got {p2v.shape} and {v2p.shape}")
    if not (np.isfinite(p2v).all() and np.isfinite(v2p).all()):
        raise ValueError("restoration grids contain non-finite values")
    n_layers, n_heads = p2v.shape
    points = np.column_stack([p2v.ravel(), v2p.ravel()])
    centred = points - points.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov(centred.T))
    pc1 = eigenvectors[:, -1]
    if pc1.sum() < 0:
        pc1 = -pc1
    scores = centred @ pc1
    mean, std = float(scores.mean()), float(scores.std())

    def heads_where(mask) -> list[tuple[int, int]]:
        return [(int(i // n_heads), int(i % n_heads)) for i in np.flatnonzero(mask)]

    return {
        "promoting": heads_where(scores > mean + sigma * std),
        "suppressing": heads_where(scores < mean - sigma * std),
        "pc1_scores": scores.reshape(n_layers, n_heads),
        "pc1_mean": mean,
        "pc1_std": std,
        "pc1_direction": (float(pc1[0]), float(pc1[1])),
        "sigma": sigma,
    }


def rank_heads(pc1_scores) -> dict[tuple[int, int], int]:
    """1-based rank of every head by PC1 score, highest first."""
    scores = np.asarray(pc1_scores, float)
    n_heads = scores.shape[1]
    order = np.argsort(-scores.ravel(), kind="stable")
    return {(int(i // n_heads), int(i % n_heads)): rank for rank, i in enumerate(order, 1)}
