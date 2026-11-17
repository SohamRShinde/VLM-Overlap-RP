"""Exact layer replay for last-position head ablations (D78 (3)).

Zero (or mean) ablation of a head's ``o_proj`` input at the **last position**
of layer ``l`` cannot change anything that enters layers ``0..l``, at any
position -- even in PaliGemma, whose prefix attends both ways. So one
unablated full forward pass can record what enters each decoder layer, and an
ablated configuration only needs the decoder layers from its own earliest
ablated layer onwards, then the final norm and unembedding at the last
position. Several configurations run as rows of one batch, each row with its
own heads ablated.

What could go wrong silently, and what guards it:

* **a decoder layer needs inputs we did not record** (attention mask, rotary
  embeddings, cache positions): every layer's call is recorded *as made* by
  the model (args and kwargs, via a pre-hook) and replayed with the same
  arguments, so nothing is rebuilt by hand;
* **the head differs from the model's** (final logit soft-capping): chosen
  empirically per instance by reproducing the full pass's logits from the last
  layer, and refused if neither form matches (``ReplayMismatch``);
* **rows leak into each other** (one row's ablation reaching another): the
  batched hook writes only its own row; the notebook checks an unablated row
  inside an ablated batch, batch vs single, and a permuted batch;
* **the replay drifts from the full pass on some object**: the notebook
  replays the unablated run from the earliest start layer for every object
  and flags any mismatch (``replay_mismatch``).

The notebook checks all of this against full forward passes with
``cross_ablation.ablate_heads_last_token`` (independent code) before any loop.
torch is imported lazily: ``replay_groups`` is CPU-testable.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Iterable, Mapping, Sequence

__all__ = [
    "ReplayMismatch",
    "LayerReplay",
    "ablate_rows_last_token",
    "start_layer",
    "replay_groups",
    "patch_jobs",
    "record_with_activations",
    "patch_example_replay",
]

SOFTCAP_TOL = 0.1      # |dlogit| allowed when matching the full pass's logits


class ReplayMismatch(RuntimeError):
    """The replay does not reproduce the model's own logits."""


def start_layer(heads: Iterable) -> int | None:
    """Earliest ablated layer of a configuration (None for no heads)."""
    layers = [int(l) for l, _ in heads]
    return min(layers) if layers else None


def replay_groups(names: Sequence[str], configs: Mapping[str, Sequence], batch_size: int,
                  floor: int) -> list[tuple[int, list[str]]]:
    """Batches of configuration names, each replayed from one start layer.

    Configurations are grouped by their earliest ablated layer (an empty one
    starts at ``floor``, the earliest layer recorded) and cut into chunks of
    ``batch_size``. Order inside a group follows ``names``.
    """
    by_start: dict[int, list[str]] = defaultdict(list)
    for name in names:
        s = start_layer(configs[name])
        by_start[floor if s is None else s].append(name)
    out = []
    for s in sorted(by_start):
        group = by_start[s]
        for k in range(0, len(group), batch_size):
            out.append((s, group[k:k + batch_size]))
    return out


def patch_jobs(layers: Sequence[int], n_heads: int, batch_size: int) -> list[tuple[int, list[int]]]:
    """Single-head patching jobs as (layer, heads) batches, one layer per batch (D79 (3)).

    Each batch replays from its own layer, so a batch never mixes layers.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    heads = list(range(n_heads))
    return [(int(l), heads[k:k + batch_size]) for l in layers for k in range(0, n_heads, batch_size)]


def _decoder_layers(model):
    return model.model.language_model.layers


@contextmanager
def ablate_rows_last_token(model, row_heads: Sequence[Sequence], head_dim: int,
                           values: Mapping | None = None):
    """Row b zeroes (or sets to ``values[(l, h)]``) the heads in ``row_heads[b]``, last position only.

    The batched form of ``cross_ablation.ablate_heads_last_token``. A hook
    writes only the rows that ablate a head in its layer.
    """
    by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, heads in enumerate(row_heads):
        for layer, head in heads:
            by_layer[int(layer)].append((row, int(head)))

    def make_hook(layer, entries):
        def hook(module, args):
            x = args[0].clone()
            for row, head in entries:
                sl = slice(head * head_dim, (head + 1) * head_dim)
                if values is None:
                    x[row, -1, sl] = 0
                else:
                    x[row, -1, sl] = values[(layer, head)].to(dtype=x.dtype, device=x.device)
            return (x, *args[1:])
        return hook

    layers = _decoder_layers(model)
    handles = []
    try:
        for layer, entries in by_layer.items():
            handles.append(layers[layer].self_attn.o_proj.register_forward_pre_hook(
                make_hook(layer, entries)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _batched(obj, n: int):
    """Expand every batch-1 tensor (2+ dims, leading dim 1) in obj to n rows."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.expand(n, *obj.shape[1:]) if obj.dim() >= 2 and obj.shape[0] == 1 else obj
    if isinstance(obj, tuple):
        return tuple(_batched(o, n) for o in obj)
    if isinstance(obj, list):
        return [_batched(o, n) for o in obj]
    if isinstance(obj, dict):
        return {k: _batched(v, n) for k, v in obj.items()}
    return obj


class LayerReplay:
    """One input's unablated pass, recorded so ablated passes can resume mid-model.

    ``LayerReplay(model, inputs, first)`` runs the full model once (no
    ablation), records every decoder layer's call from ``first`` onwards, and
    keeps the last-position logits as ``self.logits``. ``replay(row_heads,
    start)`` returns the last-position logits (rows x vocab, float32) of each
    row with its heads ablated, recomputing layers ``start..`` only.

    ``softcap="auto"`` decides the head's final soft-capping by reproducing
    ``self.logits`` from the last layer (``self.head_check``). Pass the value
    found once (a float or None) to skip that extra layer, e.g. inside a
    capture hook, where re-running a layer would fire the hook again.
    """

    def __init__(self, model, inputs, first: int, softcap="auto"):
        import torch

        self.model, self.first = model, first
        layers = _decoder_layers(model)
        self.n_layers = len(layers)
        self.calls: dict[int, tuple] = {}

        def make_pre(l):
            def pre(module, args, kwargs):
                kw = dict(kwargs)
                if args:
                    hidden, rest = args[0], tuple(args[1:])
                else:
                    hidden, rest = kw.pop("hidden_states"), ()
                self.calls[l] = (hidden.detach().clone(), rest, kw)
            return pre

        handles = [layers[l].register_forward_pre_hook(make_pre(l), with_kwargs=True)
                   for l in range(first, self.n_layers)]
        try:
            with torch.no_grad():
                out = model(**inputs, use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        if sorted(self.calls) != list(range(first, self.n_layers)):
            raise ReplayMismatch(f"recorded layers {sorted(self.calls)}, expected {first}..{self.n_layers - 1}")
        self.logits = out.logits[0, -1].float()
        if softcap == "auto":
            self.softcap = None
            self.head_check = self._choose_head()
        else:
            self.softcap, self.head_check = softcap, None

    def _final(self, hidden):
        import torch

        lm = self.model.model.language_model
        x = lm.norm(hidden[:, -1:, :])
        logits = self.model.lm_head(x)[:, -1].float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    def _run(self, start: int, n_rows: int):
        hidden = self.calls[start][0].repeat(n_rows, *([1] * (self.calls[start][0].dim() - 1)))
        layers = _decoder_layers(self.model)
        for l in range(start, self.n_layers):
            _, rest, kw = self.calls[l]
            out = layers[l](hidden, *_batched(rest, n_rows), **_batched(kw, n_rows))
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden

    def _choose_head(self) -> float:
        """Reproduce the full pass from the last layer: plain head, or soft-capped."""
        import torch

        cap = getattr(getattr(self.model.config, "text_config", self.model.config),
                      "final_logit_softcapping", None)
        with torch.no_grad():
            hidden = self._run(self.n_layers - 1, 1)
            best = None
            for c in (None, cap) if cap else (None,):
                self.softcap = c
                d = float((self._final(hidden)[0] - self.logits).abs().max())
                if best is None or d < best[0]:
                    best = (d, c)
        self.softcap = best[1]
        if best[0] > SOFTCAP_TOL:
            raise ReplayMismatch(f"head does not reproduce the model's logits (max |dlogit| {best[0]:.3f})")
        return best[0]

    def replay(self, row_heads: Sequence[Sequence], start: int, head_dim: int,
               values: Mapping | None = None):
        """Last-position logits of each row, ablating ``row_heads[b]``, from layer ``start``."""
        import torch

        if start < self.first:
            raise ValueError(f"start {start} is before the first recorded layer {self.first}")
        for heads in row_heads:
            s = start_layer(heads)
            if s is not None and s < start:
                raise ValueError(f"a row ablates layer {s}, before the replay start {start}")
        with torch.no_grad(), ablate_rows_last_token(self.model, row_heads, head_dim, values):
            hidden = self._run(start, len(row_heads))
            return self._final(hidden)


def record_with_activations(model, inputs, softcap="auto"):
    """A ``LayerReplay`` from layer 0 plus the run's last-position ``o_proj`` inputs.

    One full pass does both. The head's soft-capping is decided after the
    capture hook is gone, so re-running the last layer cannot overwrite the
    captured activations.
    """
    from circuit_overlap.patching import capture_last_token_o_proj_inputs

    with capture_last_token_o_proj_inputs(model) as acts:
        rep = LayerReplay(model, inputs, 0, softcap=None)
    if softcap == "auto":
        rep.head_check = rep._choose_head()
    else:
        rep.softcap = softcap
    return rep, dict(acts)


def patch_example_replay(model, source_inputs, target_inputs, head_dim: int, batch_size: int = 8,
                         layers: Sequence[int] | None = None, softcap="auto") -> dict | None:
    """``paligemma.patch_example_batched`` by layer replay (D79 (3)): same record, same skip rule.

    Each run is recorded once from layer 0; the head patched at layer l then
    recomputes layers l.. only. The record adds ``replay_check``: the larger of
    the two runs' |unablated replay from layer 0 - full pass| (its
    ``LayerReplay.logits``), which the notebook compares with a tolerance.
    """
    import torch

    rep_source, acts_source = record_with_activations(model, source_inputs, softcap)
    rep_target, acts_target = record_with_activations(model, target_inputs, rep_source.softcap)
    out_source, out_target = rep_source.logits, rep_target.logits
    a_prior, a_visual = int(out_source.argmax()), int(out_target.argmax())
    if a_prior == a_visual:
        return None

    n_layers = rep_source.n_layers
    layers = list(range(n_layers)) if layers is None else list(layers)
    n_heads = acts_source[0].shape[0] // head_dim
    jobs = patch_jobs(layers, n_heads, batch_size)

    with torch.no_grad():
        replay_check = max(float((rep.replay([[]], 0, head_dim)[0] - rep.logits).abs().max())
                           for rep in (rep_source, rep_target))

    def sweep(rep, acts):
        logit_prior = [[float("nan")] * n_heads for _ in range(n_layers)]
        logit_visual = [[float("nan")] * n_heads for _ in range(n_layers)]
        for layer, heads in jobs:
            values = {(layer, h): acts[layer][h * head_dim:(h + 1) * head_dim] for h in heads}
            out = rep.replay([[(layer, h)] for h in heads], layer, head_dim, values)
            for row, h in enumerate(heads):
                logit_prior[layer][h] = float(out[row, a_prior])
                logit_visual[layer][h] = float(out[row, a_visual])
        return logit_prior, logit_visual

    p2v_prior, p2v_visual = sweep(rep_target, acts_source)    # source's heads into the target run
    v2p_prior, v2p_visual = sweep(rep_source, acts_target)
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
        "replay_check": replay_check,
    }
