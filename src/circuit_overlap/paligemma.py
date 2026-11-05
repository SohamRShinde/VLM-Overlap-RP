"""PaliGemma-3B: loader, inputs, full-forward runner and checks (D76).

Ported from Lietzow et al.'s ``vlm/paligemma_2.py`` (MIT, commit 145b453, read
2026-11-05): ``google/paligemma2-3b-mix-448``, 4-bit nf4 double quant, the
prompt passed as plain text with the image (PaliGemma is a prefix LM, no chat
template), and the same layer path as Qwen,
``model.model.language_model.layers[l].self_attn.o_proj``. So
``patching.capture_last_token_o_proj_inputs``, ``patching.patch_head_last_token``,
``patching.patch_example`` and ``cross_ablation.ablate_heads_last_token`` work
unchanged, given a runner with ``.logits()``.

**The runner runs the whole prompt every call** (``FullForwardRunner``), as
their code does for every model. Our Qwen speed-up (``LastTokenRunner``) runs
the prefix once and only the last token per patch; that is exact only under
causal attention, and PaliGemma's prefix may attend both ways (D76 (2)).
``last_token_reaches_earlier_positions`` measures it.

The GPU half has never run here (no torch locally). The CPU pieces
(``square_image_block``, the runner's interface) are unit-tested.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import Sequence

__all__ = [
    "CHECKPOINT",
    "load_paligemma",
    "build_inputs",
    "text_geometry",
    "FullForwardRunner",
    "square_image_block",
    "last_token_reaches_earlier_positions",
    "patch_rows_last_token",
    "patch_example_batched",
    "quiet_processor_warning",
]

CHECKPOINT = "google/paligemma2-3b-mix-448"   # D11, their MODEL_PATH_TEMPLATE


def load_paligemma(checkpoint: str = CHECKPOINT, attn_implementation: str | None = None):
    """Model and processor, as Lietzow's ``PaliGemma2Adapter.load_model``.

    One deliberate difference, as for Qwen (``t_screen.load_model``): the
    compute dtype is bf16 only on hardware with native bf16 (Ampere+), so a T4
    computes in fp16; they gate on ``torch.cuda.is_bf16_supported()``, which is
    True on a T4 through emulation (D76 (3)).
    """
    import torch
    from transformers import (
        BitsAndBytesConfig,
        PaliGemmaForConditionalGeneration,
        PaliGemmaProcessor,
    )

    from circuit_overlap.t_screen import native_bf16

    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    compute_dtype = torch.bfloat16 if native_bf16(capability) else torch.float16
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=None,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        low_cpu_mem_usage=True,
        **({"attn_implementation": attn_implementation} if attn_implementation else {}),
    ).eval()
    for param in model.parameters():
        param.requires_grad = False
    processor = PaliGemmaProcessor.from_pretrained(checkpoint, use_fast=True)
    return model, processor


def build_inputs(processor, image, question: str, device, keep_labels: bool = False):
    """Their ``build_inputs``: RGB image, plain-text question, no padding.

    Newer transformers' PaliGemmaProcessor also returns ``labels`` (Lietzow
    pinned 5.1.0). Passed to the model, they make it compute a training loss
    over every position and the full vocabulary -- wasted work, and it fails
    with last-position-only logits -- so they are dropped. The notebook checks
    that dropping them leaves the logits unchanged (it would not if they
    switched PaliGemma's attention mask to its training form).
    """
    inputs = processor(
        images=image.convert("RGB"), text=question, padding=False, return_tensors="pt"
    ).to(device)
    if not keep_labels:
        inputs.pop("labels", None)
    return inputs


def text_geometry(model) -> dict:
    """Layers, heads and head_dim of the language model, checked against o_proj.

    Gemma-2 sets ``head_dim`` explicitly and it is **not** hidden_size /
    n_heads, so the fallback Qwen's notebooks use would be wrong here; the
    o_proj input width must equal n_heads * head_dim or every head slice is
    misplaced.
    """
    cfg = model.model.language_model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = cfg.head_dim
    layers = model.model.language_model.layers
    in_features = layers[0].self_attn.o_proj.in_features
    if in_features != n_heads * head_dim:
        raise ValueError(f"o_proj takes {in_features} inputs, not {n_heads} x {head_dim}")
    if len(layers) != n_layers:
        raise ValueError(f"{len(layers)} layers, config says {n_layers}")
    return {"n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
            "n_kv_heads": getattr(cfg, "num_key_value_heads", None),
            "hidden_size": cfg.hidden_size,
            "final_logit_softcapping": getattr(cfg, "final_logit_softcapping", None),
            "attn_logit_softcapping": getattr(cfg, "attn_logit_softcapping", None)}


class FullForwardRunner:
    """``LastTokenRunner``'s interface, but every call runs the whole prompt.

    This is Lietzow's patching and knockout method: hooks on ``o_proj`` patch
    or zero the last position of the full sequence, and whatever that changes
    downstream, at any position, is included.

    ``last_only`` asks the model for the last position's logits only
    (``logits_to_keep=1``, D76a (1)). It skips the unembedding of every other
    position -- ~1,000 x a 257k vocabulary for PaliGemma -- and changes no
    number; the notebook checks it against the full output.
    """

    def __init__(self, model, inputs, last_only: bool = True):
        self.model = model
        self.inputs = inputs
        self.extra = {"logits_to_keep": 1} if last_only else {}

    def logits(self):
        """Next-token logits at the last position, as float32."""
        import torch

        with torch.no_grad():
            out = self.model(**self.inputs, use_cache=False, **self.extra)
        return out.logits[0, -1].float()


def square_image_block(input_ids: Sequence[int], image_token_id: int) -> tuple[int, int]:
    """``(offset, side)`` of the image tokens, which must be one square block.

    PaliGemma resizes every image to a fixed square (448 x 448 for mix-448), so
    its image tokens should be one contiguous run of side x side patches,
    row-major. Raises if they are not contiguous or not a square number, which
    would make the note-to-token mapping wrong.
    """
    ids = [int(t) for t in input_ids]
    where = [i for i, t in enumerate(ids) if t == image_token_id]
    if not where:
        raise ValueError("no image tokens in the prompt")
    offset, n = where[0], len(where)
    if where != list(range(offset, offset + n)):
        raise ValueError(f"image tokens are not contiguous: {n} tokens from {offset} to {where[-1]}")
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError(f"{n} image tokens is not a square grid")
    return offset, side


def last_token_reaches_earlier_positions(model, inputs, position: int = 0) -> dict:
    """Does changing only the last token change an earlier position's output?

    Under causal attention it cannot, which is what makes our cached Qwen path
    exact. Runs the prompt as given, again as given (the noise floor), and with
    the last input id replaced by its neighbour id; compares the final hidden
    state at ``position``. Reported, not acted on (D76 (2)).
    """
    import torch

    def final_state(batch):
        with torch.no_grad():
            out = model(**batch, use_cache=False, output_hidden_states=True)
        return out.hidden_states[-1][0, position].float()

    base = final_state(inputs)
    rerun = final_state(inputs)
    changed = dict(inputs)
    ids = inputs["input_ids"].clone()
    ids[0, -1] = ids[0, -1] + 1 if int(ids[0, -1]) + 1 < model.config.text_config.vocab_size else ids[0, -1] - 1
    changed["input_ids"] = ids
    other = final_state(changed)
    noise = float((rerun - base).abs().max())
    effect = float((other - base).abs().max())
    return {"position": position, "max_abs_change": effect, "rerun_noise": noise,
            "reaches_earlier_positions": effect > 10 * max(noise, 1e-6)}


def _repeat_rows(inputs, n: int) -> dict:
    """The batch-1 inputs repeated to n identical rows."""
    import torch

    return {k: v.repeat(n, *([1] * (v.dim() - 1)))
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == 1 else v
            for k, v in inputs.items()}


@contextmanager
def patch_rows_last_token(model, plan: Sequence, head_dim: int):
    """Row b of the batch gets ``plan[b] = (layer, head, value)`` at the last position.

    The batched form of ``patching.patch_head_last_token``: each row patches
    one head, so one forward pass of n identical rows does n single-head
    patches. Rows never interact (no padding, identical lengths).
    """
    from circuit_overlap.patching import o_proj_modules

    by_layer: dict[int, list] = defaultdict(list)
    for row, (layer, head, value) in enumerate(plan):
        by_layer[layer].append((row, head, value))

    def make_hook(entries):
        def hook(module, args):
            x = args[0].clone()
            for row, head, value in entries:
                x[row, -1, head * head_dim:(head + 1) * head_dim] = value.to(dtype=x.dtype, device=x.device)
            return (x, *args[1:])
        return hook

    modules = o_proj_modules(model)
    handles = []
    try:
        for layer, entries in by_layer.items():
            handles.append(modules[layer].register_forward_pre_hook(make_hook(entries)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def patch_example_batched(model, source_inputs, target_inputs, head_dim: int,
                          batch_size: int = 8, layers: Sequence[int] | None = None) -> dict | None:
    """``patching.patch_example``, with ``batch_size`` heads patched per forward pass.

    Same record, same keys, same skip rule (``None`` when the two runs' argmax
    tokens coincide); the clean runs and the captured activations are batch-1
    full forward passes, exactly as in ``patch_example``. Checked against it
    on the GPU before use (D76a (1)).
    """
    import torch

    from circuit_overlap.patching import capture_last_token_o_proj_inputs, o_proj_modules

    source, target = FullForwardRunner(model, source_inputs), FullForwardRunner(model, target_inputs)
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
    jobs = [(layer, head) for layer in layers for head in range(n_heads)]

    def sweep(inputs, acts):
        logit_prior = [[float("nan")] * n_heads for _ in range(n_layers)]
        logit_visual = [[float("nan")] * n_heads for _ in range(n_layers)]
        full = _repeat_rows(inputs, batch_size)
        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start:start + batch_size]
            batch = full if len(chunk) == batch_size else _repeat_rows(inputs, len(chunk))
            plan = [(l, h, acts[l][h * head_dim:(h + 1) * head_dim]) for l, h in chunk]
            with patch_rows_last_token(model, plan, head_dim), torch.no_grad():
                out = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            for row, (l, h) in enumerate(chunk):
                logit_prior[l][h] = float(out[row, a_prior])
                logit_visual[l][h] = float(out[row, a_visual])
        return logit_prior, logit_visual

    p2v_prior, p2v_visual = sweep(target_inputs, acts_source)
    v2p_prior, v2p_visual = sweep(source_inputs, acts_target)
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


_PROCESSOR_NAG = "You are passing both `text` and `images`"


class _DropProcessorNag(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _PROCESSOR_NAG not in record.getMessage()


def quiet_processor_warning() -> None:
    """Silence PaliGemmaProcessor's per-call warning about missing ``<image>`` tokens.

    We pass plain text, as Lietzow do; the processor then adds the image
    tokens itself and warns on every call. The notebook checks the resulting
    input directly, so the warning carries no information. Only that one
    message is dropped; every other warning still prints.
    """
    warnings.filterwarnings("ignore", message=_PROCESSOR_NAG + ".*")
    flt = _DropProcessorNag()
    for name in ("transformers", "transformers.models.paligemma.processing_paligemma"):
        logger = logging.getLogger(name)
        logger.addFilter(flt)
        for handler in logger.handlers:
            handler.addFilter(flt)
