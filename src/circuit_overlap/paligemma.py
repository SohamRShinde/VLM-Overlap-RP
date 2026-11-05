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

import math
from typing import Sequence

__all__ = [
    "CHECKPOINT",
    "load_paligemma",
    "build_inputs",
    "text_geometry",
    "FullForwardRunner",
    "square_image_block",
    "last_token_reaches_earlier_positions",
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


def build_inputs(processor, image, question: str, device):
    """Their ``build_inputs``: RGB image, plain-text question, no padding."""
    return processor(
        images=image.convert("RGB"), text=question, padding=False, return_tensors="pt"
    ).to(device)


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
    """

    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs

    def logits(self):
        """Next-token logits at the last position, as float32."""
        import torch

        with torch.no_grad():
            out = self.model(**self.inputs, use_cache=False)
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
