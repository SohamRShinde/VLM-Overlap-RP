"""GPU driver for the T-condition behavioural screen (plan section 4, step 3).

**This module has never been run.** It needs a GPU, Qwen2.5-VL-3B weights and
the Visual-Counterfact images, none of which exist on the dev machine. Every
CPU-testable part of the screen lives elsewhere and *is* tested:
``notes`` (overlay + mask), ``tokens`` (mask -> image tokens), ``screen``
(scoring), ``data`` (manifest), ``provenance`` (result envelope). What is left
here is model plumbing, to be smoke-run on Kaggle over ~5 objects before the
full sweep.

Known unverified points, all cheap to check in that smoke run:
  * the exact HF checkpoint id (docs/STATE.md blocker 2)
  * the exact Visual prompt string -- "here?" differs between Lietzow section 3.1
    and their Table 1 caption (D22)
  * that each colour word has a distinct single first token (D26)
  * fp16 numerics on Qwen (plan risk 7)

torch and transformers are imported lazily inside the functions, so the module
stays importable on a CPU box with neither installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .data import ObjectEntry, collect_color_vocabulary
from .notes import NoteSpec, matched_neutral_word, render_note, variant_texts
from .screen import ObjectRecord, parse_color_word, summarise_screen

__all__ = [
    "VISUAL_PROMPT",
    "PRIOR_PROMPT",
    "ANSWER_SUFFIX",
    "ScreenConfig",
    "load_notes_config",
    "color_first_token_ids",
    "best_accepted_color",
    "load_model",
    "answer_and_color_logits",
    "run_screen",
]

# Verified against Lietzow's repo (src/vdpo/runner/core.py::_get_question;
# MIT, commit 145b453, read 2026-10-19). The answer-format suffix is part of
# the prompt in both modes -- our first version omitted it.
ANSWER_SUFFIX = "Respond with the color name only."
VISUAL_PROMPT = "What color is this {object} here? " + ANSWER_SUFFIX
# Prior mode takes the indefinite article from inflect: "a cow" / "an apple".
PRIOR_PROMPT = "What color is {a_object} usually? " + ANSWER_SUFFIX


@dataclass(frozen=True)
class ScreenConfig:
    checkpoint: str = "Qwen/Qwen2.5-VL-3B-Instruct"  # verified in their vlm/qwen.py
    # min_pixels == max_pixels pins the image *area*, as Lietzow do
    # (vlm/qwen.py: 1280 * 28 * 28). The side lengths still follow the aspect
    # ratio, both multiples of 28, so the token grid varies per image -- use
    # tokens.token_grid_shape on the resized image rather than assuming a square.
    # Matching their budget also keeps the note legible: ~1280 image tokens
    # rather than the 256 of a 448x448 grid (D62).
    pixels: int = 1280 * 28 * 28
    max_new_tokens: int = 8
    # 4-bit, matching their default (settings.load_in_4bit = True) so the
    # replication gate compares like with like (D10, decided 2026-10-19).
    load_in_4bit: bool = True
    dtype: str = "float16"
    seed: int = 0
    prompt_template: str = VISUAL_PROMPT
    min_usable: int = 50


def load_notes_config(path: str | Path = "configs/notes.json") -> dict:
    """Load configs/notes.json and turn its layouts into ``NoteSpec`` objects."""
    cfg = json.loads(Path(path).read_text())
    specs = [
        NoteSpec(
            name=l["name"],
            rel_width=l["rel_width"],
            anchor=tuple(l["anchor"]),
            rotation_deg=l.get("rotation_deg", 0.0),
        )
        for l in cfg["layouts"]
    ]
    return {
        "color_vocabulary": cfg["color_vocabulary"],
        "neutral_words": cfg["neutral_words"],
        "specs": specs,
    }


def color_first_token_ids(
    tokenizer, colors: Iterable[str]
) -> tuple[dict[str, list[int]], list[str]]:
    """First-token ids per colour, following Lietzow's convention.

    They tokenise **both** case variants of a colour ("Blue" and "blue") with
    ``add_special_tokens=False`` and **no leading space** -- the answer opens
    the assistant turn, so nothing precedes it -- and keep the first token id of
    each (``core.py::get_answer_token_ids``). A colour's logit is the max over
    its variants.

    Returns ``(ids, collisions)``. A collision is a token id claimed by two
    different colours, which makes their logit difference meaningless; drop
    those colours, or the objects using them, before running (D26).
    """
    ids: dict[str, list[int]] = {}
    for color in colors:
        base = color.strip()
        token_ids: set[int] = set()
        for variant in {base.capitalize(), base.lower()}:
            pieces = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if pieces:
                token_ids.add(int(pieces[0]))
        if not token_ids:
            raise ValueError(f"colour {color!r} produced no tokens")
        ids[base] = sorted(token_ids)

    owners: dict[int, list[str]] = {}
    for color, token_ids in ids.items():
        for t in token_ids:
            owners.setdefault(t, []).append(color)
    collisions = sorted({c for cs in owners.values() if len(cs) > 1 for c in cs})
    return ids, collisions


def load_model(config: ScreenConfig):
    """Load Qwen2.5-VL and its processor, mirroring Lietzow's loader.

    Their 4-bit setup (``vlm/qwen.py``): nf4, double quantisation, and a compute
    dtype of bfloat16 **where the GPU supports it, float16 otherwise**. A T4
    does not support bf16, so on Kaggle the compute dtype will be fp16 while
    their runs were most likely bf16 -- a residual difference to note in
    RESULTS.md rather than a bug.
    """
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else getattr(torch, config.dtype)
    )
    quantization_config = None
    dtype = compute_dtype
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        dtype = None

    processor = AutoProcessor.from_pretrained(
        config.checkpoint,
        use_fast=True,
        min_pixels=config.pixels,
        max_pixels=config.pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.checkpoint,
        dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    ).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, processor


def answer_and_color_logits(
    model, processor, image, prompt: str, color_token_ids: dict[str, list[int]],
    max_new_tokens: int = 8,
) -> tuple[str, dict[str, float]]:
    """Greedy answer plus the first-token logit of every colour word.

    Returns the *whole* colour-logit dictionary rather than a single Delta,
    because Visual-Counterfact's ``correct_answer`` can list several acceptable
    colours (e.g. ``['brown', 'white']``) and the reference colour for
    Delta = logit(c_o) - logit(c_cf) is only known after the model has answered.
    Keeping all of them costs a dozen floats per generation and lets any pair be
    recomputed later without a second GPU run.

    ``scores[0]`` holds the logits that produced the first generated token --
    the position the whole study intervenes on (plan 3.4).
    """
    import torch

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    first_logits = out.scores[0][0].float()
    color_logits = {
        c: float(max(first_logits[i] for i in ids)) for c, ids in color_token_ids.items()
    }
    generated = out.sequences[0, inputs["input_ids"].shape[1] :]
    answer = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer, color_logits


def best_accepted_color(
    color_logits: dict[str, float], original_colors: Sequence[str]
) -> str:
    """The accepted colour with the highest first-token logit in this run.

    Lietzow score a run as ``max over correct answers`` vs ``max over the
    incorrect answer`` (``runner/inference.py``), precisely because
    ``correct_answer`` lists several acceptable colours. Delta = logit(best
    accepted) - logit(c_cf) is that difference, so the reference colour is
    whichever accepted colour the model currently likes best, chosen per run.

    Records also store every colour's logit, so the alternative convention --
    fix the reference on the neutral run and reuse it for the attacked run --
    is recomputable offline without another GPU pass.
    """
    present = [c for c in original_colors if c in color_logits]
    if not present:
        raise KeyError(f"none of {tuple(original_colors)} is in the colour logits")
    return max(present, key=lambda c: color_logits[c])


def run_screen(
    entries: Sequence[ObjectEntry],
    model,
    processor,
    config: ScreenConfig | None = None,
    notes_config: dict | None = None,
    progress: bool = True,
) -> tuple[list[ObjectRecord], dict]:
    """Run the screen over a manifest and return records plus a summary.

    Two generations per object per layout: neutral note and incongruent note.
    The congruent and no-note baselines (D24) are secondary and are skipped
    here to keep the kill-test cheap -- add them once the gate passes.
    """
    config = config or ScreenConfig()
    notes_config = notes_config or load_notes_config()

    # The parser's dictionary must cover every colour the data actually uses,
    # not just the placeholder list in configs/notes.json.
    vocab = sorted(set(notes_config["color_vocabulary"]) | set(collect_color_vocabulary(entries)))
    specs = notes_config["specs"]
    token_ids, collisions = color_first_token_ids(processor.tokenizer, vocab)
    if collisions:
        raise ValueError(
            f"colour words sharing a first token (D26): {collisions}. "
            "Drop the affected objects, or the words from configs/notes.json, "
            "before running -- the logit difference is undefined for them."
        )

    records: list[ObjectRecord] = []
    for n, entry in enumerate(entries, 1):
        if progress and n % 25 == 0:
            print(f"  {n}/{len(entries)} objects", flush=True)
        image = entry.open_image()
        prompt = config.prompt_template.format(object=entry.object_name)
        texts = variant_texts(
            entry.primary_original_color,
            entry.counterfactual_color,
            matched_neutral_word(entry.counterfactual_color, notes_config["neutral_words"]),
        )

        for spec in specs:
            neutral_answer, neutral_logits = answer_and_color_logits(
                model, processor, render_note(image, texts["neutral"], spec).image,
                prompt, token_ids, config.max_new_tokens,
            )
            attack_answer, attack_logits = answer_and_color_logits(
                model, processor, render_note(image, texts["incongruent"], spec).image,
                prompt, token_ids, config.max_new_tokens,
            )
            cf = entry.counterfactual_color
            # Lietzow's convention: max over the accepted colours, per run.
            ref_neutral = best_accepted_color(neutral_logits, entry.original_colors)
            ref_attack = best_accepted_color(attack_logits, entry.original_colors)
            records.append(
                ObjectRecord(
                    object_id=entry.object_id,
                    object_name=entry.object_name,
                    original_colors=entry.original_colors,
                    counterfactual_color=cf,
                    answer_neutral=neutral_answer,
                    answer_incongruent=attack_answer,
                    delta_neutral=neutral_logits[ref_neutral] - neutral_logits[cf],
                    delta_incongruent=attack_logits[ref_attack] - attack_logits[cf],
                    layout=spec.name,
                    reference_color=ref_neutral,
                    color_logits_neutral=neutral_logits,
                    color_logits_incongruent=attack_logits,
                )
            )

    summary = summarise_screen(records, vocab, min_usable=config.min_usable)
    summary["color_vocabulary"] = vocab
    return records, summary
