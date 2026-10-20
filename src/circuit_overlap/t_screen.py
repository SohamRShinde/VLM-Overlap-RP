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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .data import ObjectEntry, collect_color_vocabulary
from .notes import NoteSpec, matched_neutral_word, render_note, variant_texts
from .screen import ObjectRecord, parse_color_word, summarise_screen

__all__ = [
    "VISUAL_PROMPT",
    "answer_and_color_logits_batch",
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
    # One colour word is all we score, and the suffix asks for exactly that;
    # 8 tokens spent decode steps for nothing.
    max_new_tokens: int = 4
    # Renders scored per forward pass. Every image gets the same pixel budget,
    # so sequence lengths are identical and no padding is involved.
    batch_size: int = 6
    # D24 variants to run. "congruent" is required for the D38 threshold: it is
    # the "a note exists and names a colour" baseline that carries no conflict.
    variants: tuple[str, ...] = ("neutral", "incongruent", "congruent")
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
            anchor=tuple(l.get("anchor", (0.5, 0.5))),
            rotation_deg=l.get("rotation_deg", 0.0),
            mode=l.get("mode", "fixed"),
            uppercase=l.get("uppercase", False),
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


def native_bf16(capability: tuple[int, int] | None) -> bool:
    """Whether bf16 is a *hardware* dtype on a device of this compute capability.

    ``torch.cuda.is_bf16_supported()`` answers True on Turing (T4, sm_75)
    because it counts **emulation**, and emulated bf16 is much slower than
    fp16. Observed 2026-10-19: a T4 reported bf16 support, the 4-bit compute
    dtype was set to bfloat16, and the screen ran at 8.4 s/generation.

    Native bf16 starts with Ampere (sm_80).
    """
    if capability is None:
        return False
    return capability[0] >= 8


def load_model(config: ScreenConfig):
    """Load Qwen2.5-VL and its processor, mirroring Lietzow's loader.

    Their 4-bit setup (``vlm/qwen.py``): nf4, double quantisation, and a compute
    dtype of bfloat16 where the GPU supports it, float16 otherwise.

    We deviate from their code in one place, deliberately: they gate on
    ``torch.cuda.is_bf16_supported()``, which counts emulation and so returns
    True on a T4. ``native_bf16`` requires real hardware support (Ampere or
    newer), so a T4 computes in fp16. Their own runs were presumably on
    Ampere+, where both tests agree. Record the compute dtype with every
    result: it is a real difference from their setup.
    """
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    compute_dtype = (
        torch.bfloat16 if native_bf16(capability) else getattr(torch, config.dtype)
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


def _build_batch(processor, images, prompt: str):
    messages = [
        {"role": "user",
         "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor(
        text=[text] * len(images), images=list(images), padding=True,
        return_tensors="pt",
    )


def answer_and_color_logits_batch(
    model, processor, images, prompt: str, color_token_ids: dict[str, list[int]],
    max_new_tokens: int = 4,
) -> list[tuple[str, dict[str, float]]]:
    """Score several renders of one prompt in a single generate pass.

    Every image carries the same pixel budget, so the sequences have equal
    length and the batch needs no padding -- which matters, because Qwen's
    tokenizer pads on the right and right padding breaks generation. Unequal
    lengths raise rather than silently producing garbage.

    Returns one ``(answer, color_logits)`` per image, in order. ``scores[0]``
    holds the logits that produced each row's first generated token: the
    position the whole study intervenes on (plan 3.4).
    """
    import torch

    inputs = _build_batch(processor, images, prompt).to(model.device)
    lengths = inputs["attention_mask"].sum(dim=1)
    if not bool((lengths == lengths[0]).all()):
        raise ValueError(
            "batched renders have unequal prompt lengths; pin the pixel budget "
            "(min_pixels == max_pixels) or set batch_size=1"
        )

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )

    first = out.scores[0].float()
    prompt_len = inputs["input_ids"].shape[1]
    results = []
    for row in range(len(images)):
        color_logits = {
            c: float(max(first[row, i] for i in ids))
            for c, ids in color_token_ids.items()
        }
        answer = processor.tokenizer.decode(
            out.sequences[row, prompt_len:], skip_special_tokens=True
        ).strip()
        results.append((answer, color_logits))
    return results


def answer_and_color_logits(
    model, processor, image, prompt: str, color_token_ids: dict[str, list[int]],
    max_new_tokens: int = 4,
) -> tuple[str, dict[str, float]]:
    """Single-image wrapper over :func:`answer_and_color_logits_batch`."""
    return answer_and_color_logits_batch(
        model, processor, [image], prompt, color_token_ids, max_new_tokens
    )[0]


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
    checkpoint_path: str | Path | None = None,
) -> tuple[list[ObjectRecord], dict]:
    """Run the screen over a manifest and return records plus a summary.

    Renders ``config.variants`` for every layout and scores them in batches of
    ``config.batch_size``.

    checkpoint_path:
        Append one JSON line per object as it finishes, and skip objects already
        present when resuming. A 469-object run takes hours on a T4, so a lost
        session must not cost the whole thing (plan section 5).
    """
    config = config or ScreenConfig()
    notes_config = notes_config or load_notes_config()

    vocab = sorted(
        set(notes_config["color_vocabulary"]) | set(collect_color_vocabulary(entries))
    )
    specs = notes_config["specs"]
    token_ids, collisions = color_first_token_ids(processor.tokenizer, vocab)
    if collisions:
        raise ValueError(
            f"colour words sharing a first token (D26): {collisions}. "
            "Drop the affected objects, or the words from configs/notes.json, "
            "before running -- the logit difference is undefined for them."
        )

    done: set[str] = set()
    records: list[ObjectRecord] = []
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    if ckpt and ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            done.add(payload["object_id"])
            records.extend(ObjectRecord(**r) for r in payload["records"])
        if progress:
            print(f"resuming: {len(done)} objects already in {ckpt}", flush=True)
    elif ckpt:
        ckpt.parent.mkdir(parents=True, exist_ok=True)

    todo = [e for e in entries if e.object_id not in done]
    started = time.time()
    for n, entry in enumerate(todo, 1):
        image = entry.open_image()
        prompt = config.prompt_template.format(object=entry.object_name)
        texts = variant_texts(
            entry.primary_original_color,
            entry.counterfactual_color,
            matched_neutral_word(entry.counterfactual_color, notes_config["neutral_words"]),
        )
        wanted = [v for v in config.variants if texts.get(v)]

        jobs = [(spec, v) for spec in specs for v in wanted]
        renders = [
            render_note(image, texts[v], spec, bbox=entry.object_bbox)
            for spec, v in jobs
        ]
        step = max(1, config.batch_size)
        scored: list[tuple[str, dict[str, float]]] = []
        for start in range(0, len(renders), step):
            chunk = renders[start : start + step]
            scored.extend(
                answer_and_color_logits_batch(
                    model, processor, [r.image for r in chunk], prompt,
                    token_ids, config.max_new_tokens,
                )
            )

        got: dict[str, dict[str, tuple]] = {}
        shown: dict[str, dict[str, object]] = {}
        for (spec, variant), render, result in zip(jobs, renders, scored):
            got.setdefault(spec.name, {})[variant] = result
            shown.setdefault(spec.name, {})[variant] = render

        cf = entry.counterfactual_color
        object_records: list[ObjectRecord] = []
        for spec in specs:
            per_variant = got[spec.name]
            neutral_answer, neutral_logits = per_variant["neutral"]
            attack_answer, attack_logits = per_variant["incongruent"]
            ref_neutral = best_accepted_color(neutral_logits, entry.original_colors)
            ref_attack = best_accepted_color(attack_logits, entry.original_colors)
            attack_render = shown[spec.name]["incongruent"]

            congruent = per_variant.get("congruent")
            if congruent is None:
                congruent_answer = congruent_logits = delta_congruent = None
            else:
                congruent_answer, congruent_logits = congruent
                ref_congruent = best_accepted_color(
                    congruent_logits, entry.original_colors
                )
                delta_congruent = (
                    congruent_logits[ref_congruent] - congruent_logits[cf]
                )

            object_records.append(
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
                    answer_congruent=congruent_answer,
                    delta_congruent=delta_congruent,
                    color_logits_congruent=congruent_logits,
                    note_area_frac=attack_render.area_frac,
                    object_occlusion=attack_render.object_occlusion(entry.object_bbox),
                )
            )

        records.extend(object_records)
        if ckpt:
            with ckpt.open("a") as f:
                f.write(json.dumps({
                    "object_id": entry.object_id,
                    "records": [asdict(r) for r in object_records],
                }) + "\n")
        if progress and (n % 25 == 0 or n == len(todo)):
            rate = (time.time() - started) / n
            print(f"  {n}/{len(todo)} objects  {rate:.1f}s/object  "
                  f"~{rate * (len(todo) - n) / 60:.0f} min left", flush=True)

    summary = summarise_screen(records, vocab, min_usable=config.min_usable)
    summary["color_vocabulary"] = vocab
    return records, summary
