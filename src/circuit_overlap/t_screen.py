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

from .data import ObjectEntry
from .notes import NoteSpec, matched_neutral_word, render_note, variant_texts
from .screen import ObjectRecord, summarise_screen

__all__ = [
    "VISUAL_PROMPT",
    "ScreenConfig",
    "load_notes_config",
    "single_first_token_ids",
    "load_model",
    "answer_and_delta",
    "run_screen",
]

# [VERIFY] against Lietzow's repo before the replication gate (D22).
VISUAL_PROMPT = "What color is this {object} here?"


@dataclass(frozen=True)
class ScreenConfig:
    checkpoint: str = "Qwen/Qwen2.5-VL-3B-Instruct"  # [VERIFY] blocker 2
    # min_pixels == max_pixels pins the image grid so masks map to tokens (3.7).
    # 448x448 -> 16x16 = 256 image tokens. Do not lower this without checking
    # the note is still legible: at 224x224 the note's text is a few pixels tall,
    # and an unreadable note would look exactly like a model that resists the
    # attack -- a false negative on the kill-test. Raising it costs GPU time.
    pixels: int = 448 * 448
    max_new_tokens: int = 8
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


def single_first_token_ids(
    tokenizer, words: Iterable[str], prefix: str = " "
) -> tuple[dict[str, int], list[str]]:
    """First-token id per word, plus the words that fail the D26 restriction.

    A word fails if it does not survive as a single token, or if it shares its
    first token with another word -- in either case logit(c_o) - logit(c_cf) is
    not measuring what it claims to.
    """
    ids: dict[str, int] = {}
    multi: list[str] = []
    for w in words:
        pieces = tokenizer(prefix + w, add_special_tokens=False)["input_ids"]
        if not pieces:
            multi.append(w)
            continue
        ids[w] = int(pieces[0])
        if len(pieces) > 1:
            multi.append(w)
    seen: dict[int, str] = {}
    for w, i in ids.items():
        if i in seen:
            multi.extend([w, seen[i]])
        seen[i] = w
    return ids, sorted(set(multi))


def load_model(config: ScreenConfig):
    """Load Qwen2.5-VL and its processor with a pinned image grid."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = getattr(torch, config.dtype)
    processor = AutoProcessor.from_pretrained(
        config.checkpoint, min_pixels=config.pixels, max_pixels=config.pixels
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.checkpoint, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    return model, processor


def answer_and_delta(
    model, processor, image, prompt: str, id_co: int, id_cf: int, max_new_tokens: int = 8
):
    """Greedy answer plus Delta = logit(c_o) - logit(c_cf) on the first new token.

    One ``generate`` call returns both: ``scores[0]`` holds the logits that
    produced the first generated token, which is the position the whole study
    intervenes on (plan 3.4).
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
    delta = float(first_logits[id_co] - first_logits[id_cf])
    generated = out.sequences[0, inputs["input_ids"].shape[1] :]
    answer = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer, delta


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
    vocab = notes_config["color_vocabulary"]
    specs = notes_config["specs"]
    token_ids, bad = single_first_token_ids(processor.tokenizer, vocab)
    if bad:
        raise ValueError(
            f"colour words violating D26 (not a unique single first token): {bad}. "
            "Drop them from configs/notes.json before running."
        )

    records: list[ObjectRecord] = []
    for n, entry in enumerate(entries, 1):
        if progress and n % 25 == 0:
            print(f"  {n}/{len(entries)} objects", flush=True)
        image = entry.open_image()
        prompt = config.prompt_template.format(object=entry.object_id)
        texts = variant_texts(
            entry.original_color,
            entry.counterfactual_color,
            matched_neutral_word(entry.counterfactual_color, notes_config["neutral_words"]),
        )
        id_co = token_ids[entry.original_color]
        id_cf = token_ids[entry.counterfactual_color]

        for spec in specs:
            answers = {}
            for variant in ("neutral", "incongruent"):
                rendered = render_note(image, texts[variant], spec)
                answers[variant] = answer_and_delta(
                    model, processor, rendered.image, prompt, id_co, id_cf,
                    max_new_tokens=config.max_new_tokens,
                )
            records.append(
                ObjectRecord(
                    object_id=entry.object_id,
                    original_color=entry.original_color,
                    counterfactual_color=entry.counterfactual_color,
                    answer_neutral=answers["neutral"][0],
                    answer_incongruent=answers["incongruent"][0],
                    delta_neutral=answers["neutral"][1],
                    delta_incongruent=answers["incongruent"][1],
                    layout=spec.name,
                )
            )

    summary = summarise_screen(records, vocab, min_usable=config.min_usable)
    return records, summary
