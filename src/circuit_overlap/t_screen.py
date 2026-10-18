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
    "ScreenConfig",
    "load_notes_config",
    "single_first_token_ids",
    "load_model",
    "answer_and_color_logits",
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


def answer_and_color_logits(
    model, processor, image, prompt: str, color_token_ids: dict[str, int],
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
    color_logits = {c: float(first_logits[i]) for c, i in color_token_ids.items()}
    generated = out.sequences[0, inputs["input_ids"].shape[1] :]
    answer = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer, color_logits


def choose_reference_color(
    answer: str, original_colors: tuple[str, ...], vocabulary: Sequence[str]
) -> str:
    """The c_o used in Delta when several colours are acceptable.

    Prefers the colour the model itself gave on the *neutral* note, provided it
    is one of the accepted answers: that is the belief the attack has to
    overturn, so the shift in Delta measures the attack rather than an arbitrary
    choice between two correct colours. Falls back to the first accepted colour
    when the neutral answer is unparseable or wrong, and the fallback is
    recorded in the result row either way.
    """
    parsed = parse_color_word(answer, vocabulary)
    if parsed is not None and parsed in original_colors:
        return parsed
    return original_colors[0]


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
    token_ids, bad = single_first_token_ids(processor.tokenizer, vocab)
    if bad:
        raise ValueError(
            f"colour words violating D26 (not a unique single first token): {bad}. "
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
            # Pick c_o only after the neutral run, then score both runs with it.
            reference = choose_reference_color(
                neutral_answer, entry.original_colors, vocab
            )
            attack_answer, attack_logits = answer_and_color_logits(
                model, processor, render_note(image, texts["incongruent"], spec).image,
                prompt, token_ids, config.max_new_tokens,
            )
            cf = entry.counterfactual_color
            records.append(
                ObjectRecord(
                    object_id=entry.object_id,
                    object_name=entry.object_name,
                    original_colors=entry.original_colors,
                    counterfactual_color=cf,
                    answer_neutral=neutral_answer,
                    answer_incongruent=attack_answer,
                    delta_neutral=neutral_logits[reference] - neutral_logits[cf],
                    delta_incongruent=attack_logits[reference] - attack_logits[cf],
                    layout=spec.name,
                    reference_color=reference,
                )
            )

    summary = summarise_screen(records, vocab, min_usable=config.min_usable)
    summary["color_vocabulary"] = vocab
    return records, summary
