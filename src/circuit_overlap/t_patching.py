"""T-circuit discovery: the P patching pipeline, applied to the note contrast.

Design pre-registered in DECISIONS.md D71 before this ran. The GPU work reuses
``patching.LastTokenRunner`` / ``patch_example`` / ``classify_heads``
**unchanged** (D30, D68), so the P and T circuits come out of one pipeline and
only the pair of inputs differs:

    P: Prior prompt vs Visual prompt, on the counterfactual image
    T: incongruent-note vs neutral-note image, both with the Visual prompt

**Roles, not names.** ``patch_example`` calls its runs "prior" (source) and
"visual" (target). For T those field names are kept, so the records feed the
same scoring and classification code, but they mean:

    source  "prior"  = incongruent note  -- the competing signal wins
    target  "visual" = neutral note      -- the image wins

so a promoting T head is one that carries the competing answer into the
image-grounded run, the job P's prior-override heads do. The colour tokens
swap roles between the conditions (P's competing run answers the *real*
colour, T's answers the *note's*): plan 3.2's acknowledged asymmetry.

This module holds the CPU-testable pieces; torch is never imported here.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .notes import NoteSpec, matched_neutral_word, object_bbox_from_pair, render_note
from .replication import color_token_sets

__all__ = [
    "T_ROLES",
    "LAYOUT",
    "render_t_pair",
    "is_t_conflicting",
    "sweep_order",
    "has_nonfinite",
    "finite_records",
    "load_patching_records",
    "screen_object_index",
    "screen_cross_check",
]

T_ROLES = {
    "source": "incongruent note (competing signal wins); record fields *_prior",
    "target": "neutral note (image wins); record fields *_visual",
    "prompt": "Visual prompt for both runs",
}

# D71: the layout with the highest screen fool rate (58.3%) and the lowest
# occlusion (median 4.7%), so the note is least able to win by hiding evidence.
LAYOUT = "on_object_small"


def render_t_pair(original, counterfact, attack: str, neutral_words: Sequence[str],
                  spec: NoteSpec) -> dict:
    """Neutral- and incongruent-note renders of one object, as the screen made them.

    The object box comes from diffing the original and counterfactual images
    (D63), and the neutral word is length-matched to the attack colour, exactly
    as ``t_screen.run_screen`` does, so these are the screen's images.
    Both renders share one layout and one box: only the word on the note
    differs, which is what makes this a single-factor contrast (plan 1).
    """
    original = original.convert("RGB")
    bbox = object_bbox_from_pair(original, counterfact.convert("RGB"))
    neutral_word = matched_neutral_word(attack, neutral_words)
    neutral = render_note(original, neutral_word, spec, bbox=bbox)
    incongruent = render_note(original, attack, spec, bbox=bbox)
    return {
        "neutral": neutral.image,
        "incongruent": incongruent.image,
        "neutral_word": neutral_word,
        "bbox": bbox,
        "note_area_frac": incongruent.area_frac,
        "object_occlusion": incongruent.object_occlusion(bbox),
        # The note's pixel footprint must not depend on its word, or the
        # contrast is no longer single-factor.
        "same_note_footprint": bool((neutral.mask == incongruent.mask).all()),
    }


def is_t_conflicting(row: Mapping, top1_neutral: int, top1_incongruent: int,
                     color_ids: Mapping[str, Sequence[int]],
                     ambiguous: Iterable[str] = ()) -> bool:
    """T's "correctly conflicting": right on the neutral note, fooled by the attack.

    The role-mapped analogue of ``replication.is_correctly_conflicting``:
    the neutral run's top-1 is one of the real colours and the incongruent
    run's top-1 is the note's colour, both as first-token ids. Objects using a
    colour that shares a first token with another (D26) are excluded.
    """
    if set(ambiguous) & ({row["attack"]} | set(row["correct"])):
        return False
    attack, real = color_token_sets(row, color_ids)
    return top1_neutral in real and top1_incongruent in attack


def sweep_order(selected: Sequence[int], p_ids: Iterable[int], seed: int = 0) -> list[int]:
    """D71's sweep order: objects shared with P first, then the rest shuffled.

    A run this long can be cut off. In this order a partial run still yields
    the plan 3.5 shared-object comparison and an unbiased random sample, never
    the alphabetically first N objects.
    """
    p = set(p_ids)
    shared = sorted(i for i in selected if i in p)
    rest = sorted(i for i in selected if i not in p)
    rng = np.random.default_rng(seed)
    rest = [rest[k] for k in rng.permutation(len(rest))]
    return shared + rest


_LOGIT_FIELDS = (
    "prior_token_logit_prior_run", "visual_token_logit_prior_run",
    "prior_token_logit_visual_run", "visual_token_logit_visual_run",
)
_GRID_FIELDS = ("p2v_logit_prior", "p2v_logit_visual", "v2p_logit_prior", "v2p_logit_visual")


def has_nonfinite(record: Mapping) -> bool:
    """Whether any logit in a patching record is NaN or infinite (plan risk 7)."""
    for key in _LOGIT_FIELDS:
        value = record.get(key)
        if value is not None and not math.isfinite(value):
            return True
    for key in _GRID_FIELDS:
        grid = record.get(key)
        if grid is not None and not np.isfinite(np.asarray(grid, dtype=float)).all():
            return True
    return False


def finite_records(records: Iterable[Mapping]) -> tuple[list[Mapping], list]:
    """Split records into usable ones and the ids of those with non-finite logits.

    ``classify_heads`` refuses non-finite grids, so a single overflowed head in
    one object would otherwise stop the whole classification; dropping that
    object, and reporting it, is the plan-risk-7 policy used for the screen.
    """
    kept, dropped = [], []
    for r in records:
        (dropped.append(r.get("i")) if has_nonfinite(r) else kept.append(r))
    return kept, dropped


def load_patching_records(path: str | Path) -> list[dict]:
    """Per-object patching records from any of the forms they are saved in.

    * ``*.json``: a result envelope with ``result.records`` (the P file);
    * ``*.jsonl`` or ``*.jsonl.gz``: one record per line (the T files).

    Only records whose status is ``"patched"`` are returned.
    """
    path = Path(path)
    if path.name.endswith(".json"):
        records = json.loads(path.read_text())["result"]["records"]
    else:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt") as f:
            records = [json.loads(line) for line in f if line.strip()]
    return [r for r in records if r.get("status", "patched") == "patched"]


_SCREEN_ID = re.compile(r"_(\d{4})$")


def screen_object_index(object_id: str) -> int:
    """``"acorn_squash_0030"`` -> 30: the dataset row index in a screen id."""
    match = _SCREEN_ID.search(object_id)
    if not match:
        raise ValueError(f"not a screen object id: {object_id!r}")
    return int(match.group(1))


def screen_cross_check(selected: Iterable[int], screen_usable_ids: Iterable[str]) -> dict:
    """How the cached-path selection compares with the behavioural screen.

    They differ by construction -- the screen used ``generate`` with emulated
    bf16 and pooled three layouts, this uses one layout's first-token logits in
    fp16 -- so this is a sanity check, not a criterion.
    """
    sel = set(selected)
    screen = {screen_object_index(o) for o in screen_usable_ids}
    return {
        "n_selected": len(sel),
        "n_screen_usable": len(screen),
        "n_both": len(sel & screen),
        "n_selected_only": len(sel - screen),
        "n_screen_only": len(screen - sel),
        "share_of_selected_also_in_screen": len(sel & screen) / len(sel) if sel else float("nan"),
    }
