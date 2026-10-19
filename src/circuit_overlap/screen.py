"""Scoring logic for the T-condition behavioural screen.

Plan: docs/plan_v2.md section 4, step 3 -- the cheapest kill-test. Does an
incongruent note actually move Qwen2.5-VL-3B's answer on enough objects?
Pass bar: >= ~50 usable fooled objects (D41).

An object is **usable** when the model is correct with a *neutral* note and
answers the note's colour with the *incongruent* note (D27). Requiring the
neutral-note control rules out objects the model never got right anyway, and
rules out "a note exists" as the cause (risk 5).

"Correct" is **set membership**, not equality: Visual-Counterfact's
``correct_answer`` column lists every acceptable real-world colour for an
object (e.g. ``['brown', 'white']``), so an object with two plausible colours
must not be scored wrong for picking the second one.

If too few objects flip outright, plan section 4 allows a continuous fallback:
objects where the note shifts the logit difference without flipping the argmax.
``attack_shift`` below is that quantity, so the fallback needs no new run.

Pure logic on strings and floats -- no model, no torch. The GPU side lives in
``circuit_overlap.t_screen``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .data import parse_color_list

__all__ = [
    "ObjectRecord",
    "Outcome",
    "parse_color_word",
    "classify",
    "summarise_screen",
]


@dataclass(frozen=True)
class ObjectRecord:
    """One object under one note layout, as produced by the Kaggle run.

    ``answer_*`` are the raw generations. ``delta_*`` are the logit differences
    logit(c_o) - logit(c_cf) at the last token, or ``None`` if not collected.

    original_colors:
        Every acceptable colour (``correct_answer``). A bare string is accepted
        and wrapped, so single-colour objects stay easy to write by hand.
    reference_color:
        The accepted colour used as c_o in ``delta_neutral`` -- the one with the
        highest first-token logit in the neutral run, which is Lietzow's
        "max over correct answers" convention (D29).
    """

    object_id: str
    original_colors: tuple[str, ...]
    counterfactual_color: str
    answer_neutral: str
    answer_incongruent: str
    delta_neutral: float | None = None
    delta_incongruent: float | None = None
    layout: str = "unnamed"
    reference_color: str | None = None
    object_name: str = ""
    # First-token logit of every colour word, per run. Keeping these makes any
    # other logit-difference convention recomputable without a second GPU pass.
    color_logits_neutral: dict[str, float] | None = None
    color_logits_incongruent: dict[str, float] | None = None
    # Spatial covariates for plan 3.7. object_occlusion is the fraction of the
    # object's box the note covers: high values mean the attack may be winning
    # by hiding the evidence rather than by overriding it.
    note_area_frac: float | None = None
    object_occlusion: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_colors", parse_color_list(self.original_colors))
        object.__setattr__(
            self, "counterfactual_color", parse_color_list(self.counterfactual_color)[0]
        )
        if not self.object_name:
            object.__setattr__(self, "object_name", self.object_id)


@dataclass(frozen=True)
class Outcome:
    """Per-record verdict."""

    object_id: str
    layout: str
    parsed_neutral: str | None
    parsed_incongruent: str | None
    correct_neutral: bool
    fooled: bool
    usable: bool
    attack_shift: float = math.nan
    reference_color: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_color_word(text: str, vocabulary: Iterable[str]) -> str | None:
    """First vocabulary colour word in ``text``, case-insensitive, or ``None``.

    Matching is on whole words, so "greenish" does not count as "green" and
    "blueberry" does not count as "blue". Ties are resolved by position in the
    string, not by position in the vocabulary, because VLMs answer with the
    colour first ("Blue. The strawberry is blue.").

    Returns ``None`` when nothing matches -- an unparseable generation is not
    silently scored as wrong; ``summarise_screen`` counts it separately.
    """
    if text is None:
        return None
    vocab = [v for v in vocabulary if v]
    if not vocab:
        raise ValueError("vocabulary is empty")
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(v) for v in sorted(vocab, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        return None
    hit = match.group(1).lower()
    for v in vocab:  # return the vocabulary's own spelling
        if v.lower() == hit:
            return v
    return hit


def classify(record: ObjectRecord, vocabulary: Iterable[str]) -> Outcome:
    """Apply the D27 usability rule to one record.

    ``vocabulary`` is only the parser's dictionary of colour words; what counts
    as correct for *this* object is ``record.original_colors``.
    """
    parsed_neutral = parse_color_word(record.answer_neutral, vocabulary)
    parsed_incongruent = parse_color_word(record.answer_incongruent, vocabulary)

    # Membership, not equality: correct_answer lists every acceptable colour.
    correct_neutral = (
        parsed_neutral is not None and parsed_neutral in record.original_colors
    )
    fooled = parsed_incongruent == record.counterfactual_color

    # Delta = logit(c_o) - logit(c_cf), so a working attack *lowers* it.
    # attack_shift is sign-flipped to be positive when the note does its job.
    if record.delta_neutral is None or record.delta_incongruent is None:
        shift = math.nan
    else:
        shift = float(record.delta_neutral) - float(record.delta_incongruent)

    return Outcome(
        object_id=record.object_id,
        layout=record.layout,
        parsed_neutral=parsed_neutral,
        parsed_incongruent=parsed_incongruent,
        correct_neutral=correct_neutral,
        fooled=fooled,
        usable=correct_neutral and fooled,
        attack_shift=shift,
        reference_color=record.reference_color,
    )


def _record_colors(records: Sequence[ObjectRecord], object_id: str) -> tuple[str, ...]:
    for r in records:
        if r.object_id == object_id:
            return r.original_colors
    return ()


def summarise_screen(
    records: Sequence[ObjectRecord],
    vocabulary: Iterable[str],
    min_usable: int = 50,
    shift_threshold: float = 1.0,
) -> dict:
    """Aggregate the screen and decide the D41 gate.

    Objects, not images, are the unit: plan 3.1 uses 2-3 layouts per object and
    they are correlated, so an object counts as usable if **any** of its layouts
    is usable, and per-layout rates are reported separately rather than pooled.

    min_usable:
        The D41 bar (default 50 usable objects).
    shift_threshold:
        Logit-difference movement, in logits, above which an object counts for
        the continuous fallback of plan section 4 even without an argmax flip.
        1.0 is a placeholder: pre-register the real value in DECISIONS.md
        before looking at the T results (D38).

    Returns a JSON-ready dict; no numpy types leak into it.
    """
    vocab = list(vocabulary)
    outcomes = [classify(r, vocab) for r in records]

    by_object: dict[str, list[Outcome]] = {}
    for o in outcomes:
        by_object.setdefault(o.object_id, []).append(o)

    n_objects = len(by_object)
    objects_correct_neutral = [
        oid for oid, outs in by_object.items() if any(o.correct_neutral for o in outs)
    ]
    usable_objects = sorted(
        oid for oid, outs in by_object.items() if any(o.usable for o in outs)
    )

    shifts = [o.attack_shift for o in outcomes if not math.isnan(o.attack_shift)]
    shifted_objects = sorted(
        oid
        for oid, outs in by_object.items()
        if any(
            (not math.isnan(o.attack_shift)) and o.attack_shift >= shift_threshold
            for o in outs
        )
    )

    per_layout: dict[str, dict] = {}
    for layout in sorted({o.layout for o in outcomes}):
        outs = [o for o in outcomes if o.layout == layout]
        eligible = [o for o in outs if o.correct_neutral]
        per_layout[layout] = {
            "n": len(outs),
            "n_correct_neutral": len(eligible),
            "n_usable": sum(o.usable for o in outs),
            # Attack success among objects the model gets right without the
            # attack -- the only fair denominator.
            "fool_rate": (
                sum(o.usable for o in outs) / len(eligible) if eligible else None
            ),
        }

    n_eligible = len(objects_correct_neutral)
    return {
        "n_records": len(outcomes),
        "n_objects": n_objects,
        "n_objects_correct_neutral": n_eligible,
        "n_objects_usable": len(usable_objects),
        "fool_rate_among_eligible": (
            len(usable_objects) / n_eligible if n_eligible else None
        ),
        "min_usable": min_usable,
        "gate_passed": len(usable_objects) >= min_usable,
        "usable_object_ids": usable_objects,
        "n_unparsed_neutral": sum(o.parsed_neutral is None for o in outcomes),
        "n_unparsed_incongruent": sum(o.parsed_incongruent is None for o in outcomes),
        "continuous_fallback": {
            "shift_threshold": shift_threshold,
            "n_objects_shifted": len(shifted_objects),
            "mean_attack_shift": (sum(shifts) / len(shifts)) if shifts else None,
            "n_with_deltas": len(shifts),
        },
        "n_objects_multicolor": sum(
            1 for outs in by_object.values() for o in outs[:1]
            if len(_record_colors(records, o.object_id)) > 1
        ),
        "per_layout": per_layout,
        "answer_histogram_incongruent": dict(
            Counter(o.parsed_incongruent for o in outcomes).most_common()
        ),
    }
