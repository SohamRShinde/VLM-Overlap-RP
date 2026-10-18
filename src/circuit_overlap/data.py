"""Object manifest: the seam between the dataset and the rest of the code.

Source (verified on the Hub by the user, 18 Oct 2026): ``mgolov/Visual-Counterfact``,
``split="color"``, 493 rows. Columns are **not** what plan_v2 section 3.1 guessed:

    object             the object noun, e.g. "strawberry"
    correct_answer     a *stringified Python list* of acceptable real-world
                       colours, e.g. "['brown', 'white']"
    incorrect_answer   the counterfactual colour
    original_image     the real-colour image  (what the T condition uses)
    counterfact_image  the recoloured image   (what the P and V conditions use)

Two consequences that reach the whole codebase:

1. **The original colour is a set, not a value.** "Correct" means the answer is
   *in* ``original_colors``. ``parse_color_list`` below accepts the stringified
   list, a real list, or a bare string, so neither the notebook nor the scorer
   has to care which form it was handed.
2. **The logit difference needs one reference colour** chosen from that set --
   see ``circuit_overlap.t_screen.run_screen``, which uses the model's own
   neutral-note answer and records it in every result row.

Open: 493 rows here vs the 469 (467 after dropping radish and spider) that
Lietzow report. Release or filtering difference, unresolved -- docs/STATE.md.

Everything downstream reads a *manifest*, a JSON list the notebook writes:

    [{"object_id": "strawberry_017",      # unique key
      "object_name": "strawberry",        # the noun used in the prompt
      "image": "images/strawberry_017.png",
      "original_colors": ["red"],         # or "['brown', 'white']"
      "counterfactual_color": "blue"}, ...]
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "ObjectEntry",
    "parse_color_list",
    "load_manifest",
    "check_colors_in_vocabulary",
    "collect_color_vocabulary",
]


def parse_color_list(value: Any) -> tuple[str, ...]:
    """Normalise a colour field to a tuple of lowercase colour words.

    Handles every shape Visual-Counterfact and a hand-written manifest can
    produce:

    >>> parse_color_list("['brown', 'white']")
    ('brown', 'white')
    >>> parse_color_list("red")
    ('red',)
    >>> parse_color_list(["Red", " blue "])
    ('red', 'blue')

    A stringified list is parsed with ``ast.literal_eval`` (literals only, no
    code execution). Order is preserved and duplicates are dropped, so the
    first element is a stable "primary" colour. Raises on an empty or
    unparseable field rather than returning an empty tuple, which would
    silently make an object impossible to score.
    """
    if value is None:
        raise ValueError("colour field is None")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("colour field is empty")
        if text.startswith("[") or text.startswith("("):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"cannot parse colour list {value!r}") from exc
            items: Iterable[Any] = parsed
        else:
            items = [text]
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        raise TypeError(f"unsupported colour field type: {type(value).__name__}")

    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"colour entries must be strings, got {item!r}")
        word = item.strip().lower()
        if not word:
            continue
        if word not in out:
            out.append(word)
    if not out:
        raise ValueError(f"colour field has no usable words: {value!r}")
    return tuple(out)


@dataclass(frozen=True)
class ObjectEntry:
    """One object: its image, its acceptable colours, and its attack colour.

    object_id:
        Unique key. The dataset's ``object`` column is *not* guaranteed unique
        across its 493 rows, so the notebook suffixes it.
    object_name:
        The noun that goes into the prompt ("What color is this {object} here?").
    original_colors:
        Every colour that counts as correct (``correct_answer``).
    counterfactual_color:
        The colour the note claims (``incorrect_answer``).
    """

    object_id: str
    image: Path
    original_colors: tuple[str, ...]
    counterfactual_color: str
    object_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_colors", parse_color_list(self.original_colors))
        object.__setattr__(
            self, "counterfactual_color", parse_color_list(self.counterfactual_color)[0]
        )
        if not self.object_name:
            object.__setattr__(self, "object_name", self.object_id)
        if self.counterfactual_color in self.original_colors:
            raise ValueError(
                f"object {self.object_id!r}: counterfactual colour "
                f"{self.counterfactual_color!r} is also an accepted answer "
                f"{self.original_colors} -- the conflict would be undefined"
            )

    @property
    def primary_original_color(self) -> str:
        """First acceptable colour. A fallback only: prefer the model's own
        neutral-note answer as the reference for Delta (see t_screen)."""
        return self.original_colors[0]

    def open_image(self):
        from PIL import Image  # local import: manifest handling needs no PIL

        return Image.open(self.image).convert("RGB")


_REQUIRED = ("object_id", "image", "counterfactual_color")


def _colors_field(item: dict, index: int) -> Any:
    for key in ("original_colors", "original_color", "correct_answer"):
        if key in item:
            return item[key]
    raise ValueError(
        f"entry {index} has no original colours "
        "(expected 'original_colors', 'original_color' or 'correct_answer')"
    )


def load_manifest(path: str | Path, require_images: bool = True) -> list[ObjectEntry]:
    """Read and validate an object manifest.

    Raises on duplicate ids, missing fields, unparseable colour lists, or a
    counterfactual colour that is also an accepted answer -- each would quietly
    corrupt the screen's denominators rather than fail loudly.
    """
    p = Path(path)
    raw = json.loads(p.read_text())
    items = raw["objects"] if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError(f"{p}: manifest must be a list, or a dict with 'objects'")

    entries: list[ObjectEntry] = []
    seen: set[str] = set()
    for i, item in enumerate(items):
        missing = [k for k in _REQUIRED if k not in item]
        if missing:
            raise ValueError(f"{p}: entry {i} is missing {missing}")
        oid = str(item["object_id"])
        if oid in seen:
            raise ValueError(
                f"{p}: duplicate object_id {oid!r}; the dataset's 'object' column "
                "repeats, so give each row a unique id"
            )
        seen.add(oid)
        image = Path(item["image"])
        if not image.is_absolute():
            image = p.parent / image
        if require_images and not image.exists():
            raise FileNotFoundError(f"{p}: object {oid!r} image not found: {image}")
        try:
            entry = ObjectEntry(
                object_id=oid,
                image=image,
                original_colors=_colors_field(item, i),
                counterfactual_color=item["counterfactual_color"],
                object_name=str(item.get("object_name", "")),
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{p}: entry {i} ({oid!r}): {exc}") from exc
        entries.append(entry)
    if not entries:
        raise ValueError(f"{p}: manifest is empty")
    return entries


def collect_color_vocabulary(entries: Iterable[ObjectEntry]) -> list[str]:
    """Sorted union of every colour word the manifest uses.

    Use it to fill ``configs/notes.json`` with the dataset's real vocabulary
    instead of the placeholder list, and to drive the D26 single-token check.
    """
    words: set[str] = set()
    for e in entries:
        words.update(e.original_colors)
        words.add(e.counterfactual_color)
    return sorted(words)


def check_colors_in_vocabulary(
    entries: Iterable[ObjectEntry], vocabulary: Iterable[str]
) -> list[str]:
    """Colour words used by the manifest but absent from the parser vocabulary.

    A non-empty return means ``screen.parse_color_word`` can never score those
    objects; fix configs/notes.json before running.
    """
    vocab = {v.lower() for v in vocabulary}
    return sorted(w for w in collect_color_vocabulary(entries) if w not in vocab)
