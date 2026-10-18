"""Object manifest: the seam between the dataset and the rest of the code.

Visual-Counterfact (plan 3.1) is loaded on Kaggle, where its exact release and
field names are still **[VERIFY]** (docs/STATE.md). Rather than guess them here,
everything downstream reads a *manifest*: a plain JSON list of objects with an
image path, an original colour and a counterfactual colour. One notebook cell
adapts the dataset to this shape; all tested code works from the manifest.

Manifest format (a JSON list, or a dict with an "objects" key):

    [{"object_id": "strawberry",
      "image": "images/strawberry_original.png",
      "original_color": "red",
      "counterfactual_color": "blue"}, ...]

Paths are resolved relative to the manifest file unless absolute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

__all__ = ["ObjectEntry", "load_manifest", "check_colors_in_vocabulary"]


@dataclass(frozen=True)
class ObjectEntry:
    object_id: str
    image: Path
    original_color: str
    counterfactual_color: str

    def open_image(self):
        from PIL import Image  # local import: manifest handling needs no PIL

        return Image.open(self.image).convert("RGB")


_REQUIRED = ("object_id", "image", "original_color", "counterfactual_color")


def load_manifest(path: str | Path, require_images: bool = True) -> list[ObjectEntry]:
    """Read and validate an object manifest.

    Raises on duplicate object ids, missing fields, or an object whose original
    and counterfactual colours are equal -- each of which would quietly corrupt
    the screen's denominators rather than fail loudly.
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
            raise ValueError(f"{p}: duplicate object_id {oid!r}")
        seen.add(oid)
        if item["original_color"] == item["counterfactual_color"]:
            raise ValueError(
                f"{p}: object {oid!r} has equal original and counterfactual colours"
            )
        image = Path(item["image"])
        if not image.is_absolute():
            image = p.parent / image
        if require_images and not image.exists():
            raise FileNotFoundError(f"{p}: object {oid!r} image not found: {image}")
        entries.append(
            ObjectEntry(
                object_id=oid,
                image=image,
                original_color=str(item["original_color"]),
                counterfactual_color=str(item["counterfactual_color"]),
            )
        )
    if not entries:
        raise ValueError(f"{p}: manifest is empty")
    return entries


def check_colors_in_vocabulary(
    entries: Iterable[ObjectEntry], vocabulary: Iterable[str]
) -> list[str]:
    """Colour words used by the manifest but absent from the parser vocabulary.

    A non-empty return means ``screen.parse_color_word`` can never score those
    objects; fix the vocabulary in configs/notes.json before running.
    """
    vocab = {v.lower() for v in vocabulary}
    missing = set()
    for e in entries:
        for c in (e.original_color, e.counterfactual_color):
            if c.lower() not in vocab:
                missing.add(c)
    return sorted(missing)
