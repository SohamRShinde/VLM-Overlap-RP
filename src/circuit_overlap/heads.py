"""Head-set loading and validation.

Head lists are **data**, never literals in code (CLAUDE.md, plan section 7).
They live in ``configs/heads_*.json`` and are transcribed from Lietzow et al.'s
Tables 5-7 -- their repo does not publish them (``data/`` is git-ignored there).

A head is a 0-based ``(layer, head)`` tuple (D30, D60), matching their
``interventions.head_knockout``, which zeroes the ``o_proj`` input slice
``[head*head_dim : (head+1)*head_dim]`` at the last token.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

__all__ = ["HeadSet", "load_head_set", "parse_head_label", "format_head_label"]


def parse_head_label(label: str) -> tuple[int, int]:
    """``"L26H5"`` -> ``(26, 5)``. Used for the Table 7 single-head entries."""
    text = label.strip().upper()
    if not text.startswith("L") or "H" not in text:
        raise ValueError(f"not a head label: {label!r}")
    layer_part, _, head_part = text[1:].partition("H")
    try:
        return int(layer_part), int(head_part)
    except ValueError as exc:
        raise ValueError(f"not a head label: {label!r}") from exc


def format_head_label(head: tuple[int, int]) -> str:
    """``(26, 5)`` -> ``"L26H5"``, for tables and figures."""
    return f"L{head[0]}H{head[1]}"


@dataclass(frozen=True)
class HeadSet:
    """One model's classified heads, as loaded from a config file."""

    name: str
    checkpoint: str
    promoting: tuple[tuple[int, int], ...]
    suppressing: tuple[tuple[int, int], ...]
    single_head_prior_flip_pct: dict[str, float]
    source: str
    path: Path

    def __len__(self) -> int:
        return len(self.promoting)

    @property
    def classified(self) -> tuple[tuple[int, int], ...]:
        """Promoting and suppressing together, sorted."""
        return tuple(sorted(set(self.promoting) | set(self.suppressing)))

    def layer_range(self) -> tuple[int, int]:
        layers = [l for l, _ in self.classified]
        return min(layers), max(layers)


def _as_heads(raw: Iterable, field: str, path: Path) -> tuple[tuple[int, int], ...]:
    heads: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{path}: {field} entry {item!r} is not [layer, head]")
        layer, head = item
        if not isinstance(layer, int) or not isinstance(head, int):
            raise ValueError(f"{path}: {field} entry {item!r} must be integers")
        if layer < 0 or head < 0:
            raise ValueError(
                f"{path}: {field} entry {item!r} is negative; indices are 0-based"
            )
        heads.append((layer, head))
    if len(set(heads)) != len(heads):
        dupes = sorted({h for h in heads if heads.count(h) > 1})
        raise ValueError(f"{path}: duplicate heads in {field}: {dupes}")
    return tuple(heads)


def load_head_set(
    path: str | Path,
    n_layers: int | None = None,
    n_heads_per_layer: int | None = None,
) -> HeadSet:
    """Load and validate one ``configs/heads_*.json``.

    Checks that would otherwise surface as a silently wrong circuit:

    * indices are 0-based non-negative integer pairs, with no duplicates;
    * promoting and suppressing do not overlap -- a head cannot do both;
    * the counts match the ``counts_reported`` block, catching a transcription
      slip against the paper's own "count" column;
    * every Table 7 single-head entry is in the promoting set;
    * optionally, indices fit a model's actual shape. Pass ``n_layers`` and
      ``n_heads_per_layer`` from the loaded config once a model is in memory;
      the config files leave them null because they are still unverified.
    """
    p = Path(path)
    raw = json.loads(p.read_text())

    promoting = _as_heads(raw.get("promoting", []), "promoting", p)
    suppressing = _as_heads(raw.get("suppressing", []), "suppressing", p)

    overlap = sorted(set(promoting) & set(suppressing))
    if overlap:
        raise ValueError(f"{p}: heads classified both promoting and suppressing: {overlap}")

    counts = raw.get("counts_reported", {})
    for field, heads in (("promoting", promoting), ("suppressing", suppressing)):
        expected = counts.get(field)
        if expected is not None and expected != len(heads):
            raise ValueError(
                f"{p}: {field} has {len(heads)} heads but counts_reported says "
                f"{expected} -- transcription error against the paper's table"
            )

    single = dict(raw.get("single_head_prior_flip_pct", {}))
    for label in single:
        head = parse_head_label(label)
        if head not in promoting:
            raise ValueError(
                f"{p}: Table 7 head {label} is not in the promoting set; "
                "the two tables disagree, or the transcription is wrong"
            )

    if n_layers is not None or n_heads_per_layer is not None:
        for layer, head in promoting + suppressing:
            if n_layers is not None and layer >= n_layers:
                raise ValueError(
                    f"{p}: layer {layer} is out of range for a {n_layers}-layer model; "
                    "check whether the paper's indices are 1-based"
                )
            if n_heads_per_layer is not None and head >= n_heads_per_layer:
                raise ValueError(
                    f"{p}: head {head} is out of range for {n_heads_per_layer} heads/layer"
                )

    return HeadSet(
        name=raw.get("paper_model_name", p.stem),
        checkpoint=raw.get("checkpoint", ""),
        promoting=promoting,
        suppressing=suppressing,
        single_head_prior_flip_pct=single,
        source=raw.get("_source", ""),
        path=p,
    )
