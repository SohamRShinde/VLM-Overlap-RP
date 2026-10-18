"""The configs are data the experiments depend on; check they stay loadable."""

import json
from pathlib import Path

from circuit_overlap.notes import NoteSpec, matched_neutral_word
from circuit_overlap.t_screen import load_notes_config

ROOT = Path(__file__).resolve().parents[1]


def test_notes_config_builds_valid_specs():
    cfg = load_notes_config(ROOT / "configs" / "notes.json")
    assert len(cfg["specs"]) >= 2
    assert all(isinstance(s, NoteSpec) for s in cfg["specs"])
    assert len({s.name for s in cfg["specs"]}) == len(cfg["specs"])


def test_notes_config_has_a_length_matched_neutral_for_every_color():
    cfg = json.loads((ROOT / "configs" / "notes.json").read_text())
    for color in cfg["color_vocabulary"]:
        neutral = matched_neutral_word(color, cfg["neutral_words"])
        assert neutral not in cfg["color_vocabulary"]
        assert abs(len(neutral) - len(color)) <= 2


def test_data_config_is_an_unfilled_placeholder():
    """Guards against a dataset id being invented rather than verified."""
    cfg = json.loads((ROOT / "configs" / "data.json").read_text())
    ds = cfg["dataset"]
    assert ds["source_id"] is None, "fill this in only after verifying the release"
    assert ds["excluded_objects"] == ["radish", "spider"]


def test_screen_config_pins_a_grid_that_divides_into_tokens():
    """A grid that is not a multiple of 28 would break the mask mapping."""
    import math

    from circuit_overlap.tokens import token_grid_shape
    from circuit_overlap.t_screen import ScreenConfig

    side = int(math.isqrt(ScreenConfig().pixels))
    assert side * side == ScreenConfig().pixels
    rows, cols = token_grid_shape(side, side)
    assert rows == cols >= 16, "too coarse a grid makes note text unreadable"
