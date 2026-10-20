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


def test_data_config_records_who_verified_the_dataset():
    """Guards against a dataset id being invented rather than verified: an id
    may only appear alongside the date it was checked against the source."""
    cfg = json.loads((ROOT / "configs" / "data.json").read_text())
    ds = cfg["dataset"]
    assert ds["source_id"] is None or ds.get("verified_on"), (
        "a source_id must come with verified_on / verified_by"
    )
    assert ds["excluded_objects"] == ["radish", "spider"]
    # The 493-vs-467 discrepancy must stay visible until it is resolved.
    assert any("493" in q for q in ds.get("open_questions", []))


def test_screen_config_matches_lietzows_pixel_budget():
    """They pin the image *area* at 1280 merged cells (vlm/qwen.py). A smaller
    budget renders the note's text too small to read, which would look like a
    robust model rather than an illegible note."""
    from circuit_overlap.t_screen import ScreenConfig

    cfg = ScreenConfig()
    cell = 28 * 28
    assert cfg.pixels % cell == 0
    assert cfg.pixels // cell == 1280
    assert cfg.load_in_4bit is True, "D10: 4-bit, matching Lietzow's default"


def test_native_bf16_requires_ampere_not_emulation():
    """torch.cuda.is_bf16_supported() says True on a T4 (sm_75) because it
    counts emulation; emulated bf16 is much slower than fp16, and the first
    full screen ran at 8.4 s/generation because of it."""
    from circuit_overlap.t_screen import native_bf16

    assert native_bf16((7, 5)) is False      # T4, Turing
    assert native_bf16((8, 0)) is True       # A100, Ampere
    assert native_bf16((8, 9)) is True       # L4
    assert native_bf16((9, 0)) is True       # H100
    assert native_bf16(None) is False        # no CUDA
