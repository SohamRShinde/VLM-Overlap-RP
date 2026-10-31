"""CPU tests for the T-circuit discovery helpers (D71). No model, no torch."""

import gzip
import json
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from circuit_overlap.notes import NoteSpec
from circuit_overlap.t_patching import (
    LAYOUT,
    T_ROLES,
    finite_records,
    has_nonfinite,
    is_t_conflicting,
    load_patching_records,
    render_t_pair,
    screen_cross_check,
    screen_object_index,
    sweep_order,
)

COLOR_IDS = {"red": [1, 2], "blue": [3, 4], "white": [5, 6], "brown": [7, 8]}


def pair(box=(60, 70, 150, 160)):
    orig = Image.new("RGB", (224, 224), (240, 240, 235))
    cf = Image.new("RGB", (224, 224), (240, 240, 235))
    ImageDraw.Draw(orig).ellipse(box, fill=(190, 40, 40))
    ImageDraw.Draw(cf).ellipse(box, fill=(40, 60, 190))
    return orig, cf


# --- the image pair --------------------------------------------------------


def test_pair_differs_only_in_the_note_word():
    orig, cf = pair()
    out = render_t_pair(orig, cf, "blue", ["table", "paper", "metal"],
                        NoteSpec(rel_width=0.2, mode="on", name=LAYOUT))
    assert out["same_note_footprint"] is True
    a, b = np.asarray(out["neutral"]), np.asarray(out["incongruent"])
    assert a.shape == b.shape and (a != b).any()
    # Outside the note the two images are pixel-identical.
    differ = (a != b).any(axis=-1)
    assert out["bbox"] is not None
    assert differ.mean() < 0.05


def test_pair_uses_the_length_matched_neutral_word():
    orig, cf = pair()
    # "blue" has 4 letters; "cup" and "table" are both one off, and the
    # alphabetical tie-break makes it "cup" -- the screen's exact choice.
    out = render_t_pair(orig, cf, "blue", ["cup", "table", "window"], NoteSpec())
    assert out["neutral_word"] == "cup"


def test_pair_reports_occlusion_from_the_recovered_box():
    orig, cf = pair()
    out = render_t_pair(orig, cf, "blue", ["table"], NoteSpec(rel_width=0.2, mode="on"))
    assert 0.0 < out["object_occlusion"] < 0.3
    assert 0.0 < out["note_area_frac"] < 0.1


# --- selection -------------------------------------------------------------


ROW = {"i": 7, "name": "strawberry", "correct": ["red"], "attack": "blue"}


def test_conflicting_needs_right_on_neutral_and_fooled_on_incongruent():
    assert is_t_conflicting(ROW, top1_neutral=1, top1_incongruent=3, color_ids=COLOR_IDS)
    assert is_t_conflicting(ROW, top1_neutral=2, top1_incongruent=4, color_ids=COLOR_IDS)
    # not fooled: the attacked run still answers the real colour
    assert not is_t_conflicting(ROW, top1_neutral=1, top1_incongruent=1, color_ids=COLOR_IDS)
    # wrong without the attack
    assert not is_t_conflicting(ROW, top1_neutral=5, top1_incongruent=3, color_ids=COLOR_IDS)
    # fooled into a third colour is not fooled by the note
    assert not is_t_conflicting(ROW, top1_neutral=1, top1_incongruent=7, color_ids=COLOR_IDS)


def test_roles_are_the_mirror_of_p():
    """P's competing run answers the REAL colour; T's answers the NOTE's. A
    swap here would silently invert which heads count as promoting."""
    from circuit_overlap.replication import is_correctly_conflicting

    # P passes with the competing (prior) run on a real colour...
    assert is_correctly_conflicting(ROW, top1_visual=3, top1_prior=1, color_ids=COLOR_IDS)
    # ...T passes with the competing (incongruent) run on the attack colour.
    assert is_t_conflicting(ROW, top1_neutral=1, top1_incongruent=3, color_ids=COLOR_IDS)
    assert "incongruent" in T_ROLES["source"] and "neutral" in T_ROLES["target"]


def test_multi_colour_objects_accept_any_real_colour():
    row = {"i": 1, "name": "cow", "correct": ["brown", "white"], "attack": "blue"}
    assert is_t_conflicting(row, top1_neutral=5, top1_incongruent=3, color_ids=COLOR_IDS)
    assert is_t_conflicting(row, top1_neutral=8, top1_incongruent=4, color_ids=COLOR_IDS)


def test_ambiguous_colours_are_excluded():
    assert not is_t_conflicting(ROW, 1, 3, COLOR_IDS, ambiguous={"blue"})


# --- sweep order -----------------------------------------------------------


def test_shared_objects_come_first_then_a_seeded_shuffle():
    selected = list(range(20))
    order = sweep_order(selected, p_ids=[3, 15, 99], seed=0)
    assert order[:2] == [3, 15]                    # shared with P, first
    assert sorted(order) == selected               # every object exactly once
    assert order[2:] != sorted(order[2:])          # the rest is shuffled
    assert order == sweep_order(selected, [3, 15, 99], seed=0)
    assert order != sweep_order(selected, [3, 15, 99], seed=1)


# --- numerics --------------------------------------------------------------


def record(i=0, bad=False):
    grid = [[0.1, 0.2], [0.3, 0.4]]
    r = {"i": i, "status": "patched",
         "prior_token_logit_prior_run": 1.0, "visual_token_logit_prior_run": 0.0,
         "prior_token_logit_visual_run": 0.0, "visual_token_logit_visual_run": 1.0,
         "p2v_logit_prior": grid, "p2v_logit_visual": grid,
         "v2p_logit_prior": grid, "v2p_logit_visual": grid}
    if bad:
        r["p2v_logit_prior"] = [[0.1, float("nan")], [0.3, 0.4]]
    return r


def test_nonfinite_records_are_detected_and_dropped():
    assert has_nonfinite(record()) is False
    assert has_nonfinite(record(bad=True)) is True
    assert has_nonfinite({**record(), "visual_token_logit_visual_run": float("inf")})
    kept, dropped = finite_records([record(0), record(1, bad=True), record(2)])
    assert [r["i"] for r in kept] == [0, 2] and dropped == [1]


# --- loading records -------------------------------------------------------


def test_records_load_from_envelope_jsonl_and_gzip(tmp_path):
    recs = [record(0), record(1), {"i": 2, "status": "not_conflicting"}]
    env = tmp_path / "p.json"
    env.write_text(json.dumps({"result": {"records": recs}}))
    plain = tmp_path / "t.jsonl"
    plain.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    gz = tmp_path / "t.jsonl.gz"
    with gzip.open(gz, "wt") as f:
        f.write("\n".join(json.dumps(r) for r in recs) + "\n")
    for path in (env, plain, gz):
        loaded = load_patching_records(path)
        assert [r["i"] for r in loaded] == [0, 1], path.name   # status filter applied


# --- cross-check against the behavioural screen ----------------------------


def test_screen_object_index():
    assert screen_object_index("acorn_squash_0030") == 30
    assert screen_object_index("fire_truck_0412") == 412
    with pytest.raises(ValueError):
        screen_object_index("acorn_squash")


def test_screen_cross_check_counts():
    out = screen_cross_check([30, 31, 40], ["acorn_squash_0030", "b_0031", "c_0099"])
    assert out["n_both"] == 2 and out["n_selected_only"] == 1 and out["n_screen_only"] == 1
    assert math.isclose(out["share_of_selected_also_in_screen"], 2 / 3)


def test_layout_exists_in_the_shipped_config():
    from pathlib import Path

    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "notes.json").read_text())
    assert LAYOUT in {l["name"] for l in cfg["layouts"]}
