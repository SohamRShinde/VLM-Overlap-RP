"""Object-relative note placement (plan 3.7, D63).

The thing being guarded: a note that covers the object removes the colour
evidence, so "the model followed the note" stops being override and becomes
absence of evidence.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from circuit_overlap.notes import (
    NoteSpec,
    object_bbox_from_pair,
    render_note,
    resolve_anchor,
)


def pair(box=(60, 70, 150, 160), size=(224, 224)):
    """An original/counterfactual pair differing only on the object."""
    orig = Image.new("RGB", size, (240, 240, 235))
    cf = Image.new("RGB", size, (240, 240, 235))
    ImageDraw.Draw(orig).ellipse(box, fill=(190, 40, 40))
    ImageDraw.Draw(cf).ellipse(box, fill=(40, 60, 190))
    return orig, cf


# --- recovering the object box from the image pair ------------------------


def test_bbox_tracks_the_recoloured_region():
    orig, cf = pair(box=(60, 70, 150, 160))
    x0, y0, x1, y1 = object_bbox_from_pair(orig, cf)
    assert x0 == pytest.approx(60 / 224, abs=0.02)
    assert y0 == pytest.approx(70 / 224, abs=0.02)
    assert x1 == pytest.approx(151 / 224, abs=0.02)
    assert y1 == pytest.approx(161 / 224, abs=0.02)


def test_bbox_is_none_when_the_diff_carries_no_information():
    orig, _ = pair()
    assert object_bbox_from_pair(orig, orig) is None          # identical
    whole = Image.new("RGB", (224, 224), (10, 200, 10))
    assert object_bbox_from_pair(orig, whole) is None         # global recolour
    assert object_bbox_from_pair(orig, orig.resize((112, 112))) is None  # shape clash


def test_bbox_ignores_sub_threshold_noise():
    orig, cf = pair()
    noisy = Image.fromarray(
        np.clip(np.asarray(cf, dtype=np.int16) + 3, 0, 255).astype(np.uint8)
    )
    a = object_bbox_from_pair(orig, cf)
    b = object_bbox_from_pair(orig, noisy, threshold=16)
    assert a == pytest.approx(b, abs=0.02)


# --- placement modes -------------------------------------------------------


def test_on_mode_centres_the_note_on_the_object_not_the_frame():
    bbox = (0.6, 0.6, 0.9, 0.9)                     # object off-centre
    spec = NoteSpec(rel_width=0.2, mode="on")
    assert resolve_anchor(spec, (224, 224), bbox) == pytest.approx((0.75, 0.75))


def test_beside_mode_keeps_clear_of_the_object():
    bbox = (0.55, 0.55, 0.95, 0.95)                 # object bottom-right
    spec = NoteSpec(rel_width=0.2, mode="beside")
    ax, ay = resolve_anchor(spec, (224, 224), bbox)
    assert ax < 0.5 and ay < 0.5                    # note goes top-left


def test_modes_fall_back_to_the_fixed_anchor_without_a_box():
    for mode in ("on", "beside", "fixed"):
        spec = NoteSpec(rel_width=0.2, mode=mode, anchor=(0.3, 0.7))
        assert resolve_anchor(spec, (224, 224), None) == (0.3, 0.7)


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="fixed/on/beside"):
        NoteSpec(mode="middle")


# --- occlusion, the number that matters ------------------------------------


def test_occlusion_falls_as_the_note_shrinks():
    orig, cf = pair()
    bbox = object_bbox_from_pair(orig, cf)
    big = render_note(orig, "blue", NoteSpec(rel_width=0.44, mode="on"), bbox=bbox)
    small = render_note(orig, "blue", NoteSpec(rel_width=0.20, mode="on"), bbox=bbox)
    assert big.object_occlusion(bbox) > 0.5         # the layout the author caught
    assert small.object_occlusion(bbox) < 0.25
    assert small.area_frac < big.area_frac


def test_beside_placement_leaves_the_object_visible():
    orig, cf = pair(box=(120, 120, 210, 210))
    bbox = object_bbox_from_pair(orig, cf)
    r = render_note(orig, "blue", NoteSpec(rel_width=0.26, mode="beside"), bbox=bbox)
    assert r.object_occlusion(bbox) == pytest.approx(0.0, abs=0.01)


def test_occlusion_is_nan_without_a_box():
    orig, _ = pair()
    r = render_note(orig, "blue", NoteSpec(rel_width=0.2))
    assert np.isnan(r.object_occlusion(None))


def test_shipped_layouts_keep_occlusion_moderate():
    """Whatever configs/notes.json ships must not hide the object."""
    import json
    from pathlib import Path

    from circuit_overlap.t_screen import load_notes_config

    cfg = load_notes_config(Path(__file__).resolve().parents[1] / "configs" / "notes.json")
    orig, cf = pair()
    bbox = object_bbox_from_pair(orig, cf)
    for spec in cfg["specs"]:
        occ = render_note(orig, "blue", spec, bbox=bbox).object_occlusion(bbox)
        assert occ < 0.4, f"{spec.name} occludes {occ:.0%} of the object"
