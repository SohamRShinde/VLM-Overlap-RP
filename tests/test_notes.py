"""CPU tests for the note-overlay generator (plan 3.1). No model, no dataset."""

import numpy as np
import pytest
from PIL import Image

from circuit_overlap.notes import (
    NoteSpec,
    matched_neutral_word,
    render_note,
    variant_texts,
)


def blank(w=224, h=224, color=(30, 120, 60)):
    return Image.new("RGB", (w, h), color)


# --- spec validation -------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rel_width": 0.0},
        {"rel_width": 1.4},
        {"anchor": (1.2, 0.5)},
        {"anchor": (0.5, -0.1)},
        {"pad_frac": 0.7},
        {"aspect": 0.0},
    ],
)
def test_notespec_rejects_out_of_range_layouts(kwargs):
    with pytest.raises(ValueError):
        NoteSpec(**kwargs)


# --- mask correctness ------------------------------------------------------


def test_mask_is_exact_where_the_image_changed():
    img = blank()
    r = render_note(img, "blue", NoteSpec(rel_width=0.4))
    before = np.asarray(img)
    after = np.asarray(r.image)
    changed = (before != after).any(axis=-1)
    # Every changed pixel is inside the mask. The converse does not hold: note
    # pixels can coincidentally equal the background colour.
    assert np.all(r.mask[changed])
    assert changed.sum() > 0


def test_mask_shape_dtype_and_box_agree():
    r = render_note(blank(224, 224), "red", NoteSpec(rel_width=0.3, anchor=(0.5, 0.5)))
    assert r.mask.shape == (224, 224)
    assert r.mask.dtype == np.bool_
    x0, y0, x1, y1 = r.box
    assert r.mask[y0:y1, x0:x1].any()
    outside = r.mask.copy()
    outside[y0:y1, x0:x1] = False
    assert not outside.any()  # nothing marked beyond the box


def test_note_area_scales_with_rel_width():
    small = render_note(blank(), "red", NoteSpec(rel_width=0.2))
    large = render_note(blank(), "red", NoteSpec(rel_width=0.5))
    assert large.area_frac > 2 * small.area_frac
    # rel_width 0.5, aspect 0.62 -> about 0.5 * 0.31 of the frame
    assert large.area_frac == pytest.approx(0.5 * 0.5 * 0.62, rel=0.1)


def test_anchor_moves_the_note():
    centre = render_note(blank(), "red", NoteSpec(rel_width=0.2, anchor=(0.5, 0.5)))
    corner = render_note(blank(), "red", NoteSpec(rel_width=0.2, anchor=(0.1, 0.1)))
    assert corner.box[0] < centre.box[0] and corner.box[1] < centre.box[1]
    assert corner.area_frac == pytest.approx(centre.area_frac, rel=0.02)


def test_note_stays_inside_the_frame_for_extreme_anchors():
    for anchor in [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]:
        r = render_note(blank(), "red", NoteSpec(rel_width=0.4, anchor=anchor))
        x0, y0, x1, y1 = r.box
        assert 0 <= x0 < x1 <= 224 and 0 <= y0 < y1 <= 224
        # Clipping would shrink the attack; the note must keep its full area.
        assert r.area_frac == pytest.approx(0.4 * 0.4 * 0.62, rel=0.1)


def test_rotation_keeps_a_nonempty_mask_and_does_not_leave_the_frame():
    r = render_note(blank(), "blue", NoteSpec(rel_width=0.3, rotation_deg=25.0))
    assert r.mask.any()
    assert r.mask.sum() < 224 * 224


def test_render_does_not_mutate_the_input_image():
    img = blank()
    before = np.asarray(img).copy()
    render_note(img, "blue", NoteSpec())
    np.testing.assert_array_equal(np.asarray(img), before)


def test_render_is_deterministic():
    a = render_note(blank(), "blue", NoteSpec(rel_width=0.33, rotation_deg=7.0))
    b = render_note(blank(), "blue", NoteSpec(rel_width=0.33, rotation_deg=7.0))
    np.testing.assert_array_equal(np.asarray(a.image), np.asarray(b.image))
    np.testing.assert_array_equal(a.mask, b.mask)


def test_longer_text_shrinks_the_font_but_not_the_note():
    short = render_note(blank(), "red", NoteSpec(rel_width=0.3))
    long = render_note(blank(), "chartreuse", NoteSpec(rel_width=0.3))
    assert long.font_size_px <= short.font_size_px
    assert long.area_frac == pytest.approx(short.area_frac, rel=0.02)


def test_render_rejects_empty_text():
    with pytest.raises(ValueError):
        render_note(blank(), "", NoteSpec())


def test_render_accepts_non_rgb_input():
    r = render_note(Image.new("L", (112, 112), 128), "red", NoteSpec(rel_width=0.4))
    assert r.image.mode == "RGB" and r.mask.shape == (112, 112)


# --- variants and neutral words (D24, D26) ---------------------------------


def test_variant_texts_covers_the_four_conditions():
    v = variant_texts("red", "blue", "table")
    assert v == {
        "incongruent": "blue",
        "neutral": "table",
        "congruent": "red",
        "none": None,
    }


def test_variant_texts_rejects_equal_colors():
    with pytest.raises(ValueError):
        variant_texts("red", "red", "table")


def test_matched_neutral_word_matches_length_and_breaks_ties_alphabetically():
    # "purple" has 6 letters; "letter" and "window" both match, "letter" wins.
    assert matched_neutral_word("purple", ["table", "window", "letter"]) == "letter"
    assert matched_neutral_word("red", ["table", "cup", "window"]) == "cup"


def test_matched_neutral_word_is_order_independent():
    pool = ["window", "letter", "table"]
    assert matched_neutral_word("purple", pool) == matched_neutral_word(
        "purple", list(reversed(pool))
    )


def test_matched_neutral_word_rejects_empty_pool():
    with pytest.raises(ValueError):
        matched_neutral_word("red", [])
