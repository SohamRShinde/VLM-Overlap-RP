"""CPU tests for mask -> image-token mapping (plan 3.7, Qwen 2x2 merge)."""

import numpy as np
import pytest

from circuit_overlap.notes import NoteSpec, render_note
from circuit_overlap.tokens import (
    mask_to_token_indices,
    token_coverage,
    token_grid_shape,
)
from PIL import Image


def test_grid_shape_accounts_for_the_2x2_merge():
    # 224 / 14 = 16 patches per side, merged 2x2 -> 8 tokens per side.
    assert token_grid_shape(224, 224) == (8, 8)
    assert token_grid_shape(280, 168) == (10, 6)


def test_grid_shape_rejects_sizes_that_do_not_divide():
    with pytest.raises(ValueError, match="min_pixels"):
        token_grid_shape(225, 224)
    with pytest.raises(ValueError):
        token_grid_shape(0, 224)


def test_coverage_is_the_fraction_of_each_28x28_cell():
    mask = np.zeros((56, 56), dtype=bool)
    mask[0:28, 0:28] = True       # token (0,0) fully covered
    mask[28:42, 28:56] = True     # token (1,1) half covered
    cov = token_coverage(mask)
    assert cov.shape == (2, 2)
    assert cov[0, 0] == pytest.approx(1.0)
    assert cov[1, 1] == pytest.approx(0.5)
    assert cov[0, 1] == 0.0 and cov[1, 0] == 0.0


def test_indices_are_row_major_over_the_merged_grid():
    mask = np.zeros((56, 56), dtype=bool)
    mask[28:56, 28:56] = True  # bottom-right token of a 2x2 grid -> flat index 3
    np.testing.assert_array_equal(mask_to_token_indices(mask), [3])


def test_threshold_controls_partial_cells():
    mask = np.zeros((28, 56), dtype=bool)
    mask[:, 0:28] = True       # token 0 fully covered
    mask[0:7, 28:56] = True    # token 1 quarter covered
    np.testing.assert_array_equal(mask_to_token_indices(mask, threshold=0.5), [0])
    np.testing.assert_array_equal(mask_to_token_indices(mask, threshold=0.2), [0, 1])


def test_threshold_zero_is_rejected_because_it_would_select_everything():
    with pytest.raises(ValueError):
        mask_to_token_indices(np.zeros((28, 28), dtype=bool), threshold=0.0)


def test_empty_and_full_masks():
    assert mask_to_token_indices(np.zeros((56, 56), bool)).size == 0
    np.testing.assert_array_equal(
        mask_to_token_indices(np.ones((56, 56), bool)), [0, 1, 2, 3]
    )


def test_return_grid_matches_indices():
    mask = np.zeros((56, 56), dtype=bool)
    mask[0:28, 28:56] = True
    idx, grid = mask_to_token_indices(mask, return_grid=True)
    assert grid.shape == (2, 2)
    np.testing.assert_array_equal(np.flatnonzero(grid.reshape(-1)), idx)


def test_mask_must_be_two_dimensional():
    with pytest.raises(ValueError):
        token_coverage(np.zeros((56, 56, 3), dtype=bool))


def test_end_to_end_a_rendered_note_lands_on_plausible_tokens():
    """The real path: render a note, map its mask, sanity-check the token count."""
    img = Image.new("RGB", (224, 224), (40, 90, 40))
    r = render_note(img, "blue", NoteSpec(rel_width=0.5, anchor=(0.5, 0.5)))
    idx, grid = mask_to_token_indices(r.mask, return_grid=True)
    assert idx.size > 0
    assert idx.size < grid.size  # a note covering every token would be useless
    # Centred note -> selected tokens are centred too.
    rows, cols = np.nonzero(grid)
    assert 1 <= rows.mean() <= 6 and 1 <= cols.mean() <= 6
