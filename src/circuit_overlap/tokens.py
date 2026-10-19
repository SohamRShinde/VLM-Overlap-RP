"""Pixel mask -> image-token mapping.

Plan: docs/plan_v2.md section 3.7. The H3 analysis asks how much attention the
T-circuit heads put on *note* tokens versus *object* tokens, so the overlay mask
from ``circuit_overlap.notes`` has to be expressed in the model's image-token
grid.

Qwen2.5-VL: patches are 14x14 and are merged 2x2 before entering the language
model, so one image token covers a 28x28 pixel cell. Set
``min_pixels = max_pixels`` in the processor to pin the grid (plan 3.7);
otherwise the resize is input-dependent and the mapping below is wrong.

**The processor resizes the image**, so a mask drawn on the original must be
mapped onto the grid the model actually sees. Read ``image_grid_thw`` from the
processor output, convert it with ``merged_grid_from_grid_thw``, and use
``coverage_on_grid`` / ``mask_to_token_indices_on_grid``. The plain
``token_coverage`` only applies when the mask is already at the resized
resolution; it raises otherwise rather than mapping to the wrong tokens.

**[VERIFY]** two conventions against the real processor before trusting H3
numbers: that the merged grid is flattened row-major, and that no extra tokens
(vision start/end markers) shift the offset. Both are cheap to check with one
forward pass and are recorded as open questions in docs/STATE.md.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "QWEN25VL_PATCH_SIZE",
    "QWEN25VL_MERGE_SIZE",
    "token_grid_shape",
    "token_coverage",
    "mask_to_token_indices",
    "merged_grid_from_grid_thw",
    "coverage_on_grid",
    "mask_to_token_indices_on_grid",
]

QWEN25VL_PATCH_SIZE = 14
QWEN25VL_MERGE_SIZE = 2


def _check_grid(
    height: int, width: int, patch_size: int, merge_size: int
) -> tuple[int, int]:
    if patch_size <= 0 or merge_size <= 0:
        raise ValueError("patch_size and merge_size must be positive")
    cell = patch_size * merge_size
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if height % cell or width % cell:
        raise ValueError(
            f"image {height}x{width} is not divisible by patch_size*merge_size="
            f"{cell}; resize with min_pixels = max_pixels so the grid is fixed"
        )
    return height // cell, width // cell


def token_grid_shape(
    height: int,
    width: int,
    patch_size: int = QWEN25VL_PATCH_SIZE,
    merge_size: int = QWEN25VL_MERGE_SIZE,
) -> tuple[int, int]:
    """``(rows, cols)`` of image tokens after the 2x2 merge."""
    return _check_grid(height, width, patch_size, merge_size)


def token_coverage(
    mask,
    patch_size: int = QWEN25VL_PATCH_SIZE,
    merge_size: int = QWEN25VL_MERGE_SIZE,
) -> np.ndarray:
    """Fraction of each token's pixel cell that the mask covers.

    Returns a ``(rows, cols)`` float array in [0, 1]. Partial coverage is the
    normal case at the note's edge, which is why the threshold in
    ``mask_to_token_indices`` is explicit rather than hidden.
    """
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-d (H, W), got shape {arr.shape}")
    h, w = arr.shape
    rows, cols = _check_grid(h, w, patch_size, merge_size)
    cell = patch_size * merge_size
    return arr.reshape(rows, cell, cols, cell).mean(axis=(1, 3))


def mask_to_token_indices(
    mask,
    patch_size: int = QWEN25VL_PATCH_SIZE,
    merge_size: int = QWEN25VL_MERGE_SIZE,
    threshold: float = 0.5,
    return_grid: bool = False,
):
    """Flat indices of the image tokens the mask covers.

    threshold:
        Minimum covered fraction of a token's cell for it to count as a note
        token. 0.5 (default) is the majority rule; pass a small positive value
        for "any overlap". ``threshold=0`` would select every token, so it is
        rejected.
    return_grid:
        Also return the boolean ``(rows, cols)`` token grid, for plotting.

    Indices are row-major over the merged grid and are **relative to the first
    image token**, not to the full input sequence: add the offset of the image
    block in the prompt before indexing an attention matrix.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    coverage = token_coverage(mask, patch_size=patch_size, merge_size=merge_size)
    grid = coverage >= threshold
    indices = np.flatnonzero(grid.reshape(-1))
    if return_grid:
        return indices, grid
    return indices


def merged_grid_from_grid_thw(grid_thw, merge_size: int = QWEN25VL_MERGE_SIZE):
    """``image_grid_thw`` -> ``(rows, cols)`` of image tokens after the merge.

    The processor reports ``(t, h, w)`` in **patches**, for example
    ``[[1, 62, 86]]`` for a 440x612 input at the pinned pixel budget. After the
    2x2 merge that is 31x43 = 1333 image tokens.
    """
    arr = np.asarray(grid_thw).reshape(-1, 3)
    if arr.shape[0] != 1:
        raise ValueError(f"expected one image, got grid_thw with {arr.shape[0]} rows")
    _, h, w = (int(v) for v in arr[0])
    if h % merge_size or w % merge_size:
        raise ValueError(
            f"patch grid {h}x{w} is not divisible by merge_size={merge_size}"
        )
    return h // merge_size, w // merge_size


def coverage_on_grid(mask, rows: int, cols: int) -> np.ndarray:
    """Fraction of each token cell covered, for a mask at **any** resolution.

    The mask is area-averaged down to ``(rows, cols)`` -- the grid the model
    actually sees -- so it does not matter that the overlay was drawn before
    the processor's resize. Use this, not ``token_coverage``, in any real run.
    """
    from PIL import Image

    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-d (H, W), got shape {arr.shape}")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    # BOX resampling averages over each destination cell: exactly the coverage
    # fraction, up to the 1/255 quantisation of the intermediate image.
    small = img.resize((cols, rows), Image.BOX)
    return np.asarray(small, dtype=np.float64) / 255.0


def mask_to_token_indices_on_grid(
    mask, rows: int, cols: int, threshold: float = 0.5, return_grid: bool = False
):
    """``mask_to_token_indices`` for a mask that is not at the model's resolution.

    Indices are row-major over the merged grid and relative to the first image
    token: add the offset of the image block in the prompt before indexing an
    attention matrix.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    coverage = coverage_on_grid(mask, rows, cols)
    grid = coverage >= threshold
    indices = np.flatnonzero(grid.reshape(-1))
    if return_grid:
        return indices, grid
    return indices
