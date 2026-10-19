"""Note-overlay generator for the T (typographic) condition.

Plan: docs/plan_v2.md section 3.1. For each *original-colour* image, paste a
post-it-style note containing a colour word, with controlled position, size,
font and background, and record the exact pixel mask **by construction** -- the
mask is read back from the alpha channel actually pasted, never re-estimated.

Note variants per object (D24):
    incongruent  word = c_cf   (the attack)
    neutral      non-colour word of matched length (primary baseline)
    congruent    word = c_o    (secondary baseline)
    none         no note       (secondary baseline)

Nothing here loads a model. Colour vocabulary, neutral words and layout presets
are data: see configs/notes.json.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "NoteSpec",
    "object_bbox_from_pair",
    "resolve_anchor",
    "NoteRender",
    "FONT_CANDIDATES",
    "render_note",
    "variant_texts",
    "matched_neutral_word",
]

# Searched in order; falls back to Pillow's built-in bitmap font. Kaggle images
# ship DejaVu with matplotlib, this dev machine has neither, and the font must
# not silently change the note's size between environments -- so the resolved
# font path is reported on every render (NoteRender.font_name) and belongs in
# the result JSON.
FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/opt/conda/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


@dataclass(frozen=True)
class NoteSpec:
    """A layout for one note. Relative units, so it transfers across image sizes.

    anchor:
        Centre of the note in relative image coordinates, (x, y) in [0, 1].
        (0.5, 0.5) sits on a centred object; corners sit beside it. Until we
        have per-object segmentation, "on object" vs "beside object" (plan 3.7)
        is approximated by centre vs corner -- recorded as an assumption in
        docs/STATE.md.
    rel_width:
        Note width as a fraction of image width.
    aspect:
        Note height / note width.
    rotation_deg:
        Counter-clockwise tilt. The mask stays exact: it is read from the
        rotated alpha channel.
    mode:
        How ``anchor`` is interpreted, given an object bounding box:

        ``"fixed"``   use ``anchor`` as-is (frame-relative).
        ``"on"``      centre the note on the object.
        ``"beside"``  place the note in the emptiest corner, clear of the object.

        ``"on"`` and ``"beside"`` fall back to ``anchor`` when no box is known.
        This replaces the centre-vs-corner approximation of D63: an object is
        not always in the middle of its frame.
    """

    rel_width: float = 0.34
    aspect: float = 0.62
    anchor: tuple[float, float] = (0.5, 0.5)
    rotation_deg: float = 0.0
    bg_color: tuple[int, int, int] = (255, 236, 120)   # post-it yellow
    text_color: tuple[int, int, int] = (20, 20, 20)
    border_color: tuple[int, int, int] | None = (210, 185, 60)
    pad_frac: float = 0.14      # inner margin, fraction of note height
    font_path: str | None = None
    mode: str = "fixed"
    name: str = "unnamed"

    def __post_init__(self) -> None:
        if not 0.0 < self.rel_width <= 1.0:
            raise ValueError(f"rel_width must be in (0, 1], got {self.rel_width}")
        if not 0.0 < self.aspect <= 10.0:
            raise ValueError(f"aspect must be in (0, 10], got {self.aspect}")
        x, y = self.anchor
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"anchor must lie in [0, 1]^2, got {self.anchor}")
        if not 0.0 <= self.pad_frac < 0.5:
            raise ValueError(f"pad_frac must be in [0, 0.5), got {self.pad_frac}")
        if self.mode not in ("fixed", "on", "beside"):
            raise ValueError(f"mode must be fixed/on/beside, got {self.mode!r}")


@dataclass(frozen=True)
class NoteRender:
    """Result of pasting one note.

    image:
        RGB ``PIL.Image`` of the same size as the input.
    mask:
        Boolean ``(H, W)`` array, True where note pixels were pasted. Read from
        the pasted alpha channel, so it is exact by construction (plan 3.1)
        rather than re-estimated from the image. Under rotation the antialiased
        border counts as note (alpha > 0), making the mask a tight superset.
        Feed it to ``circuit_overlap.tokens.mask_to_token_indices``.
    box:
        ``(x0, y0, x1, y1)`` pixel bounds of the pasted note, clipped to the image.
    text:
        The word rendered, for the record.
    font_name, font_size_px:
        Which font actually got used, and at what size.
    """

    image: Image.Image
    mask: np.ndarray
    box: tuple[int, int, int, int]
    text: str
    font_name: str
    font_size_px: int

    @property
    def area_frac(self) -> float:
        """Note area as a fraction of the image -- the size axis of plan 3.7."""
        return float(self.mask.mean())

    def object_occlusion(self, bbox: tuple[float, float, float, float] | None) -> float:
        """Fraction of the object's box the note covers.

        The number that matters for the T condition: a note that hides the
        object removes the colour evidence, so the model following the note is
        not *override* but absence of evidence. Keep it low, report it as the
        size covariate in the plan 3.7 spatial analysis, and treat high-occlusion
        layouts as a separate stratum rather than pooling them.
        """
        if bbox is None:
            return float("nan")
        h, w = self.mask.shape
        x0, y0, x1, y1 = bbox
        c0, r0 = int(x0 * w), int(y0 * h)
        c1, r1 = max(int(x1 * w), c0 + 1), max(int(y1 * h), r0 + 1)
        region = self.mask[r0:r1, c0:c1]
        return float(region.mean()) if region.size else float("nan")


def object_bbox_from_pair(
    original: Image.Image, counterfact: Image.Image, threshold: int = 16
) -> tuple[float, float, float, float] | None:
    """Approximate the object's box by diffing the two Visual-Counterfact images.

    The counterfactual image recolours the object and leaves the rest of the
    scene alone, so the pixels that differ are, to a good approximation, the
    object. That gives a free object localisation with no segmentation model --
    and it is what makes "on the object" and "beside the object" (plan 3.7)
    real rather than a guess about the object being centred.

    Returns ``(x0, y0, x1, y1)`` in **relative** coordinates, or ``None`` when
    the diff is empty (identical images) or covers almost the whole frame
    (a global recolour or a resize mismatch), in which cases the box carries no
    information and the caller should fall back to a fixed anchor.
    """
    a = np.asarray(original.convert("RGB"), dtype=np.int16)
    b = np.asarray(counterfact.convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return None
    diff = np.abs(a - b).max(axis=-1) >= threshold
    if not diff.any() or diff.mean() > 0.9:
        return None
    rows = np.flatnonzero(diff.any(axis=1))
    cols = np.flatnonzero(diff.any(axis=0))
    h, w = diff.shape
    return (
        float(cols[0] / w), float(rows[0] / h),
        float((cols[-1] + 1) / w), float((rows[-1] + 1) / h),
    )


def resolve_anchor(
    spec: NoteSpec,
    image_size: tuple[int, int],
    bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float]:
    """Anchor for ``spec``, given the object box (relative coords) if known."""
    if spec.mode == "fixed" or bbox is None:
        return spec.anchor

    x0, y0, x1, y1 = bbox
    if spec.mode == "on":
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    # "beside": the corner whose distance from the object box is greatest, with
    # the note nudged inside the frame by half its own size.
    w, h = image_size
    note_w = spec.rel_width
    note_h = spec.rel_width * spec.aspect * (w / h if h else 1.0)
    candidates = [
        (note_w / 2, note_h / 2), (1 - note_w / 2, note_h / 2),
        (note_w / 2, 1 - note_h / 2), (1 - note_w / 2, 1 - note_h / 2),
    ]

    def clearance(point):
        cx, cy = point
        dx = max(x0 - (cx + note_w / 2), (cx - note_w / 2) - x1, 0.0)
        dy = max(y0 - (cy + note_h / 2), (cy - note_h / 2) - y1, 0.0)
        return (dx * dx + dy * dy) ** 0.5

    return max(candidates, key=clearance)


def _load_font(size_px: int, font_path: str | None) -> tuple[ImageFont.ImageFont, str]:
    paths = (font_path,) if font_path else FONT_CANDIDATES
    for path in paths:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size=size_px), path
    if font_path:
        raise FileNotFoundError(f"font_path does not exist: {font_path}")
    try:
        return ImageFont.load_default(size=size_px), "pillow-default"
    except TypeError:  # Pillow < 10.1 cannot scale the built-in font
        return ImageFont.load_default(), "pillow-default-unscaled"


def _fit_font(
    text: str, box_w: int, box_h: int, font_path: str | None
) -> tuple[ImageFont.ImageFont, str, int]:
    """Largest font size whose rendered text fits the padded note box."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    best = None
    for size_px in range(max(6, box_h), 5, -1):
        font, name = _load_font(size_px, font_path)
        left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
        if (right - left) <= box_w and (bottom - top) <= box_h:
            return font, name, size_px
        best = (font, name, size_px)
    return best  # type: ignore[return-value]  # loop always runs at least once


def render_note(
    image: Image.Image,
    text: str,
    spec: NoteSpec,
    bbox: tuple[float, float, float, float] | None = None,
) -> NoteRender:
    """Paste one note onto ``image`` and return the image plus its exact mask.

    The input image is not modified. The note is drawn into an RGBA tile and
    composited, so the mask is the tile's alpha channel in image coordinates --
    correct even under rotation, and never an approximation of what was drawn.

    bbox:
        The object's box in relative coordinates (see ``object_bbox_from_pair``),
        used when ``spec.mode`` is ``"on"`` or ``"beside"``. Pass it whenever it
        is known: without it, placement falls back to ``spec.anchor``.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")

    base = image.convert("RGB")
    W, H = base.size
    if W == 0 or H == 0:
        raise ValueError("image has zero extent")

    note_w = max(1, int(round(spec.rel_width * W)))
    note_h = max(1, int(round(note_w * spec.aspect)))
    pad = int(round(spec.pad_frac * note_h))
    inner_w = max(1, note_w - 2 * pad)
    inner_h = max(1, note_h - 2 * pad)

    tile = Image.new("RGBA", (note_w, note_h), (*spec.bg_color, 255))
    draw = ImageDraw.Draw(tile)
    if spec.border_color is not None:
        draw.rectangle([0, 0, note_w - 1, note_h - 1], outline=(*spec.border_color, 255))

    font, font_name, font_size_px = _fit_font(text, inner_w, inner_h, spec.font_path)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tx = (note_w - (right - left)) / 2 - left
    ty = (note_h - (bottom - top)) / 2 - top
    draw.text((tx, ty), text, font=font, fill=(*spec.text_color, 255))

    if spec.rotation_deg % 360:
        tile = tile.rotate(spec.rotation_deg, expand=True, resample=Image.BICUBIC)

    tw, th = tile.size
    anchor = resolve_anchor(spec, (W, H), bbox)
    cx = int(round(anchor[0] * W))
    cy = int(round(anchor[1] * H))
    # Keep the note inside the frame: an off-image note would make the mask and
    # the attack itself depend on the image size.
    x0 = int(np.clip(cx - tw // 2, 0, max(0, W - tw)))
    y0 = int(np.clip(cy - th // 2, 0, max(0, H - th)))

    out = base.copy()
    alpha = tile.getchannel("A")
    out.paste(tile.convert("RGB"), (x0, y0), alpha)

    mask = np.zeros((H, W), dtype=bool)
    alpha_arr = np.asarray(alpha) > 0
    ah, aw = alpha_arr.shape
    vh, vw = min(ah, H - y0), min(aw, W - x0)
    mask[y0 : y0 + vh, x0 : x0 + vw] = alpha_arr[:vh, :vw]

    return NoteRender(
        image=out,
        mask=mask,
        box=(x0, y0, x0 + vw, y0 + vh),
        text=text,
        font_name=font_name,
        font_size_px=font_size_px,
    )


def variant_texts(
    original_color: str, counterfactual_color: str, neutral_word: str
) -> dict[str, str | None]:
    """The four note variants of D24, keyed by variant name.

    ``"none"`` maps to ``None``: that condition renders no note at all.
    """
    if original_color == counterfactual_color:
        raise ValueError("original and counterfactual colours must differ")
    return {
        "incongruent": counterfactual_color,
        "neutral": neutral_word,
        "congruent": original_color,
        "none": None,
    }


def matched_neutral_word(color_word: str, candidates: Iterable[str]) -> str:
    """Pick a non-colour word whose length matches ``color_word`` (plan 3.1).

    The neutral note controls for "a note exists" (risk 5), so it must differ
    from the attack only in meaning -- length is the crude proxy the plan asks
    for. Ties break alphabetically, so the choice is deterministic and does not
    depend on the order of the candidate list.
    """
    pool: Sequence[str] = [c for c in candidates]
    if not pool:
        raise ValueError("candidates is empty")
    target = len(color_word)
    return min(pool, key=lambda w: (abs(len(w) - target), w))


def with_name(spec: NoteSpec, name: str) -> NoteSpec:
    """Copy of ``spec`` with a new ``name`` (layouts are recorded per result)."""
    return replace(spec, name=name)
