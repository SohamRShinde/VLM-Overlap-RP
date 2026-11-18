"""Paper figures, drawn only from committed files in results/.

    .venv/bin/python scripts/make_figures.py [out_dir]     # default: figures/

Writes one PDF (for the paper) and one PNG (for a quick look) per figure.
Every plotted number is read from a results file; nothing is computed here
that the analysis scripts did not already report, except rescaling PC1
scores to units of their own standard deviation (the classification cut is
at +/-2 of those units, D70/Lietzow).

Style (17 Nov 2026 restyle; the data drawn is unchanged): boxed axes with a
light grid, legends inside each figure, and one saturated colour per role,
used the same way in every figure:

    P / H_P / P-only      blue      T / H_T / T-only   orange
    shared (P and T)      green     V                  magenta
    random head sets      grey      split-half ceiling gold band

The four series colours pass the dataviz skill's validator with all pairs
compared (colour-blind dE >= 8.7, normal-vision dE >= 20.8). Orange is just
under 3:1 on white, so every mark also carries a text label or legend entry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODEL = "qwen25vl3b"

# Roles, not raw hex, in the plotting code below.
INK, INK2, MUTED = "#1a1a1a", "#3d3d3d", "#6b6b6b"
GRID, FRAME = "#e4e4e4", "#a8a8a8"
RANDOM, RANDOM_DARK = "#c4c4c4", "#5a5a5a"
P_COL, T_COL, SHARED_COL, V_COL = "#1e6fd9", "#ec5f0d", "#12a150", "#c2187a"
CEIL_COL = "#f2b705"
SET_COL = {"P": P_COL, "T": T_COL, "shared": SHARED_COL, "P_only": P_COL, "T_only": T_COL}
MODELS = [("Qwen2.5-VL-3B", MODEL, ""), ("PaliGemma2-3B", "paligemma2_3b", "pg_")]   # title, file suffix, prefix


def short_labels(sets: dict) -> dict:
    """Tick labels with each set's size, read from the analysis file's `sets`."""
    n = {k: len(v) for k, v in sets.items()}
    return {"P": f"$\\mathcal{{H}}_P$\n(all {n['P']})", "T": f"$\\mathcal{{H}}_T$\n(all {n['T']})",
            "shared": f"shared\n({n['shared']})", "P_only": f"P-only\n({n['P_only']})",
            "T_only": f"T-only\n({n['T_only']})"}


COLUMN_W, TEXT_W = 3.3, 7.0   # AAAI two-column widths, inches


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.titlepad": 5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.8, "axes.edgecolor": FRAME, "axes.linewidth": 0.7,
        "axes.labelcolor": INK, "xtick.color": FRAME, "ytick.color": FRAME,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "text.color": INK,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": True, "axes.spines.right": True, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": True, "legend.fancybox": True, "legend.edgecolor": "#cfcfcf",
        "legend.framealpha": 0.95, "legend.borderpad": 0.45, "legend.handlelength": 1.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "lines.linewidth": 1.4, "hatch.linewidth": 0.6,
    })


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())["result"]


def save(fig, out: Path, name: str) -> list[Path]:
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths


def head_name(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def bottom_legend(fig, handles, ncol, y=0.0) -> None:
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               columnspacing=1.4, handletextpad=0.5)


def null_edge(r: dict) -> float:
    """The random-set percentile a test is read against: 95th for 'greater', 5th for 'less'."""
    return r["null_p95"] if r.get("direction", "greater") == "greater" else r["null_p5"]


def bar_pair(ax, x, observed, ci, colour, null=None, w=0.36, hatch=None, label=None, fmt="{:.2f}",
             label_pad=0.0):
    """One observed bar (with its CI) and, beside it, a grey bar for the random sets.

    null = (mean, edge): grey bar to the random-set mean, whisker to the 95th/5th percentile.
    """
    xo = x - w / 2 if null is not None else x
    ax.bar(xo, observed, w, color="white" if hatch else colour, edgecolor=colour, hatch=hatch,
           lw=1.0 if hatch else 0, zorder=3)
    if ci is not None:
        ax.errorbar(xo, observed, yerr=[[observed - ci[0]], [ci[1] - observed]], fmt="none",
                    ecolor=INK, elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=4)
    if null is not None:
        mean, edge = null
        ax.bar(x + w / 2, mean, w, color=RANDOM, lw=0, zorder=3)
        ax.errorbar(x + w / 2, mean, yerr=[[max(mean - edge, 0)], [max(edge - mean, 0)]], fmt="none",
                    ecolor=RANDOM_DARK, elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=4)
    if label is not None:
        top = max(observed, ci[1] if ci is not None else observed)
        bottom = min(observed, ci[0] if ci is not None else observed)
        neg = observed < 0
        ax.annotate(label, (xo, bottom if neg else top), xytext=(0, -3 - label_pad if neg else 3 + label_pad),
                    textcoords="offset points", ha="center", va="top" if neg else "bottom",
                    fontsize=6.5, color=INK, fontweight="bold")


# --------------------------------------------------------------------------- 0
RED, BLUE = "#d6453d", "#2b8ad6"             # the example object's colours (image content, not roles)
TILE = "#ffffff"
COND_COL = {"P": P_COL, "T": T_COL, "V": V_COL}
INPUTS = ROOT / "figures" / "inputs"         # real Visual-Counterfact images, from scripts/fig1_inputs.py


def tile(ax, x, y, s, image, edge) -> None:
    """One input image from the dataset (figures/inputs/<image>.png), framed in the condition's colour."""
    from matplotlib.patches import FancyBboxPatch
    img = plt.imread(INPUTS / f"{image}.png")
    frame = FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=0.04",
                           fc=TILE, ec=edge, lw=0.9, zorder=3)
    ax.imshow(img, extent=(x, x + s, y, y + s), zorder=2, interpolation="lanczos")
    clip = FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=0.04", transform=ax.transData)
    ax.images[-1].set_clip_path(clip)
    frame.set_facecolor("none")
    ax.add_patch(frame)


def fig_design(out: Path) -> list[Path]:
    """Figure 1, a schematic of the three conflicts (no result is drawn).

    The tiles are real Visual-Counterfact inputs (the strawberry, row 422) and, for T, the
    experiment's own notes on it; see scripts/fig1_inputs.py.

    Each conflict is a pair of runs that differ in one factor; patching swaps one head's
    last-token output between them, in both directions.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    W, H, s = TEXT_W, 1.92, 0.62
    prior_q, visual_q = "What color is a\nstrawberry usually?", "What color is this\nstrawberry here?"
    blocks = [
        ("P", "prompted prior", "prompt varies",
         [("counterfact", prior_q, "red", RED, "memory"), ("counterfact", visual_q, "blue", BLUE, "image")]),
        ("T", "typographic", "note's word varies",
         [("note_attack", visual_q, "blue", BLUE, "note"), ("note_neutral", visual_q, "red", RED, "image")]),
        ("V", "image-varying", "pixels vary",
         [("original", visual_q, "red", RED, "image"), ("counterfact", visual_q, "blue", BLUE, "image")]),
    ]
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    bw, gap, head_h = 2.27, 0.08, 0.27
    for i, (code, name, varies, runs) in enumerate(blocks):
        col = COND_COL[code]
        x0 = 0.02 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x0, 0.03), bw, H - 0.06, boxstyle="round,pad=0,rounding_size=0.07",
                                    fc=to_rgba(col, 0.045), ec=col, lw=1.1, zorder=0))
        ax.add_patch(FancyBboxPatch((x0, H - 0.03 - head_h), bw, head_h,
                                    boxstyle="round,pad=0,rounding_size=0.07", fc=col, ec=col, lw=1.1, zorder=0.5))
        ax.add_patch(plt.Rectangle((x0 + 0.004, H - 0.03 - head_h), bw - 0.008, 0.08, fc=col, lw=0, zorder=0.6))
        ax.text(x0 + 0.12, H - 0.03 - head_h / 2, f"{code}   {name}", ha="left", va="center",
                fontsize=8, fontweight="bold", color="white")
        ax.text(x0 + bw - 0.12, H - 0.03 - head_h / 2, varies, ha="right", va="center", fontsize=6.3,
                color="white", style="italic")
        xs = [x0 + 0.24, x0 + bw - 0.24 - s]
        for x, role in zip(xs, ("source run", "target run")):
            ax.text(x + s / 2, H - 0.45, role, ha="center", va="center", fontsize=6.3, color=INK,
                    fontweight="bold")
        for x, (image, prompt, answer, ans_col, basis) in zip(xs, runs):
            tile(ax, x, 0.68, s, image, col)
            ax.text(x + s / 2, 0.5, prompt, ha="center", va="center", fontsize=5.6,
                    color=INK2, linespacing=1.15, style="italic")
            ax.text(x + s / 2, 0.24, f"$\\rightarrow$ {answer}", ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color=ans_col)
            ax.text(x + s / 2, 0.1, f"(from {basis})", ha="center", va="center", fontsize=6, color=MUTED)
        ax.add_patch(FancyArrowPatch((xs[0] + s + 0.05, 0.99), (xs[1] - 0.05, 0.99),
                                     arrowstyle="<|-|>", mutation_scale=8, color=col, lw=1.2))
        ax.text((xs[0] + s + xs[1]) / 2, 1.09, "patch\none head", ha="center", va="bottom", fontsize=5.8,
                color=col, fontweight="bold", linespacing=1.1)
    return save(fig, out, "fig_design")


# --------------------------------------------------------------------------- 1
def _zscores(r: dict) -> np.ndarray:
    return (np.asarray(r["pc1_scores"]) - r["pc1_mean"]) / r["pc1_std"]


def _place_labels(ax, labels, points) -> None:
    """Direct labels that avoid each other and the points: try right, left, above, below, in turn."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.asarray(points))
    placed = []
    offsets = [((5, 0), "left", "center"), ((-5, 0), "right", "center"), ((0, 7), "center", "bottom"),
               ((0, -7), "center", "top"), ((5, 8), "left", "bottom"), ((5, -8), "left", "top")]
    for text, x, y in labels:
        best = None
        for off, ha, va in offsets:
            t = ax.annotate(text, (x, y), xytext=off, textcoords="offset points", fontsize=6, color=INK,
                            fontweight="bold", ha=ha, va=va, zorder=5)
            bb = t.get_window_extent(renderer).expanded(1.05, 1.15)
            hits = sum(bb.overlaps(b) for b in placed) * 10 + sum(bb.contains(*p) for p in pts)
            if best is None or hits < best[0]:
                if best is not None:
                    best[1].remove()
                best = (hits, t, bb)
            else:
                t.remove()
            if hits == 0:
                break
        placed.append(best[2])


def _head_panel(ax, zx, zy, groups, labelled, sig, rho, xlabel, ylabel, xcode, ycode) -> None:
    """Every head's PC1 score in one condition against another's; classified heads marked."""
    lim_x = (min(zx.min(), -sig) - 1.2, zx.max() + 1.6)
    lim_y = (min(zy.min(), -sig) - 1.2, zy.max() + 0.27 * (zy.max() - min(zy.min(), -sig)))
    # shaded zone: promoting in both conditions
    ax.add_patch(plt.Rectangle((sig, sig), lim_x[1] - sig, lim_y[1] - sig, color=SHARED_COL, alpha=0.12,
                               lw=0, zorder=0))
    ax.text(sig + 0.02 * (lim_x[1] - lim_x[0]), lim_y[1] - 0.02 * (lim_y[1] - lim_y[0]),
            f"promoting in\n{xcode} and {ycode}", ha="left", va="top",
            fontsize=6.3, color=SHARED_COL, fontweight="bold", linespacing=1.1)
    for v in (sig, -sig):
        ax.axvline(v, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.axhline(v, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    marked = set().union(*(set(heads) for *_, heads in groups))
    rest = [h for h in np.ndindex(zx.shape) if h not in marked]
    ax.scatter([zx[h] for h in rest], [zy[h] for h in rest], s=7, color=RANDOM, edgecolor="none",
               zorder=2, label="other heads")
    for colour, marker, label, heads in groups:   # an empty group still gets a legend entry (in fig_heads)
        ax.scatter([zx[h] for h in heads], [zy[h] for h in heads], s=38, marker=marker,
                   color=colour, edgecolor="white", lw=0.7, zorder=4, label=label)
    ax.set_xlim(*lim_x)
    ax.set_ylim(*lim_y)
    _place_labels(ax, [(head_name(*h), zx[h], zy[h]) for h in sorted(labelled, key=lambda h: -zx[h])],
                  [(zx[h], zy[h]) for h in np.ndindex(zx.shape)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"$\\rho$ = {rho:.2f}", loc="right", fontsize=7, fontweight="normal", color=INK2)


def fig_heads(out: Path) -> list[Path]:
    """Every head, in units of each condition's std: P against T and V against T, one row per model.

    PaliGemma's V scores are those of the patching file (logits as recorded, soft-capped); D79b's
    analysis on uncapped logits classifies the same heads, which the assert below checks.
    """
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_W, 3.6), gridspec_kw={"hspace": 0.42, "wspace": 0.2})
    for row, (title, model, pre) in zip(axes, MODELS):
        p = load(f"{pre}replication_patching_{model}.json")
        t, v = load(f"{pre}t_patching_{model}.json"), load(f"{pre}v_patching_{model}.json")
        vo = load(f"{pre}v_circuit_overlap_{model}.json")
        zp, zt, zv = _zscores(p), _zscores(t), _zscores(v)
        sig = p["sigma"]
        P, T, V = ({tuple(h) for h in r["promoting"]} for r in (p, t, v))
        assert all(zp[h] > sig for h in P) and all(zt[h] > sig for h in T) and all(zv[h] > sig for h in V), \
            "PC1 sign or scale mismatch"
        assert {head_name(*h) for h in V} == {h if isinstance(h, str) else head_name(*h) for h in vo["v_promoting"]}, \
            "V set in the patching file differs from the analysed one"
        rho_pt = load(f"{pre}t_circuit_overlap_{model}.json")["rank_correlation"]["rho"]
        rho_vt = vo["j_vt"]["rank_correlation"]["rho"]
        _head_panel(row[0], zp, zt,
                    [(P_COL, "o", "P only", sorted(P - T)), (T_COL, "^", "T only", sorted(T - P)),
                     (SHARED_COL, "D", "P and T", sorted(P & T))],
                    set(), sig, rho_pt, r"P: PC1 score ($\sigma$ units)", r"T: PC1 score ($\sigma$ units)", "P", "T")
        _head_panel(row[1], zv, zt,
                    [(V_COL, "s", "V only", sorted(V - T)), (T_COL, "^", "T only", sorted(T - V)),
                     (SHARED_COL, "D", "V and T", sorted(V & T))],
                    set(), sig, rho_vt, r"V: PC1 score ($\sigma$ units)", r"T: PC1 score ($\sigma$ units)", "V", "T")
        row[0].set_title(f"{title}: (a) P vs T", loc="left", fontsize=7.5)
        row[1].set_title(f"{title}: (b) V vs T", loc="left", fontsize=7.5)
    cut = Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 2)), label=f"$\\pm{sig:g}\\sigma$ cut")
    other = Line2D([], [], ls="", marker="o", ms=3, color=RANDOM, label="other heads")
    for col, (a, a_col, a_mk, b_name) in enumerate([("P", P_COL, "o", "P and T"), ("V", V_COL, "s", "V and T")]):
        handles = [Line2D([], [], ls="", marker=a_mk, ms=5.5, color=a_col, mec="white", label=f"{a} only"),
                   Line2D([], [], ls="", marker="^", ms=6, color=T_COL, mec="white", label="T only"),
                   Line2D([], [], ls="", marker="D", ms=5, color=SHARED_COL, mec="white", label=b_name),
                   other, cut]
        box = axes[1, col].get_position()
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=((box.x0 + box.x1) / 2, 0.0), ncol=5,
                   columnspacing=0.8, handletextpad=0.2, fontsize=6.6, borderpad=0.35)
    return save(fig, out, "fig_heads")


# --------------------------------------------------------------------------- 4
def fig_h3(out: Path) -> list[Path]:
    """H3: do the heads attend to the note, and write its colour? And the T-only heads with no note.

    One row per model; attention masses are not on one scale across models (different image grids).
    """
    order = ["T_only", "shared", "P_only", "T"]
    key = {"m_note": {"T_only": "A", "shared": "A:shared", "P_only": "A:P_only", "T": "A:T"},
           "dla": {"T_only": "W", "shared": "W:shared", "P_only": "W:P_only", "T": "W:T"}}
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_W, 4.3), gridspec_kw={"wspace": 0.42, "hspace": 0.72})
    for row, (model_title, model, pre) in zip(axes, MODELS):
        h = load(f"{pre}h3_analysis_{model}.json")
        tests = {**h["primary"], **h["secondary"]}
        short = short_labels(h["sets"])
        panels = [("m_note", "(a) attention to note", "attention on note", "{:.2f}"),
                  ("dla", "(b) push to note's colour", "direct effect (logits)", "{:+.2f}")]
        for ax, (measure, title, ylabel, fmt) in zip(row, panels):
            for i, s in enumerate(order):
                r = tests[key[measure][s]]
                bar_pair(ax, i, r["observed"], r["ci95"], SET_COL[s], null=(r["null_mean"], null_edge(r)),
                         label=fmt.format(r["observed"]))
            ax.set_title(title, loc="left")
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(order)), [short[s] for s in order], fontsize=6.4)
            ax.set_xlim(-0.6, len(order) - 0.4)
            ax.grid(axis="x", visible=False)
            ax.tick_params(axis="x", length=0)
        row[0].set_ylim(0, row[0].get_ylim()[1] * 1.15)
        lo, hi = row[1].get_ylim()
        row[1].set_ylim(min(lo, 0) - 0.35, hi * 1.18)
        row[1].axhline(0, color=INK2, lw=0.7, zorder=2)

        ax = row[2]   # the T-only heads' direct effect on three kinds of input
        rows = [("W", "note\n(T)"), ("image_colour:clean", "no note\n(T clean)"),
                ("image_colour:P_visual", "no note\n(P Visual)")]
        for i, (k, label) in enumerate(rows):
            r = tests[k]
            bar_pair(ax, i, r["observed"], r["ci95"], T_COL, null=(r["null_mean"], null_edge(r)),
                     label=f"{r['observed']:+.2f}")
        ax.axhline(0, color=INK2, lw=0.7, zorder=2)
        ax.set_xticks(range(len(rows)), [r[1] for r in rows], fontsize=6.2)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo * 1.3, hi * 1.3)
        ax.set_title("(c) T-only heads by input", loc="left")
        row[0].text(-0.3, 1.17, model_title, transform=row[0].transAxes, fontsize=8.5, fontweight="bold",
                    ha="left", va="bottom")
        ax.set_ylabel("direct effect (logits)")
        ax.text(1.03, 0.97, "toward competing\ncolour $\\uparrow$", transform=ax.transAxes, ha="left", va="top",
                fontsize=6, color=MUTED, linespacing=1.1)
        ax.text(1.03, 0.03, "toward image's\ncolour $\\downarrow$", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=6, color=MUTED, linespacing=1.1)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
    bottom_legend(fig, [
        Patch(color=T_COL, label="T heads"), Patch(color=SHARED_COL, label="shared heads"),
        Patch(color=P_COL, label="P heads"),
        Patch(color=RANDOM, label="100 random sets (as H2): mean, whisker to 95th/5th pct."),
        Line2D([], [], color=INK, marker="_", ms=6, lw=0.9, label="95% bootstrap CI"),
    ], 5, y=0.0)
    return save(fig, out, "fig_h3")


# --------------------------------------------------------------------------- 5
def fig_spatial(out: Path) -> list[Path]:
    """Plan 3.7, descriptive: T-only attention to the note and the fool rate, by note layout, per model."""
    names = {"on_object_small": "on object,\nsmall", "on_object_medium": "on object,\nmedium",
             "beside_object": "beside\nobject"}
    NEUTRAL = "#7a7a7a"
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_W, 3.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.75, 1], "wspace": 0.08, "hspace": 1.0})
    for (ax, ax2), (title, model, pre) in zip(axes, MODELS):
        sp = load(f"{pre}h3_analysis_{model}.json")["descriptive"]["spatial"]
        layouts = [k for k in names if k in sp]
        for y, k in enumerate(layouts):
            s = sp[k]
            inc, neu = s["T_only_note_mass_incongruent"], s["T_only_note_mass_neutral"]
            ax.annotate("", (inc, y), (neu, y), arrowprops=dict(arrowstyle="-|>", color=T_COL, lw=1.3,
                                                                shrinkA=3.5, shrinkB=4, mutation_scale=7), zorder=2)
            ax.plot(neu, y, "o", ms=6, color=NEUTRAL, mec="white", mew=0.7, zorder=3)
            ax.plot(inc, y, "^", ms=7, color=T_COL, mec="white", mew=0.7, zorder=3)
            ax.annotate(f"{inc:.2f}", (inc, y), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=6, color=T_COL, fontweight="bold")
            ax.annotate(f"{neu:.2f}", (neu, y), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=6, color=NEUTRAL)
            fr = 100 * s["fool_rate"]
            ax2.barh(y, fr, 0.5, color=T_COL, alpha=0.85, lw=0, zorder=3)
            ax2.annotate(f"{fr:.0f}%", (fr, y), xytext=(-3, 0), textcoords="offset points", va="center",
                         ha="right", fontsize=6.3, color="white", fontweight="bold")
        ax.set_yticks(range(len(layouts)), [names[k] for k in layouts])
        ax.set_ylim(len(layouts) - 0.5, -0.65)
        lo = min(min(sp[k]["T_only_note_mass_neutral"] for k in layouts), 0.1)
        ax.set_xlim(lo * 0.8, max(sp[k]["T_only_note_mass_incongruent"] for k in layouts) * 1.12)
        ax.set_xlabel("T-only attention on note")
        ax.set_title("(a) attention", loc="left")
        ax.text(-0.55, 1.28, title, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")
        ax2.set_xlim(0, 100)
        ax2.set_xticks([0, 50, 100])
        ax2.set_xlabel("objects fooled (%)")
        ax2.set_title("(b) fooled", loc="left")
        for a in (ax, ax2):
            a.grid(axis="y", visible=False)
            a.tick_params(axis="y", length=0)
    bottom_legend(fig, [
        Line2D([], [], ls="", marker="o", ms=5, color=NEUTRAL, label="neutral word"),
        Line2D([], [], ls="", marker="^", ms=6, color=T_COL, label="colour word (incongruent)"),
    ], 2, y=0.0)
    return save(fig, out, "fig_spatial")


FIGURES = [fig_design, fig_heads, fig_h3, fig_spatial]


def main(out_dir: str | None = None) -> list[Path]:
    out = Path(out_dir) if out_dir else ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    style()
    written = []
    for f in FIGURES:
        written += f(out)
    for p in written:
        print(p.relative_to(out.parent) if p.is_relative_to(out.parent) else p)
    return written


if __name__ == "__main__":
    main(*sys.argv[1:2])
