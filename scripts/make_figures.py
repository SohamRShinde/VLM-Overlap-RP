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
SET_STYLE = {   # colour, marker, label
    "P": (P_COL, "o", r"$\mathcal{H}_P$ (all 9)"),
    "T": (T_COL, "^", r"$\mathcal{H}_T$ (all 12)"),
    "shared": (SHARED_COL, "D", "shared (5)"),
    "P_only": (P_COL, "o", "P-only (4)"),
    "T_only": (T_COL, "^", "T-only (7)"),
}
SHORT = {"P": "$\\mathcal{H}_P$\n(all 9)", "T": "$\\mathcal{H}_T$\n(all 12)", "shared": "shared\n(5)",
         "P_only": "P-only\n(4)", "T_only": "T-only\n(7)"}

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
RED, PURPLE = "#d6453d", "#8e5cc8"            # the example object's colours (image content, not roles)
NOTE_FILL, NOTE_EDGE, TILE = "#ffe066", "#e0b400", "#ffffff"
COND_COL = {"P": P_COL, "T": T_COL, "V": V_COL}


def strawberry(ax, cx, cy, colour, a=0.17, b=0.2) -> None:
    """An egg-shaped fruit, wider at the top, with a leaf cap and a few seeds."""
    t = np.linspace(0, 2 * np.pi, 120)
    ax.add_patch(plt.Polygon(np.column_stack([cx + a * np.cos(t) * (1 + 0.28 * np.sin(t)),
                                              cy + b * np.sin(t)]), color=colour, lw=0, zorder=2))
    for dx in (-0.07, 0.0, 0.07):
        ax.add_patch(plt.Polygon([(cx + dx - 0.045, cy + b - 0.02), (cx + 1.6 * dx, cy + b + 0.07),
                                  (cx + dx + 0.045, cy + b - 0.02)], color="#3f8f4a", lw=0, zorder=3))
    for sx, sy in [(-0.08, 0.06), (0.0, 0.1), (0.08, 0.05), (-0.04, -0.03), (0.05, -0.06), (0.0, -0.13)]:
        ax.plot(cx + sx, cy + sy, "o", ms=1.1, color="#f3e7a1", zorder=3)


def tile(ax, x, y, s, colour, edge, note=None) -> None:
    """One input image: the object, and optionally a note written on it."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=TILE, ec=edge, lw=0.9, zorder=1))
    strawberry(ax, x + s / 2, y + s * 0.47, colour)
    if note:
        w, h = 0.36, 0.14
        ax.add_patch(plt.Rectangle((x + s / 2 - w / 2, y + s * 0.38 - h / 2), w, h,
                                   fc=NOTE_FILL, ec=NOTE_EDGE, lw=0.6, zorder=4))
        ax.text(x + s / 2, y + s * 0.38, note, ha="center", va="center", fontsize=5.8,
                color=INK, zorder=5, fontfamily="DejaVu Sans Mono", fontweight="bold")


def fig_design(out: Path) -> list[Path]:
    """Figure 1, a schematic of the three conflicts (illustrative; no result is drawn).

    Each conflict is a pair of runs that differ in one factor; patching swaps one head's
    last-token output between them, in both directions.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    W, H, s = TEXT_W, 1.92, 0.62
    prior_q, visual_q = "What color is a\nstrawberry usually?", "What color is this\nstrawberry here?"
    blocks = [
        ("P", "prompted prior", "prompt varies",
         [(PURPLE, None, prior_q, "red", RED, "memory"), (PURPLE, None, visual_q, "purple", PURPLE, "image")]),
        ("T", "typographic", "note's word varies",
         [(RED, "purple", visual_q, "purple", PURPLE, "note"), (RED, "pencil", visual_q, "red", RED, "image")]),
        ("V", "image-varying", "pixels vary",
         [(RED, None, visual_q, "red", RED, "image"), (PURPLE, None, visual_q, "purple", PURPLE, "image")]),
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
        for x, (obj, note, prompt, answer, ans_col, basis) in zip(xs, runs):
            tile(ax, x, 0.68, s, obj, col, note)
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


def _head_panel(ax, zx, zy, groups, labelled, sig, rho, xlabel, ylabel, xcode, ycode) -> None:
    """Every head's PC1 score in one condition against another's; classified heads marked."""
    lim_x = (min(zx.min(), -sig) - 1.2, zx.max() + 1.6)
    lim_y = (min(zy.min(), -sig) - 1.2, zy.max() + 1.6)
    # shaded zone: promoting in both conditions
    ax.add_patch(plt.Rectangle((sig, sig), lim_x[1] - sig, lim_y[1] - sig, color=SHARED_COL, alpha=0.12,
                               lw=0, zorder=0))
    ax.text(lim_x[1] - 0.3, lim_y[1] - 0.4, f"promoting in\n{xcode} and {ycode}", ha="right", va="top",
            fontsize=6.3, color=SHARED_COL, fontweight="bold", linespacing=1.1)
    for v in (sig, -sig):
        ax.axvline(v, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.axhline(v, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    marked = set().union(*(set(heads) for *_, heads in groups))
    rest = [h for h in np.ndindex(zx.shape) if h not in marked]
    ax.scatter([zx[h] for h in rest], [zy[h] for h in rest], s=7, color=RANDOM, edgecolor="none",
               zorder=2, label=f"other heads ({len(rest)})")
    for colour, marker, label, heads in groups:   # an empty group stays in the legend: "(0)" is the finding
        ax.scatter([zx[h] for h in heads], [zy[h] for h in heads], s=38, marker=marker,
                   color=colour, edgecolor="white", lw=0.7, zorder=4, label=f"{label} ({len(heads)})")
    shown = sorted(labelled, key=lambda h: zx[h])  # direct labels; below the point if the next is in the way
    for i, h in enumerate(shown):
        nxt = shown[i + 1] if i + 1 < len(shown) else None
        below = nxt is not None and zx[nxt] - zx[h] < 2.5 and abs(zy[nxt] - zy[h]) < 1.0
        ax.annotate(head_name(*h), (zx[h], zy[h]), xytext=(0, -8.5) if below else (5, 0),
                    textcoords="offset points", fontsize=6, color=INK, fontweight="bold",
                    va="center", ha="center" if below else "left", zorder=5)
    ax.set_xlim(*lim_x)
    ax.set_ylim(*lim_y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    leg = ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.3, labelspacing=0.12,
                    borderpad=0.3, fontsize=6.2, markerscale=0.85,
                    title=f"Spearman $\\rho$ = {rho:.2f} (all {zx.size} heads)", title_fontsize=6.2)
    leg.get_title().set_color(INK)


def fig_heads(out: Path) -> list[Path]:
    """All 576 heads, in units of each condition's std: (a) P against T, (b) V against T."""
    p, t = load(f"replication_patching_{MODEL}.json"), load(f"t_patching_{MODEL}.json")
    v = load(f"v_patching_{MODEL}.json")
    zp, zt, zv = _zscores(p), _zscores(t), _zscores(v)
    sig = p["sigma"]
    P, T, V = ({tuple(h) for h in r["promoting"]} for r in (p, t, v))
    assert all(zp[h] > sig for h in P) and all(zt[h] > sig for h in T) and all(zv[h] > sig for h in V), \
        "PC1 sign or scale mismatch"
    rho_pt = load(f"t_circuit_overlap_{MODEL}.json")["rank_correlation"]["rho"]
    rho_vt = load(f"v_circuit_overlap_{MODEL}.json")["j_vt"]["rank_correlation"]["rho"]

    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_W, 5.2), gridspec_kw={"hspace": 0.42})
    _head_panel(axes[0], zp, zt,
                [(P_COL, "o", "P only", sorted(P - T)), (T_COL, "^", "T only", sorted(T - P)),
                 (SHARED_COL, "D", "P and T", sorted(P & T))],
                P & T, sig, rho_pt, r"P condition: PC1 score ($\sigma$ units)",
                r"T condition: PC1 score ($\sigma$ units)", "P", "T")
    _head_panel(axes[1], zv, zt,
                [(V_COL, "s", "V only", sorted(V - T)), (T_COL, "^", "T only", sorted(T - V)),
                 (SHARED_COL, "D", "V and T", sorted(V & T))],
                set(), sig, rho_vt, r"V condition: PC1 score ($\sigma$ units)",
                r"T condition: PC1 score ($\sigma$ units)", "V", "T")
    axes[0].set_title("(a) prompted prior (P) vs typographic (T)", loc="left")
    axes[1].set_title("(b) image-varying (V) vs typographic (T)", loc="left")
    bottom_legend(fig, [Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 2)),
                               label=f"$\\pm{sig:g}\\sigma$ classification cut")], 1, y=0.045)
    return save(fig, out, "fig_heads")


# --------------------------------------------------------------------------- 2
def fig_overlap(out: Path) -> list[Path]:
    """Every circuit overlap on the ruler D70 reads it against (D72 computes, D75 extends it to V).

    One row per comparison: the observed Jaccard (bar) with its bootstrap CI; grey band, random
    head sets of the observed sizes (to the 95th percentile, tick at the mean); gold band, the
    split-half self-agreement the row is read against (2.5th-97.5th percentile, tick at the
    median). Hatched rows use only the objects both conditions keep and carry no ruler of their own.
    """
    pt = load(f"t_circuit_overlap_{MODEL}.json")
    v = load(f"v_circuit_overlap_{MODEL}.json")
    pg = load("pg_t_circuit_overlap_paligemma2_3b.json")
    ceilings = {"P": load(f"split_half_ceiling_p_{MODEL}.json")["jaccard_promoting"],
                "V": v["v_ceiling"]["jaccard_promoting"],
                "T": v["t_ceiling_summary"]["jaccard_promoting"],
                "pgP": load("split_half_ceiling_p_paligemma2_3b.json")["jaccard_promoting"]}

    def full(label, block, ceiling):
        return (label, block["primary"]["j_pt"], block["primary"]["ci95"], True,
                (block["primary"]["chance_floor_mean"], block["primary"]["chance_floor_p95_at_observed_sizes"]),
                ceilings[ceiling])

    def shared(label, block):
        so = block["shared_objects"]
        return (f"{label}\n({so['n_objects']} shared objects)", so["j_pt"], so["ci95"], False, None, None)

    panels = [
        ("Qwen2.5-VL-3B", [full("P vs T", pt, "P"), shared("P vs T", pt),
                           full("V vs T", v["j_vt"], v["j_vt"]["ceiling_binding"]),
                           full("P vs V", v["j_pv"], v["j_pv"]["ceiling_binding"])]),
        ("PaliGemma2-3B", [full("P vs T", pg, "pgP"), shared("P vs T", pg)]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_W, 3.35), sharex=True,
                             gridspec_kw={"height_ratios": [len(r) for _, r in panels], "hspace": 0.32})
    for ax, (title, rows) in zip(axes, panels):
        for y, (label, val, ci, filled, chance, ceiling) in enumerate(rows):
            if chance is not None:
                mean, edge = chance
                ax.add_patch(plt.Rectangle((0, y - 0.38), edge, 0.76, color=RANDOM, alpha=0.7, lw=0, zorder=1))
                ax.plot([mean] * 2, [y - 0.38, y + 0.38], color=RANDOM_DARK, lw=1.0, zorder=2)
            if ceiling is not None:
                ax.add_patch(plt.Rectangle((ceiling["p2_5"], y - 0.38), ceiling["p97_5"] - ceiling["p2_5"], 0.76,
                                           color=CEIL_COL, alpha=0.45, lw=0, zorder=1))
                ax.plot([ceiling["median"]] * 2, [y - 0.38, y + 0.38], color="#b38600", lw=1.2, zorder=2)
            h = 0.42
            ax.barh(y, val, h, color=SHARED_COL if filled else "white", edgecolor=SHARED_COL,
                    hatch=None if filled else "////", lw=0 if filled else 1.0, zorder=3)
            ax.errorbar(val, y, xerr=[[val - ci[0]], [ci[1] - val]], fmt="none", ecolor=INK,
                        elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=4)
            ax.annotate(f"{val:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]", (ci[1], y), xytext=(4, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=6.3, color=INK,
                        fontweight="bold" if filled else "normal")
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    dj = v["delta_j"]
    axes[0].text(1.0, -0.045, f"pre-registered: $\\Delta J = J_{{VT}} - J_{{PT}}$ = {dj['delta_j']:+.2f} "
                             f"[{dj['ci95'][0]:+.2f}, {dj['ci95'][1]:+.2f}]",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=6.2, color=INK,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cfcfcf", lw=0.6), zorder=6)
    axes[-1].set_xlim(0, 1)
    axes[-1].set_xlabel("Jaccard overlap of promoting heads")
    bottom_legend(fig, [
        Patch(color=SHARED_COL, label="observed overlap (95% CI)"),
        Patch(fc="white", ec=SHARED_COL, hatch="////", label="observed, shared objects only"),
        Patch(color=RANDOM, alpha=0.7, label="chance: random head sets"),
        Patch(color=CEIL_COL, alpha=0.45, label="ceiling: split-half self-agreement"),
    ], 2, y=0.01)
    return save(fig, out, "fig_overlap")


# --------------------------------------------------------------------------- 3
def fig_cross_ablation(out: Path) -> list[Path]:
    """Flip rate of the override when a head set is zero-ablated, against layer-matched random sets."""
    x = load(f"cross_ablation_analysis_{MODEL}.json")
    tests = {**x["primary"], **x["diagonal"], **x["decomposition"]}
    own = x["own_condition_subsets"]
    order = ["P", "T", "shared", "P_only", "T_only"]

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W * 0.66, 2.2), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, cond, title in zip(axes, ("P", "T"), ("(a) on P objects", "(b) on T objects")):
        for i, s in enumerate(order):
            key = f"{s}->{cond}"
            colour = SET_STYLE[s][0]
            if key in tests:
                r = tests[key]
                bar_pair(ax, i, 100 * r["observed"], [100 * v for v in r["ci95"]], colour,
                         null=(100 * r["null_mean"], 100 * r["null_p95"]), label=f"{100 * r['observed']:.1f}%")
            else:   # reported without a random control (descriptive in D73)
                r = own[key]
                bar_pair(ax, i, 100 * r["observed"], None, colour, hatch="////", w=0.5,
                         label=f"{100 * r['observed']:.1f}%*")
        ax.set_title(title, loc="left")
        ax.set_xticks(range(len(order)), [SHORT[s] for s in order])
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylim(0, 112)
    axes[0].set_yticks(range(0, 101, 25))
    axes[0].set_ylabel("override defeated (% of objects)")
    bottom_legend(fig, [
        Patch(color=P_COL, label="P heads"), Patch(color=T_COL, label="T heads"),
        Patch(color=SHARED_COL, label="shared"),
        Patch(color=RANDOM, label="100 random sets: mean,\nwhisker to 95th pct."),
        Patch(fc="white", ec=MUTED, hatch="////", label="* no random\ncontrol"),
    ], 5, y=-0.03)
    return save(fig, out, "fig_cross_ablation")


# --------------------------------------------------------------------------- 4
def fig_h3(out: Path) -> list[Path]:
    """H3: do the heads attend to the note, and write its colour? And the T-only heads with no note."""
    h = load(f"h3_analysis_{MODEL}.json")
    tests = {**h["primary"], **h["secondary"]}
    order = ["T_only", "shared", "P_only", "T"]
    key = {"m_note": {"T_only": "A", "shared": "A:shared", "P_only": "A:P_only", "T": "A:T"},
           "dla": {"T_only": "W", "shared": "W:shared", "P_only": "W:P_only", "T": "W:T"}}

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.25), gridspec_kw={"wspace": 0.42})
    panels = [("m_note", "(a) attention to note", "attention mass on note tokens", "{:.2f}"),
              ("dla", "(b) push to note's colour", "direct effect (logits)", "{:+.2f}")]
    for ax, (measure, title, ylabel, fmt) in zip(axes, panels):
        for i, s in enumerate(order):
            r = tests[key[measure][s]]
            bar_pair(ax, i, r["observed"], r["ci95"], SET_STYLE[s][0], null=(r["null_mean"], null_edge(r)),
                     label=fmt.format(r["observed"]))
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(order)), [SHORT[s] for s in order])
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.12)
    lo, hi = axes[1].get_ylim()
    axes[1].set_ylim(lo - 0.35, hi * 1.16)
    axes[1].axhline(0, color=INK2, lw=0.7, zorder=2)

    ax = axes[2]   # the T-only heads' direct effect on three kinds of input
    rows = [("W", "note\n(T)"), ("image_colour:clean", "no note\n(T clean)"),
            ("image_colour:P_visual", "no note\n(P Visual)")]
    for i, (k, label) in enumerate(rows):
        r = tests[k]
        bar_pair(ax, i, r["observed"], r["ci95"], T_COL, null=(r["null_mean"], null_edge(r)),
                 label=f"{r['observed']:+.2f}")
    ax.axhline(0, color=INK2, lw=0.7, zorder=2)
    ax.set_xticks(range(len(rows)), [r[1] for r in rows], fontsize=6.4)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo * 1.25, hi * 1.25)
    ax.set_title("(c) T-only heads by input", loc="left")
    ax.set_ylabel("direct effect (logits)")
    ax.text(1.03, 0.97, "toward competing\ncolour $\\uparrow$", transform=ax.transAxes, ha="left", va="top",
            fontsize=6, color=MUTED, linespacing=1.1)
    ax.text(1.03, 0.03, "toward image's\ncolour $\\downarrow$", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6, color=MUTED, linespacing=1.1)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    bottom_legend(fig, [
        Patch(color=T_COL, label="T heads"), Patch(color=SHARED_COL, label="shared heads"),
        Patch(color=P_COL, label="P heads"),
        Patch(color=RANDOM, label="100 random sets (as H2): mean, whisker to 95th/5th pct."),
        Line2D([], [], color=INK, marker="_", ms=6, lw=0.9, label="95% bootstrap CI"),
    ], 5, y=-0.03)
    return save(fig, out, "fig_h3")


# --------------------------------------------------------------------------- 5
def fig_spatial(out: Path) -> list[Path]:
    """Plan 3.7, descriptive: T-only attention to the note and the fool rate, by note layout."""
    sp = load(f"h3_analysis_{MODEL}.json")["descriptive"]["spatial"]
    names = {"on_object_small": "on object,\nsmall", "on_object_medium": "on object,\nmedium",
             "beside_object": "beside\nobject"}
    layouts = [k for k in names if k in sp]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COLUMN_W, 1.75), sharey=True,
                                  gridspec_kw={"width_ratios": [1.75, 1], "wspace": 0.08})
    NEUTRAL = "#7a7a7a"
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
    ax.set_xlim(0.1, max(sp[k]["T_only_note_mass_incongruent"] for k in layouts) * 1.12)
    ax.set_xlabel("T-only attention on note")
    ax.set_title("(a) attention", loc="left")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax2.set_xlim(0, 100)
    ax2.set_xticks([0, 50, 100])
    ax2.set_xlabel("objects fooled (%)")
    ax2.set_title("(b) fooled", loc="left")
    ax2.grid(axis="y", visible=False)
    ax2.tick_params(axis="y", length=0)
    bottom_legend(fig, [
        Line2D([], [], ls="", marker="o", ms=5, color=NEUTRAL, label="neutral word"),
        Line2D([], [], ls="", marker="^", ms=6, color=T_COL, label="colour word (incongruent)"),
    ], 2, y=-0.07)
    return save(fig, out, "fig_spatial")


FIGURES = [fig_design, fig_heads, fig_overlap, fig_cross_ablation, fig_h3, fig_spatial]


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
