"""Paper figures, drawn only from committed files in results/.

    .venv/bin/python scripts/make_figures.py [out_dir]     # default: figures/

Writes one PDF (for the paper) and one PNG (for a quick look) per figure.
Every plotted number is read from a results file; nothing is computed here
that the analysis scripts did not already report, except rescaling PC1
scores to units of their own standard deviation (the classification cut is
at +/-2 of those units, D70/Lietzow). Figures whose source file does not
exist yet (the V condition) are skipped with a message.

Colours: the dataviz skill's validated categorical slots 1-3 (blue, orange,
aqua; all-pairs CVD-safe on white), one per head set, and grey for random
heads. Aqua is below 3:1 on white, so every aqua mark is also told apart by
marker shape and a direct label.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODEL = "qwen25vl3b"

# Roles, not raw hex, in the plotting code below.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
RANDOM = "#b5b3ab"
P_COL, T_COL, SHARED_COL = "#2a78d6", "#eb6834", "#1baf7a"
SET_STYLE = {   # colour, marker, label
    "P": (P_COL, "o", r"$\mathcal{H}_P$ (all 9)"),
    "T": (T_COL, "^", r"$\mathcal{H}_T$ (all 12)"),
    "shared": (SHARED_COL, "D", "shared (5)"),
    "P_only": (P_COL, "o", "P-only (4)"),
    "T_only": (T_COL, "^", "T-only (7)"),
}

COLUMN_W, TEXT_W = 3.3, 7.0   # AAAI two-column widths, inches


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "ytick.labelcolor": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.2,
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


def null_band(ax, y, mean, edge, h=0.32):
    """Random heads: a grey bar from 0 to the 95th (or 5th) percentile, a tick at the mean."""
    lo, hi = sorted((0.0, edge))
    ax.add_patch(plt.Rectangle((lo, y - h / 2), hi - lo, h, color=RANDOM, alpha=0.45, lw=0, zorder=1))
    ax.plot([mean, mean], [y - h / 2, y + h / 2], color=INK2, lw=1.0, zorder=2)


def observed(ax, y, value, ci, set_key, filled=True):
    colour, marker, _ = SET_STYLE[set_key]
    if ci is not None:
        ax.plot(ci, [y, y], color=colour, lw=1.4, zorder=3, solid_capstyle="round")
    ax.plot(value, y, marker, ms=5, zorder=4, mew=0.8,
            color=colour if filled else "white", mec="white" if filled else colour)


# --------------------------------------------------------------------------- 0
RED, PURPLE = "#d6453d", "#8e5cc8"            # the example object's colours (image content, not roles)
NOTE_FILL, NOTE_EDGE, TILE = "#f6dc6b", "#d8bd45", "#f4f3ee"


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


def tile(ax, x, y, s, colour, note=None) -> None:
    """One input image: the object, and optionally a note written on it."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=TILE, ec=AXIS, lw=0.6, zorder=1))
    strawberry(ax, x + s / 2, y + s * 0.47, colour)
    if note:
        w, h = 0.36, 0.14
        ax.add_patch(plt.Rectangle((x + s / 2 - w / 2, y + s * 0.38 - h / 2), w, h,
                                   fc=NOTE_FILL, ec=NOTE_EDGE, lw=0.5, zorder=4))
        ax.text(x + s / 2, y + s * 0.38, note, ha="center", va="center", fontsize=5.8,
                color=INK, zorder=5, fontfamily="DejaVu Sans Mono")


def fig_design(out: Path) -> list[Path]:
    """Figure 1, a schematic of the three conflicts (illustrative; no result is drawn).

    Each conflict is a pair of runs that differ in one factor; patching swaps one head's
    last-token output between them, in both directions.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    W, H, s = TEXT_W, 1.78, 0.62
    prior_q, visual_q = "What color is a\nstrawberry usually?", "What color is this\nstrawberry here?"
    blocks = [
        ("P  prompted prior", "prompt varies",
         [(PURPLE, None, prior_q, "red", RED, "memory"), (PURPLE, None, visual_q, "purple", PURPLE, "image")]),
        ("T  typographic", "note's word varies",
         [(RED, "purple", visual_q, "purple", PURPLE, "note"), (RED, "pencil", visual_q, "red", RED, "image")]),
        ("V  image-varying", "pixels vary",
         [(RED, None, visual_q, "red", RED, "image"), (PURPLE, None, visual_q, "purple", PURPLE, "image")]),
    ]
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    bw, gap = 2.27, 0.08
    for i, (name, varies, runs) in enumerate(blocks):
        x0 = 0.02 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x0, 0.03), bw, H - 0.06, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="#fbfaf7", ec=GRID, lw=0.7, zorder=0))
        ax.text(x0 + 0.1, H - 0.17, name, ha="left", va="center", fontsize=7.5, fontweight="bold", color=INK)
        ax.text(x0 + bw - 0.1, H - 0.17, varies, ha="right", va="center", fontsize=6, color=MUTED)
        xs = [x0 + 0.24, x0 + bw - 0.24 - s]
        for x, role in zip(xs, ("source run", "target run")):
            ax.text(x + s / 2, H - 0.36, role, ha="center", va="center", fontsize=5.8, color=INK2)
        for x, (obj, note, prompt, answer, ans_col, basis) in zip(xs, runs):
            tile(ax, x, 0.62, s, obj, note)
            ax.text(x + s / 2, 0.47, prompt, ha="center", va="center", fontsize=5.4,
                    color=INK2, linespacing=1.15, style="italic")
            ax.text(x + s / 2 - 0.02, 0.17, f"$\\rightarrow$ {answer}", ha="right", va="center",
                    fontsize=7, fontweight="bold", color=ans_col)
            ax.text(x + s / 2 + 0.02, 0.17, f"({basis})", ha="left", va="center", fontsize=5.6, color=MUTED)
        ax.add_patch(FancyArrowPatch((xs[0] + s + 0.05, 0.93), (xs[1] - 0.05, 0.93),
                                     arrowstyle="<|-|>", mutation_scale=6, color=INK2, lw=0.8))
        ax.text((xs[0] + s + xs[1]) / 2, 1.03, "patch", ha="center", va="bottom", fontsize=5.6, color=INK2)
    return save(fig, out, "fig_design")


# --------------------------------------------------------------------------- 1
def _zscores(r: dict) -> np.ndarray:
    return (np.asarray(r["pc1_scores"]) - r["pc1_mean"]) / r["pc1_std"]


def _head_panel(ax, zx, zy, groups, labelled, sig, rho, xlabel, ylabel, note="", corner="right") -> None:
    """Every head's PC1 score in one condition against another's; classified heads marked."""
    for v in (sig, -sig):
        ax.axvline(v, color=AXIS, lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.axhline(v, color=AXIS, lw=0.6, ls=(0, (3, 2)), zorder=0)
    marked = set().union(*(set(heads) for *_, heads in groups))
    rest = [h for h in np.ndindex(zx.shape) if h not in marked]
    ax.scatter([zx[h] for h in rest], [zy[h] for h in rest], s=5, color=RANDOM, lw=0,
               zorder=1, label=f"other heads ({len(rest)})")
    for colour, marker, label, heads in groups:   # an empty group stays in the legend: "(0)" is the finding
        ax.scatter([zx[h] for h in heads], [zy[h] for h in heads], s=26, marker=marker,
                   color=colour, edgecolor="white", lw=0.6, zorder=3, label=f"{label} ({len(heads)})")
    shown = sorted(labelled, key=lambda h: zx[h])  # direct labels; below the point if the next is in the way
    for i, h in enumerate(shown):
        nxt = shown[i + 1] if i + 1 < len(shown) else None
        below = nxt is not None and zx[nxt] - zx[h] < 2.5 and abs(zy[nxt] - zy[h]) < 1.0
        ax.annotate(head_name(*h), (zx[h], zy[h]), xytext=(0, -8) if below else (4, 0),
                    textcoords="offset points", fontsize=5.8, color=INK2,
                    va="center", ha="center" if below else "left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.98 if corner == "right" else 0.03, 0.98, f"Spearman $\\rho$ = {rho:.2f} over all {zx.size} heads" + note,
            transform=ax.transAxes, va="top", ha=corner, fontsize=6, color=INK2)
    ax.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.2)


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

    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_W, 4.8), gridspec_kw={"hspace": 0.45})
    _head_panel(axes[0], zp, zt,
                [(P_COL, "o", "P only", sorted(P - T)), (T_COL, "^", "T only", sorted(T - P)),
                 (SHARED_COL, "D", "P and T", sorted(P & T))],
                P & T, sig, rho_pt, r"P condition: PC1 score ($\sigma$ units)",
                r"T condition: PC1 score ($\sigma$ units)",
                note=f"\ndashed: $\\pm{sig:g}\\sigma$ classification cut")
    _head_panel(axes[1], zv, zt,
                [(MUTED, "s", "V only", sorted(V - T)), (T_COL, "^", "T only", sorted(T - V)),
                 (SHARED_COL, "D", "V and T", sorted(V & T))],
                set(), sig, rho_vt, r"V condition: PC1 score ($\sigma$ units)",
                r"T condition: PC1 score ($\sigma$ units)", corner="left")
    axes[0].set_title("a. prompted prior (P) vs typographic (T)", loc="left", color=INK)
    axes[1].set_title("b. image-varying (V) vs typographic (T)", loc="left", color=INK)
    return save(fig, out, "fig_heads")


# --------------------------------------------------------------------------- 2
def fig_overlap(out: Path) -> list[Path]:
    """Every circuit overlap on the ruler D70 reads it against (D72 computes, D75 extends it to V).

    One row per comparison: the observed Jaccard with its bootstrap CI; grey, random head sets of
    the observed sizes (bar to the 95th percentile, tick at the mean); blue, the split-half
    self-agreement the row is read against (2.5th-97.5th percentile, tick at the median). Hollow
    rows use only the objects both conditions keep and carry no ruler of their own.
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
        return (f"{label} ({so['n_objects']} shared)", so["j_pt"], so["ci95"], False, None, None)

    panels = [
        ("Qwen2.5-VL-3B", [full("P vs T", pt, "P"), shared("P vs T", pt),
                           full("V vs T", v["j_vt"], v["j_vt"]["ceiling_binding"]),
                           full("P vs V", v["j_pv"], v["j_pv"]["ceiling_binding"])]),
        ("PaliGemma2-3B", [full("P vs T", pg, "pgP"), shared("P vs T", pg)]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_W - 0.3, 2.65), sharex=True,
                             gridspec_kw={"height_ratios": [len(r) for _, r in panels], "hspace": 0.55})
    for ax, (title, rows) in zip(axes, panels):
        for y, (label, val, ci, filled, chance, ceiling) in enumerate(rows):
            if chance is not None:
                null_band(ax, y, *chance)
            if ceiling is not None:
                ax.add_patch(plt.Rectangle((ceiling["p2_5"], y - 0.16), ceiling["p97_5"] - ceiling["p2_5"],
                                           0.32, color=P_COL, alpha=0.28, lw=0, zorder=1))
                ax.plot([ceiling["median"]] * 2, [y - 0.16, y + 0.16], color=P_COL, lw=1.2, zorder=2)
            ax.plot(ci, [y, y], color=SHARED_COL, lw=1.4, solid_capstyle="round", zorder=3)
            ax.plot(val, y, "D", ms=4.5, zorder=4, mew=1.0,
                    color=SHARED_COL if filled else "white", mec="white" if filled else SHARED_COL)
            ax.annotate(f"{val:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]", (val, y), xytext=(0, 5.5),
                        textcoords="offset points", ha="center", fontsize=5.8, color=INK2)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.set_ylim(len(rows) - 0.5, -0.75)
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    dj = v["delta_j"]   # in the gap between the panels (the top panel's tick labels are hidden)
    axes[0].text(1.0, -0.06, f"pre-registered test: $\\Delta J = J_{{VT}} - J_{{PT}}$ = {dj['delta_j']:+.2f} "
                             f"[{dj['ci95'][0]:+.2f}, {dj['ci95'][1]:+.2f}]",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=6, color=INK2)
    axes[-1].set_xlim(0, 1)
    axes[-1].set_xlabel("Jaccard overlap of promoting heads")   # grey/blue are explained in the caption
    return save(fig, out, "fig_overlap")


# --------------------------------------------------------------------------- 3
def fig_cross_ablation(out: Path) -> list[Path]:
    """Flip rate of the override when a head set is zero-ablated, against layer-matched random sets."""
    x = load(f"cross_ablation_analysis_{MODEL}.json")
    tests = {**x["primary"], **x["diagonal"], **x["decomposition"]}
    own = x["own_condition_subsets"]
    order = ["P", "T", "shared", "P_only", "T_only"]

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W * 0.62, 1.85), sharey=True)
    for ax, cond, title in zip(axes, ("P", "T"), ("on P objects (prompted prior)", "on T objects (in-image note)")):
        for y, s in enumerate(order):
            key = f"{s}->{cond}"
            if key in tests:
                r = tests[key]
                null_band(ax, y, 100 * r["null_mean"], 100 * r["null_p95"])
                observed(ax, y, 100 * r["observed"], [100 * v for v in r["ci95"]], s)
                txt = f"{100 * r['observed']:.1f}%"
            else:   # reported without a random control (descriptive in D73)
                r = own[key]
                observed(ax, y, 100 * r["observed"], None, s, filled=False)
                txt = f"{100 * r['observed']:.1f}%*"
            ax.annotate(txt, (100 * r["observed"], y), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=5.8, color=INK2)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlim(-3, 103)
        ax.set_xlabel("override defeated (% of objects)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), ["ablate " + SET_STYLE[s][2] for s in order])
    axes[0].set_ylim(len(order) - 0.5, -0.6)
    fig.text(0.99, -0.16, "grey: 100 layer-matched random sets (bar to 95th percentile, tick at mean); "
             "* no random control", ha="right", fontsize=5.8, color=MUTED)
    return save(fig, out, "fig_cross_ablation")


# --------------------------------------------------------------------------- 4
def fig_h3(out: Path) -> list[Path]:
    """H3: do the heads attend to the note, and write its colour? And the T-only heads with no note."""
    h = load(f"h3_analysis_{MODEL}.json")
    tests = {**h["primary"], **h["secondary"]}
    order = ["T_only", "shared", "P_only", "T"]
    key = {"m_note": {"T_only": "A", "shared": "A:shared", "P_only": "A:P_only", "T": "A:T"},
           "dla": {"T_only": "W", "shared": "W:shared", "P_only": "W:P_only", "T": "W:T"}}

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 1.9),
                             gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.75})
    panels = [("m_note", "a. attention to the note", "attention mass on note tokens"),
              ("dla", "b. push toward the note's colour", "direct effect (logits)")]
    for ax, (measure, title, xlabel) in zip(axes, panels):
        for y, s in enumerate(order):
            r = tests[key[measure][s]]
            edge = r["null_p95"] if r["direction"] == "greater" else r["null_p5"]
            null_band(ax, y, r["null_mean"], edge)
            observed(ax, y, r["observed"], r["ci95"], s)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel(xlabel)
        ax.set_yticks(range(len(order)), [SET_STYLE[s][2] for s in order])
        ax.set_ylim(len(order) - 0.5, -0.6)
        ax.grid(axis="y", visible=False)
    axes[1].axvline(0, color=AXIS, lw=0.6, zorder=0)

    ax = axes[2]   # the T-only heads' direct effect on three kinds of input
    rows = [("W", "note (T)"), ("image_colour:clean", "no note (T, clean)"),
            ("image_colour:P_visual", "no note (P, Visual)")]
    for y, (k, label) in enumerate(rows):
        r = tests[k]
        edge = r["null_p95"] if r["direction"] == "greater" else r["null_p5"]
        null_band(ax, y, r["null_mean"], edge)
        observed(ax, y, r["observed"], r["ci95"], "T_only")
    ax.axvline(0, color=AXIS, lw=0.6, zorder=0)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.9)
    ax.set_title("c. T-only heads, three inputs", loc="left", color=INK)
    ax.set_xlabel("direct effect (logits)")
    for x, ha, txt in ((0.02, "left", "$\\leftarrow$ image"), (0.98, "right", "competing $\\rightarrow$")):
        ax.text(x, 0.99, txt, transform=ax.transAxes, ha=ha, va="top", fontsize=5.8, color=MUTED)
    ax.grid(axis="y", visible=False)
    fig.text(0.99, -0.14, "grey: the same 100 layer-matched random sets as H2 (bar to the 95th/5th percentile, "
             "tick at mean); whiskers: 95% bootstrap CI over objects", ha="right", fontsize=5.8, color=MUTED)
    return save(fig, out, "fig_h3")


# --------------------------------------------------------------------------- 5
def fig_spatial(out: Path) -> list[Path]:
    """Plan 3.7, descriptive: T-only attention to the note and the fool rate, by note layout."""
    sp = load(f"h3_analysis_{MODEL}.json")["descriptive"]["spatial"]
    names = {"on_object_small": "on object, small", "on_object_medium": "on object, medium",
             "beside_object": "beside object"}
    layouts = [k for k in names if k in sp]
    fig, ax = plt.subplots(figsize=(COLUMN_W - 0.27, 1.5))
    for y, k in enumerate(layouts):
        s = sp[k]
        inc, neu = s["T_only_note_mass_incongruent"], s["T_only_note_mass_neutral"]
        ax.plot([neu, inc], [y, y], color=AXIS, lw=1.0, zorder=1)
        ax.plot(neu, y, "^", ms=5, color="white", mec=T_COL, mew=1.0, zorder=2)
        ax.plot(inc, y, "^", ms=5, color=T_COL, mec="white", mew=0.6, zorder=3)
        ax.annotate(f"fooled {100 * s['fool_rate']:.0f}%", (inc, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=6, color=INK2)
    ax.set_yticks(range(len(layouts)), [names[k] for k in layouts])
    ax.set_ylim(len(layouts) - 0.5, -0.6)
    ax.set_xlim(0, max(sp[k]["T_only_note_mass_incongruent"] for k in layouts) * 1.45)
    ax.set_xlabel("T-only attention mass on note tokens")
    ax.plot([], [], "^", color="white", mec=T_COL, label="neutral word")
    ax.plot([], [], "^", color=T_COL, label="colour word")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, handletextpad=0.2)
    ax.grid(axis="y", visible=False)
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
