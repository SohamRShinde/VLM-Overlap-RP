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


# --------------------------------------------------------------------------- 1
def fig_heads(out: Path) -> list[Path]:
    """All 576 heads: P's PC1 score against T's, in units of each condition's std."""
    p, t = load(f"replication_patching_{MODEL}.json"), load(f"t_patching_{MODEL}.json")
    zp = (np.asarray(p["pc1_scores"]) - p["pc1_mean"]) / p["pc1_std"]
    zt = (np.asarray(t["pc1_scores"]) - t["pc1_mean"]) / t["pc1_std"]
    sig = p["sigma"]
    P = {tuple(h) for h in p["promoting"]}
    T = {tuple(h) for h in t["promoting"]}
    assert all(zp[h] > sig for h in P) and all(zt[h] > sig for h in T), "PC1 sign or scale mismatch"
    rho = load(f"t_circuit_overlap_{MODEL}.json")["rank_correlation"]["rho"]

    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.9))
    for v in (sig, -sig):
        ax.axvline(v, color=AXIS, lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.axhline(v, color=AXIS, lw=0.6, ls=(0, (3, 2)), zorder=0)
    rest = [h for h in np.ndindex(zp.shape) if h not in P | T]
    ax.scatter([zp[h] for h in rest], [zt[h] for h in rest], s=5, color=RANDOM, lw=0,
               zorder=1, label=f"other heads ({len(rest)})")
    groups = [("P_only", sorted(P - T)), ("T_only", sorted(T - P)), ("shared", sorted(P & T))]
    for key, heads in groups:
        colour, marker, label = SET_STYLE[key]
        ax.scatter([zp[h] for h in heads], [zt[h] for h in heads], s=26, marker=marker,
                   color=colour, edgecolor="white", lw=0.6, zorder=3, label=label)
    shared = sorted(P & T, key=lambda h: zp[h])   # direct labels on the shared heads:
    for i, h in enumerate(shared):                 # below the point if the next one is in the way
        nxt = shared[i + 1] if i + 1 < len(shared) else None
        below = nxt is not None and zp[nxt] - zp[h] < 2.5 and abs(zt[nxt] - zt[h]) < 1.0
        ax.annotate(head_name(*h), (zp[h], zt[h]), xytext=(0, -8) if below else (4, 0),
                    textcoords="offset points", fontsize=5.8, color=INK2,
                    va="center", ha="center" if below else "left")
    ax.set_xlabel(r"P condition: PC1 score ($\sigma$ units)")
    ax.set_ylabel(r"T condition: PC1 score ($\sigma$ units)")
    ax.text(0.98, 0.98, f"Spearman $\\rho$ = {rho:.2f} over all {zp.size} heads\n"
                        f"dashed: $\\pm{sig:g}\\sigma$ classification cut",
            transform=ax.transAxes, va="top", ha="right", fontsize=6, color=INK2)
    ax.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.2)
    return save(fig, out, "fig_heads")


# --------------------------------------------------------------------------- 2
def fig_overlap(out: Path) -> list[Path]:
    """J_PT on the ruler D70 reads it against: the chance floor and P's split-half ceiling."""
    o, c = load(f"t_circuit_overlap_{MODEL}.json"), load(f"split_half_ceiling_p_{MODEL}.json")
    pr, so = o["primary"], o["shared_objects"]
    rows = []   # (label, kind, payload)
    rows.append(("P vs P, split halves\n(self-agreement ceiling)", "band",
                 (c["jaccard_promoting"]["p2_5"], c["jaccard_promoting"]["p97_5"],
                  c["jaccard_promoting"]["median"], P_COL)))
    v_path = RES / f"v_circuit_overlap_{MODEL}.json"
    if v_path.exists():   # filled in once track A's V result is committed
        print(f"note: {v_path.name} exists but fig_overlap does not plot it yet; add the J_VT row")
    rows.append((f"P vs T, {so['n_objects']} objects in both", "dot",
                 (so["j_pt"], so["ci95"], SHARED_COL, "D", False)))
    rows.append((f"P vs T, all objects\n($J_{{PT}}$, headline)", "dot",
                 (pr["j_pt"], pr["ci95"], SHARED_COL, "D", True)))
    rows.append((f"random sets of {pr['set_sizes'][0]} and {pr['set_sizes'][1]}\n(chance floor)", "null",
                 (pr["chance_floor_mean"], pr["chance_floor_p95_at_observed_sizes"])))

    fig, ax = plt.subplots(figsize=(COLUMN_W, 1.75))
    for y, (label, kind, d) in enumerate(rows):
        if kind == "band":
            lo, hi, med, col = d
            ax.add_patch(plt.Rectangle((lo, y - 0.16), hi - lo, 0.32, color=col, alpha=0.3, lw=0))
            ax.plot([med, med], [y - 0.16, y + 0.16], color=col, lw=1.4)
        elif kind == "null":
            null_band(ax, y, *d)
        else:
            val, ci, col, marker, filled = d
            ax.plot(ci, [y, y], color=col, lw=1.4, solid_capstyle="round")
            ax.plot(val, y, marker, ms=5, color=col if filled else "white", mec=col, mew=1.0)
            ax.annotate(f"{val:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]", (ci[1], y), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=6, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Jaccard overlap of promoting heads")
    ax.grid(axis="y", visible=False)
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
    fig, ax = plt.subplots(figsize=(COLUMN_W, 1.5))
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


FIGURES = [fig_heads, fig_overlap, fig_cross_ablation, fig_h3, fig_spatial]


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
