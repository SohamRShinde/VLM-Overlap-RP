"""Every number the paper quotes, written as LaTeX macros from committed results.

    .venv/bin/python scripts/paper_numbers.py [out.tex]     # default: paper/numbers.tex

The paper never types a result: it uses \\JPT, \\hTwoPtoTObs, ... from the
generated file, so a re-analysed result changes the paper on the next build
and a number can be traced to the file and key it came from (the comment on
each line). Nothing is computed here beyond formatting, picking the top
single heads by their recorded flip rate, and the T-screen logit shifts,
which use analyze_t_screen.py's own definition.

Reference numbers from Lietzow et al. are taken from the `reported` field our
knockout file stores alongside our own, and are cited as theirs in the text.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODEL = "qwen25vl3b"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_t_screen import load_records, position_vs_size  # noqa: E402
from circuit_overlap.screen import (  # noqa: E402
    has_numerics_failure, summarise_screen, vocabulary_from_records)


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())["result"]


class Macros:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.names: set[str] = set()

    def add(self, name: str, value: str, source: str) -> None:
        if not name.isalpha():
            raise ValueError(f"LaTeX macro names are letters only: {name!r}")
        if name in self.names:
            raise ValueError(f"duplicate macro {name}")
        self.names.add(name)
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value}}}  % {source}")

    def section(self, title: str) -> None:
        self.lines.append(f"\n% --- {title}")


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def num(x: float, d: int = 2, sign: bool = False) -> str:
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") or "-" in s else s


def pval(p: float) -> str:
    return f"{p:.2f}" if p >= 0.1 else f"{p:.4f}".rstrip("0")


def test(m: Macros, prefix: str, t: dict, src: str, scale=100.0, d=1, sign=False) -> None:
    """Observed value, CI, random-set mean, the random 95th (or 5th) percentile, p."""
    f = (lambda v: f"{v * scale:.{d}f}") if scale == 100.0 else (lambda v: num(v, d, sign))
    m.add(prefix + "Obs", f(t["observed"]), f"{src} observed")
    m.add(prefix + "Lo", f(t["ci95"][0]), f"{src} ci95[0]")
    m.add(prefix + "Hi", f(t["ci95"][1]), f"{src} ci95[1]")
    m.add(prefix + "Null", f(t["null_mean"]), f"{src} null_mean")
    edge = "null_p5" if t.get("direction") == "less" else "null_p95"
    m.add(prefix + "NullEdge", f(t[edge]), f"{src} {edge}")
    m.add(prefix + "P", pval(t["p"]), f"{src} p")


def build() -> Macros:
    m = Macros()

    # ------------------------------------------------------------ T screen
    m.section("T behavioural screen, results/t_screen_qwen25vl3b.json (D41)")
    records, _ = load_records(RES / f"t_screen_{MODEL}.json")
    vocab = vocabulary_from_records(records)
    s = summarise_screen(records, vocab, min_usable=50)   # the recomputed summary, as analyze_t_screen.py
    m.add("nObjects", str(s["n_objects"]), "t_screen recomputed n_objects")
    m.add("nCorrectNeutral", str(s["n_objects_correct_neutral"]), "t_screen recomputed n_objects_correct_neutral")
    m.add("nScreenUsable", str(s["n_objects_usable"]), "t_screen recomputed n_objects_usable")
    m.add("screenFoolRate", pct(s["fool_rate_among_eligible"]), "t_screen recomputed fool_rate_among_eligible")
    clean = [r for r in records if not has_numerics_failure(r)]
    pos = position_vs_size(clean, vocab)
    for k, word in [("on_object_small", "Small"), ("on_object_medium", "Medium"), ("beside_object", "Beside")]:
        m.add(f"screenFool{word}", pct(pos["layouts"][k]["fool_rate"]), f"position_vs_size {k} fool_rate")
    m.add("pairedOnOnly", str(pos["on_vs_beside"]["only_on"]), "position_vs_size on_vs_beside.only_on")
    m.add("pairedBesideOnly", str(pos["on_vs_beside"]["only_beside"]), "position_vs_size on_vs_beside.only_beside")

    def shifts(a: str, b: str) -> np.ndarray:   # analyze_t_screen.py's definition
        return np.array([getattr(r, a) - getattr(r, b) for r in clean
                         if getattr(r, a) is not None and getattr(r, b) is not None
                         and math.isfinite(getattr(r, a)) and math.isfinite(getattr(r, b))])
    attack, congruent = shifts("delta_neutral", "delta_incongruent"), shifts("delta_neutral", "delta_congruent")
    m.add("attackShift", num(attack.mean(), 2, True), "t_screen records, mean attack shift")
    m.add("attackShiftMin", num(attack.min(), 2, True), "t_screen records, min attack shift")
    m.add("congruentShift", num(congruent.mean(), 2, True), "t_screen records, mean congruent shift")
    m.add("congruentShiftMax", num(congruent.max(), 2, True), "t_screen records, max congruent shift")

    # ------------------------------------------------------------ replication
    m.section("Replication gate, results/replication_{patching,knockout}_qwen25vl3b.json (D40, D67)")
    rp = load(f"replication_patching_{MODEL}.json")
    ko = load(f"replication_knockout_{MODEL}.json")
    n_layers, n_heads = np.asarray(rp["pc1_scores"]).shape
    m.add("nLayers", str(n_layers), "replication_patching pc1_scores shape")
    m.add("nHeadsPerLayer", str(n_heads), "replication_patching pc1_scores shape")
    m.add("nHeads", str(n_layers * n_heads), "replication_patching pc1_scores size")
    m.add("nP", str(rp["n_scored"]), "replication_patching n_scored")
    m.add("nPromP", str(len(rp["promoting"])), "replication_patching promoting")
    m.add("nSupP", str(len(rp["suppressing"])), "replication_patching suppressing")
    same = {tuple(h) for h in rp["promoting"]} == {tuple(h) for h in rp["reference_promoting"]}
    m.add("promMatchesLietzow", "identical to" if same else "different from", "promoting == reference_promoting")
    g = ko["counts"]["grp_promoting"]
    m.add("repPriorFlip", pct(g["prior_flip_rate"]), "knockout grp_promoting prior_flip_rate")
    m.add("repPriorFlipCount", str(g["prior_flips"]), "knockout grp_promoting prior_flips")
    m.add("repVisualFlip", pct(g["visual_flip_rate"]), "knockout grp_promoting visual_flip_rate")
    m.add("lietzowPriorFlip", f"{100 * ko['reported']['grp_promoting']['prior_flip']:.1f}", "knockout reported (Lietzow)")
    m.add("lietzowVisualFlip", f"{100 * ko['reported']['grp_promoting']['visual_flip']:.1f}", "knockout reported (Lietzow)")
    singles = sorted(((k[4:], v["prior_flip_rate"]) for k, v in ko["counts"].items()
                      if k.startswith("ind_") and tuple(map(int, k[5:].split("H"))) in
                      {tuple(h) for h in rp["promoting"]}), key=lambda kv: -kv[1])
    for word, (name, rate) in zip(("One", "Two"), singles[:2]):
        m.add(f"topSingle{word}", name, "knockout ind_*, ranked by prior_flip_rate")
        m.add(f"topSingle{word}Flip", pct(rate), f"knockout ind_{name} prior_flip_rate")
        lz = ko["reported"]["single_head_prior_flip_pct"].get(name)
        m.add(f"topSingle{word}Lietzow", f"{lz:.1f}" if lz is not None else "n/a", "knockout reported (Lietzow)")

    # ------------------------------------------------------------ T circuit and overlap
    m.section("T circuit and J_PT, results/t_patching_qwen25vl3b.json, t_circuit_overlap_qwen25vl3b.json (D70-D72)")
    tp = load(f"t_patching_{MODEL}.json")
    o = load(f"t_circuit_overlap_{MODEL}.json")
    c = load(f"split_half_ceiling_p_{MODEL}.json")
    pr, so, rc, de = o["primary"], o["shared_objects"], o["rank_correlation"], o["descriptive"]
    m.add("nT", str(tp["n_scored"]), "t_patching n_scored")
    m.add("nPromT", str(len(tp["promoting"])), "t_patching promoting")
    m.add("nSupT", str(len(tp["suppressing"])), "t_patching suppressing")
    m.add("nShared", str(len(o["shared_promoting"])), "t_circuit_overlap shared_promoting")
    m.add("nPonly", str(len(o["p_promoting"]) - len(o["shared_promoting"])), "p_promoting - shared")
    m.add("nTonly", str(len(o["t_promoting"]) - len(o["shared_promoting"])), "t_promoting - shared")
    m.add("JPT", f"{pr['j_pt']:.3f}", "t_circuit_overlap primary.j_pt")
    m.add("JPTLo", f"{pr['ci95'][0]:.3f}", "primary.ci95[0]")
    m.add("JPTHi", f"{pr['ci95'][1]:.3f}", "primary.ci95[1]")
    m.add("JPTVerdict", pr["wording"], "primary.wording (D70 fixed phrase)")
    m.add("chanceMean", f"{pr['chance_floor_mean']:.3f}", "primary.chance_floor_mean")
    m.add("chanceEdge", f"{pr['chance_floor_p95_at_observed_sizes']:.3f}", "primary.chance_floor_p95_at_observed_sizes")
    m.add("ceilingMedian", f"{c['jaccard_promoting']['median']:.3f}", "split_half jaccard_promoting.median")
    m.add("ceilingLo", f"{c['jaccard_promoting']['p2_5']:.3f}", "split_half jaccard_promoting.p2_5")
    m.add("nHalfSplits", str(c["n_splits"]), "split_half n_splits")
    m.add("TselfJ", f"{o['stable_core']['t_self_agreement_jaccard']['median']:.3f}", "stable_core.t_self_agreement_jaccard.median")
    m.add("rankRho", f"{rc['rho']:.2f}", "rank_correlation.rho")
    m.add("rankCeiling", f"{rc['p_ceiling_full_length']:.2f}", "rank_correlation.p_ceiling_full_length")
    m.add("nSharedObjects", str(so["n_objects"]), "shared_objects.n_objects")
    m.add("JPTshared", f"{so['j_pt']:.3f}", "shared_objects.j_pt")
    m.add("JPTsharedLo", f"{so['ci95'][0]:.3f}", "shared_objects.ci95[0]")
    m.add("JPTsharedHi", f"{so['ci95'][1]:.3f}", "shared_objects.ci95[1]")
    def layers(heads) -> str:
        ls = sorted({int(h[1:].split("H")[0]) for h in heads})
        return f"{ls[0]}" if len(ls) == 1 else f"{ls[0]}--{ls[-1]}"
    shared = set(o["shared_promoting"])
    m.add("layersPonly", layers(set(o["p_promoting"]) - shared), "layers of p_promoting - shared")
    m.add("layersTonly", layers(set(o["t_promoting"]) - shared), "layers of t_promoting - shared")
    m.add("layersShared", layers(shared), "layers of shared_promoting")
    m.add("containPinT", f"{de['containment_p_in_t']:.2f}", "descriptive.containment_p_in_t")
    m.add("JSup", f"{de['j_suppressing']:.3f}", "descriptive.j_suppressing")

    # ------------------------------------------------------------ H2
    m.section("H2 cross-ablation, results/cross_ablation_analysis_qwen25vl3b.json (D73)")
    x = load(f"cross_ablation_analysis_{MODEL}.json")
    tests = {**x["primary"], **x["diagonal"], **x["decomposition"]}
    m.add("hTwoDraws", str(x["primary"]["P->T"]["n_draws"]), "primary P->T n_draws")
    for key, prefix in [("P->T", "hTwoPtoT"), ("T->P", "hTwoTtoP"), ("P->P", "hTwoPtoP"),
                        ("T->T", "hTwoTtoT"), ("shared->P", "hTwoStoP"), ("shared->T", "hTwoStoT"),
                        ("P_only->T", "hTwoPOtoT"), ("T_only->P", "hTwoTOtoP")]:
        test(m, prefix, tests[key], f"cross_ablation {key}")
        m.add(prefix + "Count", str(tests[key]["count"]), f"cross_ablation {key} count")
        m.add(prefix + "N", str(tests[key]["n"]), f"cross_ablation {key} n")
    own = x["own_condition_subsets"]
    m.add("hTwoTOtoTObs", pct(own["T_only->T"]["observed"]), "own_condition_subsets T_only->T")
    m.add("hTwoPOtoPObs", pct(own["P_only->P"]["observed"]), "own_condition_subsets P_only->P")
    m.add("hTwoTOonPShift", num(x["decomposition"]["T_only->P"]["logit_diff_change"]["mean"], 2, True),
          "decomposition T_only->P logit_diff_change.mean")
    m.add("hTwoStoPShift", num(x["decomposition"]["shared->P"]["logit_diff_change"]["mean"], 2, True),
          "decomposition shared->P logit_diff_change.mean")
    col = x["collateral_target_flips"]["P"]
    m.add("hTwoCollatT", pct(col["T"]["target_flip_rate"]), "collateral_target_flips P.T")
    m.add("hTwoCollatTO", pct(col["T_only"]["target_flip_rate"]), "collateral_target_flips P.T_only")
    acc = x["clean_accuracy"]
    m.add("hTwoCleanT", pct(acc["accuracy_kept"]["T"]), "clean_accuracy accuracy_kept.T")
    m.add("hTwoCleanTO", pct(acc["accuracy_kept"]["T_only"]), "clean_accuracy accuracy_kept.T_only")
    m.add("hTwoCleanRandom", pct(acc["random_mean_rT"]), "clean_accuracy random_mean_rT")
    m.add("hTwoMeanSentence", "Mean ablation gives the same verdicts." if x["mean_ablation"]["same_wording_as_zero"]
          else "Mean ablation changes at least one verdict.", "mean_ablation.same_wording_as_zero")

    # ------------------------------------------------------------ H3
    m.section("H3 mechanism, results/h3_analysis_qwen25vl3b.json (D74)")
    h = load(f"h3_analysis_{MODEL}.json")
    ht = {**h["primary"], **h["secondary"], **h["validation"]}
    m.add("hThreeNT", str(h["n_objects"]["T"]), "n_objects.T")
    m.add("hThreeNP", str(h["n_objects"]["P"]), "n_objects.P")
    m.add("hThreeDropT", str(h["n_dropped"]["T"]), "n_dropped.T")
    m.add("hThreeDropP", str(h["n_dropped"]["P"]), "n_dropped.P")
    m.add("hThreeMechanism", h["mechanism"], "mechanism (D74 fixed phrase)")
    for key, prefix, scale, d in [
            ("A", "hThreeA", 1, 3), ("W", "hThreeW", 1, 2),
            ("A:shared", "hThreeAS", 1, 3), ("W:shared", "hThreeWS", 1, 2),
            ("A:P_only", "hThreeAPO", 1, 3), ("W:P_only", "hThreeWPO", 1, 2),
            ("A_rise:T_only", "hThreeRise", 1, 3), ("A_rise:shared", "hThreeRiseS", 1, 3),
            ("object:clean", "hThreeObjClean", 1, 3), ("image_colour:clean", "hThreeColClean", 1, 2),
            ("object:P_visual", "hThreeObjPV", 1, 3), ("image_colour:P_visual", "hThreeColPV", 1, 2),
            ("routing:P", "hThreeRouting", 1, 3)]:
        test(m, prefix, ht[key], f"h3 {key}", scale=scale, d=d, sign=(d == 2))
    ds = h["descriptive"]
    m.add("noteShare", pct(ds["note_token_share"]), "descriptive.note_token_share")
    en = ds["note_enrichment"]
    m.add("enrichInc", f"{en['T_only:on_object_small/incongruent']:.1f}", "note_enrichment T_only incongruent")
    m.add("enrichNeu", f"{en['T_only:on_object_small/neutral']:.1f}", "note_enrichment T_only neutral")
    sp = ds["spatial"]
    for k, word in [("on_object_small", "Small"), ("on_object_medium", "Medium"), ("beside_object", "Beside")]:
        m.add(f"fool{word}", pct(sp[k]["fool_rate"], 0), f"spatial {k} fool_rate")
    return m


def main(out: str | None = None) -> Path:
    path = Path(out) if out else ROOT / "paper" / "numbers.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    m = build()
    path.write_text("% GENERATED by scripts/paper_numbers.py from results/. Do not edit by hand.\n"
                    + "\n".join(m.lines) + "\n")
    print(f"{path}: {len(m.names)} macros")
    return path


if __name__ == "__main__":
    main(*sys.argv[1:2])
