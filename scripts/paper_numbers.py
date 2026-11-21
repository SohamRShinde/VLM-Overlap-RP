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


def binding_ceiling(vo: dict, key: str, p_ceiling: dict) -> tuple[float, str]:
    """2.5th percentile of the split-half ceiling a V comparison is read against (the lower one, D75)."""
    b = vo[key]["ceiling_binding"]
    src = {"P": p_ceiling, "V": vo["v_ceiling"]["jaccard_promoting"],
           "T": vo["t_ceiling_summary"]["jaccard_promoting"]}[b]
    return src["p2_5"], b


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
    # The Method section states these once for every overlap analysis, so they must agree.
    params = o["parameters"]
    for other in ("pg_t_circuit_overlap_paligemma2_3b.json", f"v_circuit_overlap_{MODEL}.json"):
        op = load(other)["parameters"]
        for key in ("n_boot", "n_perm"):
            if op[key] != params[key]:
                raise ValueError(f"{key} differs: {params[key]} here vs {op[key]} in {other}")
    m.add("nBoot", str(params["n_boot"]), "parameters.n_boot (same for pg and V overlap)")
    m.add("nPerm", str(params["n_perm"]), "parameters.n_perm (same for pg and V overlap)")
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
    # The Method section states these once for both models, so they must agree.
    pgx = load("pg_cross_ablation_analysis_paligemma2_3b.json")
    if pgx["alpha"] != x["alpha"]:
        raise ValueError(f"alpha differs between models: {x['alpha']} vs {pgx['alpha']}")
    m.add("hTwoAlpha", f"{x['alpha']:g}", "alpha (same in pg_cross_ablation_analysis)")
    min_ps = {t["min_attainable_p"] for t in tests.values()}
    if len(min_ps) != 1:
        raise ValueError(f"zero-ablation tests differ in min_attainable_p: {min_ps}")
    m.add("hTwoMinP", pval(min_ps.pop()), "min_attainable_p, every zero-ablation test")
    for key, prefix in [("P->T", "hTwoPtoT"), ("T->P", "hTwoTtoP"), ("P->P", "hTwoPtoP"),
                        ("T->T", "hTwoTtoT"), ("shared->P", "hTwoStoP"), ("shared->T", "hTwoStoT"),
                        ("P_only->T", "hTwoPOtoT"), ("T_only->P", "hTwoTOtoP")]:
        test(m, prefix, tests[key], f"cross_ablation {key}")
        m.add(prefix + "Sig", "$^\\dagger$" if tests[key]["passes"] else "", f"cross_ablation {key} passes (alpha)")
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
        m.add(f"fool{word}", pct(sp[k]["fool_rate"], 1), f"spatial {k} fool_rate")

    # ------------------------------------------------------------ V condition (Qwen)
    m.section("V condition, results/v_patching_qwen25vl3b.json, v_circuit_overlap_qwen25vl3b.json (D75)")
    vp = load(f"v_patching_{MODEL}.json")
    vo = load(f"v_circuit_overlap_{MODEL}.json")
    m.add("nVselected", str(vp["n_selected"]), "v_patching n_selected")
    m.add("nV", str(vp["n_scored"]), "v_patching n_scored")
    m.add("nPromV", str(len(vo["v_promoting"])), "v_circuit_overlap v_promoting")
    m.add("nSupV", str(len(vo["v_suppressing"])), "v_circuit_overlap v_suppressing")
    m.add("layersV", layers(vo["v_promoting"]), "layers of v_promoting")
    m.add("VceilingLo", f"{vo['v_ceiling']['jaccard_promoting']['p2_5']:.3f}", "v_ceiling jaccard_promoting.p2_5")
    for key, name in [("j_vt", "JVT"), ("j_pv", "JPV")]:
        pr = vo[key]["primary"]
        m.add(name, f"{pr['j_pt']:.3f}", f"{key} primary.j_pt")
        m.add(name + "Lo", f"{pr['ci95'][0]:.3f}", f"{key} primary.ci95[0]")
        m.add(name + "Hi", f"{pr['ci95'][1]:.3f}", f"{key} primary.ci95[1]")
        m.add(name + "Verdict", pr["wording"], f"{key} primary.wording (D70 fixed phrase)")
        m.add("chanceEdge" + name[1:], f"{pr['chance_floor_p95_at_observed_sizes']:.3f}",
              f"{key} primary.chance_floor_p95_at_observed_sizes")
        lo, b = binding_ceiling(vo, key, load(f"split_half_ceiling_p_{MODEL}.json")["jaccard_promoting"])
        m.add("ceiling" + name[1:] + "Lo", f"{lo:.3f}", f"{key} ceiling_binding {b}, p2_5")
    m.add("nSharedPV", str(len(vo["j_pv"]["shared_promoting"])), "j_pv shared_promoting")
    dj = vo["delta_j"]
    m.add("dJ", num(dj["delta_j"], 3, True), "delta_j.delta_j")
    m.add("dJLo", num(dj["ci95"][0], 3, True), "delta_j.ci95[0]")
    m.add("dJHi", num(dj["ci95"][1], 3, True), "delta_j.ci95[1]")
    m.add("dJVerdict", dj["wording"].replace("J_VT", "$J_{VT}$").replace("J_PT", "$J_{PT}$")
          .replace("$ > $", " > "), "delta_j.wording (D75 fixed phrase)")
    vd = vo["descriptive"]
    m.add("nTonlyInV", str(vd["t_only_heads_in_v"]["n"]), "descriptive.t_only_heads_in_v.n")
    m.add("nSharedInV", str(vd["shared_heads_in_v"]["n"]), "descriptive.shared_heads_in_v.n")
    m.add("containVinP", f"{vd['containment_v_in_p_promoting']:.2f}", "descriptive.containment_v_in_p_promoting")
    vs = vo["j_vt"]["shared_objects"]
    m.add("nSharedObjectsVT", str(vs["n_objects"]), "j_vt shared_objects.n_objects")
    m.add("JVTshared", f"{vs['j_pt']:.3f}", "j_vt shared_objects.j_pt")
    m.add("JVTsharedLo", f"{vs['ci95'][0]:.3f}", "j_vt shared_objects.ci95[0]")
    m.add("JVTsharedHi", f"{vs['ci95'][1]:.3f}", "j_vt shared_objects.ci95[1]")

    # ------------------------------------------------------------ PaliGemma
    pg = "paligemma2_3b"
    m.section("PaliGemma replication, results/pg_replication_{patching,knockout}_paligemma2_3b.json, "
              "split_half_ceiling_p_paligemma2_3b.json (D76, D76a)")
    pp = load(f"pg_replication_patching_{pg}.json")
    pk = load(f"pg_replication_knockout_{pg}.json")
    pc = load(f"split_half_ceiling_p_{pg}.json")
    n_layers, n_heads = np.asarray(pp["pc1_scores"]).shape
    m.add("pgLayers", str(n_layers), "pg_replication_patching pc1_scores shape")
    m.add("pgHeadsPerLayer", str(n_heads), "pg_replication_patching pc1_scores shape")
    m.add("pgHeads", str(n_layers * n_heads), "pg_replication_patching pc1_scores size")
    m.add("pgNP", str(pp["n_scored"]), "pg_replication_patching n_scored")
    m.add("pgNPromP", str(len(pp["promoting"])), "pg_replication_patching promoting")
    m.add("pgNSupP", str(len(pp["suppressing"])), "pg_replication_patching suppressing")
    same = ({tuple(h) for h in pp["promoting"]} == {tuple(h) for h in pp["reference_promoting"]}
            and {tuple(h) for h in pp["suppressing"]} == {tuple(h) for h in pp["reference_suppressing"]})
    m.add("pgMatchesLietzow", "identical to" if same else "different from",
          "promoting and suppressing == reference")
    g = pk["counts"]["grp_promoting"]
    m.add("pgPriorFlip", pct(g["prior_flip_rate"]), "pg knockout grp_promoting prior_flip_rate")
    m.add("pgPriorFlipCount", str(g["prior_flips"]), "pg knockout grp_promoting prior_flips")
    m.add("pgVisualFlip", pct(g["visual_flip_rate"]), "pg knockout grp_promoting visual_flip_rate")
    m.add("pgLietzowPriorFlip", f"{100 * pk['reported']['grp_promoting']['prior_flip']:.1f}", "pg knockout reported (Lietzow)")
    m.add("pgLietzowVisualFlip", f"{100 * pk['reported']['grp_promoting']['visual_flip']:.1f}", "pg knockout reported (Lietzow)")
    singles = sorted(((k[4:], v["prior_flip_rate"]) for k, v in pk["counts"].items()
                      if k.startswith("ind_") and tuple(map(int, k[5:].split("H"))) in
                      {tuple(h) for h in pp["promoting"]}), key=lambda kv: -kv[1])
    for word, (name, rate) in zip(("One", "Two"), singles[:2]):
        m.add(f"pgTop{word}", name, "pg knockout ind_*, ranked by prior_flip_rate")
        m.add(f"pgTop{word}Flip", pct(rate), f"pg knockout ind_{name} prior_flip_rate")
        lz = pk["reported"]["single_head_prior_flip_pct"].get(name)
        m.add(f"pgTop{word}Lietzow", f"{lz:.1f}" if lz is not None else "n/a", "pg knockout reported (Lietzow)")
    m.add("pgCeilingMedian", f"{pc['jaccard_promoting']['median']:.3f}", "pg split_half jaccard_promoting.median")
    m.add("pgCeilingLo", f"{pc['jaccard_promoting']['p2_5']:.3f}", "pg split_half jaccard_promoting.p2_5")

    m.section("PaliGemma T circuit and J_PT, results/pg_t_patching_paligemma2_3b.json, "
              "pg_t_circuit_overlap_paligemma2_3b.json (D77)")
    pt = load(f"pg_t_patching_{pg}.json")
    po = load(f"pg_t_circuit_overlap_{pg}.json")
    pr, so, rc = po["primary"], po["shared_objects"], po["rank_correlation"]
    m.add("pgNT", str(pt["n_scored"]), "pg_t_patching n_scored")
    m.add("pgNPromT", str(len(pt["promoting"])), "pg_t_patching promoting")
    m.add("pgNSupT", str(len(pt["suppressing"])), "pg_t_patching suppressing")
    m.add("pgNShared", str(len(po["shared_promoting"])), "pg_t_circuit_overlap shared_promoting")
    m.add("pgJPT", f"{pr['j_pt']:.3f}", "pg primary.j_pt")
    m.add("pgJPTLo", f"{pr['ci95'][0]:.3f}", "pg primary.ci95[0]")
    m.add("pgJPTHi", f"{pr['ci95'][1]:.3f}", "pg primary.ci95[1]")
    m.add("pgJPTVerdict", pr["wording"], "pg primary.wording (D70 fixed phrase)")
    m.add("pgChanceEdge", f"{pr['chance_floor_p95_at_observed_sizes']:.3f}", "pg primary.chance_floor_p95_at_observed_sizes")
    m.add("pgNSharedObjects", str(so["n_objects"]), "pg shared_objects.n_objects")
    m.add("pgJPTshared", f"{so['j_pt']:.3f}", "pg shared_objects.j_pt")
    m.add("pgJPTsharedLo", f"{so['ci95'][0]:.3f}", "pg shared_objects.ci95[0]")
    m.add("pgJPTsharedHi", f"{so['ci95'][1]:.3f}", "pg shared_objects.ci95[1]")
    m.add("pgRankRho", f"{rc['rho']:.2f}", "pg rank_correlation.rho")
    m.add("pgRankCeiling", f"{rc['p_ceiling_full_length']:.2f}", "pg rank_correlation.p_ceiling_full_length")

    m.section("PaliGemma H2 cross-ablation, results/pg_cross_ablation_analysis_paligemma2_3b.json (D78)")
    px = load(f"pg_cross_ablation_analysis_{pg}.json")

    def wording(s: str) -> str:   # the fixed phrase, typeset; words unchanged
        return (s.replace("H_P", "$\\mathcal{H}_P$").replace("H_T", "$\\mathcal{H}_T$")
                .replace("at this n", "at this $n$"))
    tests = {**px["primary"], **px["diagonal"], **px["decomposition"]}
    m.add("pgDraws", str(px["primary"]["P->T"]["n_draws"]), "pg primary P->T n_draws")
    for key, prefix in [("P->T", "pgPtoT"), ("T->P", "pgTtoP"), ("P->P", "pgPtoP"), ("T->T", "pgTtoT"),
                        ("shared->P", "pgStoP"), ("shared->T", "pgStoT"),
                        ("P_only->T", "pgPOtoT"), ("T_only->P", "pgTOtoP")]:
        test(m, prefix, tests[key], f"pg_cross_ablation {key}")
        m.add(prefix + "Sig", "$^\\dagger$" if tests[key]["passes"] else "", f"pg_cross_ablation {key} passes (alpha)")
        m.add(prefix + "Count", str(tests[key]["count"]), f"pg_cross_ablation {key} count")
        m.add(prefix + "N", str(tests[key]["n"]), f"pg_cross_ablation {key} n")
    for key, prefix in [("P->T", "pgPtoT"), ("T->P", "pgTtoP")]:
        m.add(prefix + "Verdict", wording(px["primary"][key]["wording"]), f"pg primary {key} wording (D73 fixed phrase)")
    own = px["own_condition_subsets"]
    m.add("pgTOtoTObs", pct(own["T_only->T"]["observed"]), "pg own_condition_subsets T_only->T")
    m.add("pgPOtoPObs", pct(own["P_only->P"]["observed"]), "pg own_condition_subsets P_only->P")
    m.add("pgTOonPShift", num(px["decomposition"]["T_only->P"]["logit_diff_change"]["mean"], 2, True),
          "pg decomposition T_only->P logit_diff_change.mean")
    m.add("pgStoPShift", num(px["decomposition"]["shared->P"]["logit_diff_change"]["mean"], 2, True),
          "pg decomposition shared->P logit_diff_change.mean")
    ma = px["mean_ablation"]
    m.add("pgMeanDraws", str(ma["tests"]["P->T"]["n_draws"]), "pg mean_ablation P->T n_draws")
    for key, prefix in [("P->T", "pgMeanPtoT"), ("T->P", "pgMeanTtoP")]:
        test(m, prefix, ma["tests"][key], f"pg mean_ablation {key}")
        m.add(prefix + "Verdict", wording(ma["tests"][key]["wording"]), f"pg mean_ablation {key} wording")
    m.add("pgMeanSentence", "Mean ablation gives the same verdicts." if ma["same_wording_as_zero"]
          else "Mean ablation changes at least one verdict, so the verdict is not robust to the ablation type.",
          "pg mean_ablation.same_wording_as_zero")

    m.section("PaliGemma H3 mechanism, results/pg_h3_analysis_paligemma2_3b.json (D80, D80a)")
    ph = load(f"pg_h3_analysis_{pg}.json")
    pht = {**ph["primary"], **ph["secondary"], **ph["validation"]}
    m.add("pgThreeNT", str(ph["n_objects"]["T"]), "pg_h3 n_objects.T")
    m.add("pgThreeNP", str(ph["n_objects"]["P"]), "pg_h3 n_objects.P")
    m.add("pgThreeDropT", str(ph["n_dropped"]["T"]), "pg_h3 n_dropped.T (drift under eager attention)")
    m.add("pgThreeDropP", str(ph["n_dropped"]["P"]), "pg_h3 n_dropped.P (drift under eager attention)")
    m.add("pgThreeMechanism", ph["mechanism"], "pg_h3 mechanism (D74 fixed phrase)")
    for key, prefix, d in [
            ("A", "pgThreeA", 3), ("W", "pgThreeW", 2),
            ("A:shared", "pgThreeAS", 3), ("W:shared", "pgThreeWS", 2),
            ("A:P_only", "pgThreeAPO", 3), ("W:P_only", "pgThreeWPO", 2),
            ("A_rise:T_only", "pgThreeRise", 3), ("A_rise:shared", "pgThreeRiseS", 3),
            ("object:clean", "pgThreeObjClean", 3), ("image_colour:clean", "pgThreeColClean", 2),
            ("object:P_visual", "pgThreeObjPV", 3), ("image_colour:P_visual", "pgThreeColPV", 2),
            ("routing:P", "pgThreeRouting", 3)]:
        test(m, prefix, pht[key], f"pg_h3 {key}", scale=1, d=d, sign=(d == 2))
    pds = ph["descriptive"]
    m.add("pgNoteShare", pct(pds["note_token_share"]), "pg_h3 descriptive.note_token_share")
    pen = pds["note_enrichment"]
    m.add("pgEnrichInc", f"{pen['T_only:on_object_small/incongruent']:.1f}", "pg_h3 note_enrichment T_only incongruent")
    m.add("pgEnrichNeu", f"{pen['T_only:on_object_small/neutral']:.1f}", "pg_h3 note_enrichment T_only neutral")
    for k, word in [("on_object_small", "Small"), ("on_object_medium", "Medium"), ("beside_object", "Beside")]:
        m.add(f"pgFool{word}", pct(pds["spatial"][k]["fool_rate"], 1), f"pg_h3 spatial {k} fool_rate")

    m.section("PaliGemma V condition, results/pg_v_patching_paligemma2_3b.json, "
              "pg_v_circuit_overlap_paligemma2_3b.json (D79, D79a, D79b)")
    pvp = load(f"pg_v_patching_{pg}.json")
    pvo = load(f"pg_v_circuit_overlap_{pg}.json")
    if pvo["parameters"]["n_boot"] != params["n_boot"] or pvo["parameters"]["n_perm"] != params["n_perm"]:
        raise ValueError("pg V overlap parameters differ from the ones the Method section states")
    m.add("pgNVselected", str(pvp["n_selected"]), "pg_v_patching n_selected")
    m.add("pgNV", str(pvp["n_scored"]), "pg_v_patching n_scored")
    m.add("pgVcap", str(pvp["cap"]), "pg_v_patching cap (D79a)")
    m.add("pgNPromV", str(len(pvo["v_promoting"])), "pg_v_circuit_overlap v_promoting")
    m.add("pgNSupV", str(len(pvo["v_suppressing"])), "pg_v_circuit_overlap v_suppressing")
    m.add("pgLayersV", layers(pvo["v_promoting"]), "layers of pg v_promoting")
    m.add("pgVceilingLo", f"{pvo['v_ceiling']['jaccard_promoting']['p2_5']:.3f}", "pg v_ceiling jaccard_promoting.p2_5")
    for key, name in [("j_vt", "pgJVT"), ("j_pv", "pgJPV")]:
        pr = pvo[key]["primary"]
        m.add(name, f"{pr['j_pt']:.3f}", f"pg {key} primary.j_pt")
        m.add(name + "Lo", f"{pr['ci95'][0]:.3f}", f"pg {key} primary.ci95[0]")
        m.add(name + "Hi", f"{pr['ci95'][1]:.3f}", f"pg {key} primary.ci95[1]")
        m.add(name + "Verdict", pr["wording"], f"pg {key} primary.wording (D70 fixed phrase)")
        m.add(name + "ChanceEdge", f"{pr['chance_floor_p95_at_observed_sizes']:.3f}",
              f"pg {key} primary.chance_floor_p95_at_observed_sizes")
        lo, b = binding_ceiling(pvo, key, load(f"split_half_ceiling_p_{pg}.json")["jaccard_promoting"])
        m.add("pgCeiling" + name[3:] + "Lo", f"{lo:.3f}", f"pg {key} ceiling_binding {b}, p2_5")
    m.add("pgNSharedPV", str(len(pvo["j_pv"]["shared_promoting"])), "pg j_pv shared_promoting")
    m.add("pgNSharedVT", str(len(pvo["j_vt"]["shared_promoting"])), "pg j_vt shared_promoting")
    pdj = pvo["delta_j"]
    m.add("pgdJ", num(pdj["delta_j"], 3, True), "pg delta_j.delta_j")
    m.add("pgdJLo", num(pdj["ci95"][0], 3, True), "pg delta_j.ci95[0]")
    m.add("pgdJHi", num(pdj["ci95"][1], 3, True), "pg delta_j.ci95[1]")
    m.add("pgdJVerdict", pdj["wording"].replace("J_VT", "$J_{VT}$").replace("J_PT", "$J_{PT}$")
          .replace("$ > $", " > "), "pg delta_j.wording (D75 fixed phrase)")
    pvd = pvo["descriptive"]
    m.add("pgNTonlyInV", str(pvd["t_only_heads_in_v"]["n"]), "pg descriptive.t_only_heads_in_v.n")
    m.add("pgNTonly", str(pvd["t_only_heads_in_v"]["of"]), "pg descriptive.t_only_heads_in_v.of")
    m.add("pgNSharedInV", str(pvd["shared_heads_in_v"]["n"]), "pg descriptive.shared_heads_in_v.n")
    m.add("pgContainVinP", f"{pvd['containment_v_in_p_promoting']:.2f}", "pg descriptive.containment_v_in_p_promoting")
    pvs = pvo["j_vt"]["shared_objects"]
    m.add("pgNSharedObjectsVT", str(pvs["n_objects"]), "pg j_vt shared_objects.n_objects")
    m.add("pgJVTshared", f"{pvs['j_pt']:.3f}", "pg j_vt shared_objects.j_pt")
    m.add("pgJVTsharedLo", f"{pvs['ci95'][0]:.3f}", "pg j_vt shared_objects.ci95[0]")
    m.add("pgJVTsharedHi", f"{pvs['ci95'][1]:.3f}", "pg j_vt shared_objects.ci95[1]")
    m.add("pgSoftcapSentence", "Analysing the logits as recorded gives the same head sets and verdict."
          if pvo["d79b"]["same_as_primary"] else
          "Analysing the logits as recorded changes the head sets or the verdict.",
          "pg d79b.same_as_primary")

    # ------------------------------------------------------------ head lists (appendix table)
    m.section("Head lists, the promoting/suppressing fields of each discovery file (appendix table)")

    def head_list(heads) -> str:
        return ", ".join(f"L{l}H{h}" for l, h in sorted(tuple(x) for x in heads)) or "none"
    for stem, fname in [("listP", f"replication_patching_{MODEL}.json"), ("listT", f"t_patching_{MODEL}.json"),
                        ("listV", f"v_patching_{MODEL}.json"),
                        ("pgListP", f"pg_replication_patching_{pg}.json"),
                        ("pgListT", f"pg_t_patching_{pg}.json")]:
        r = load(fname)
        m.add(stem + "Prom", head_list(r["promoting"]), f"{fname} promoting")
        m.add(stem + "Sup", head_list(r["suppressing"]), f"{fname} suppressing")
    # PaliGemma V: the D79b (uncapped) analysis's sets, which the text reports.
    def label_list(labels) -> str:
        return ", ".join(sorted(labels, key=lambda s: tuple(map(int, s[1:].split("H"))))) or "none"
    m.add("pgListVProm", label_list(pvo["v_promoting"]), f"pg_v_circuit_overlap_{pg}.json v_promoting")
    m.add("pgListVSup", label_list(pvo["v_suppressing"]), f"pg_v_circuit_overlap_{pg}.json v_suppressing")
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
