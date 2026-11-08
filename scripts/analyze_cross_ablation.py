"""H2 cross-ablation, read by D73. Committed before any H2 run existed.

Run it on the two files that notebooks/cross_ablation.ipynb saves:

    .venv/bin/python scripts/analyze_cross_ablation.py \\
        results/cross_ablation_qwen25vl3b.json \\
        results/cross_ablation_records_qwen25vl3b.jsonl.gz \\
        results/cross_ablation_analysis_qwen25vl3b.json

CPU only. Every number it prints goes into RESULTS.md as is, and each
verdict line is one of D73's two fixed phrases. If
results/replication_knockout_qwen25vl3b.json is present, the P records are
also checked against Kush's knockout of the same 9 heads.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from circuit_overlap.cross_ablation import analyse_cross_ablation, source_flips, target_flips  # noqa: E402
from circuit_overlap.provenance import result_envelope, write_result  # noqa: E402

# D73. Changing either after the H2 results exist breaks the pre-registration.
N_BOOT = 1000
SEED = 0
KNOCKOUT = ROOT / "results" / "replication_knockout_qwen25vl3b.json"


def load_records(path: str) -> dict[str, list]:
    opener = gzip.open if path.endswith(".gz") else open
    out: dict[str, list] = {"P": [], "T": [], "clean": []}
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["condition"]].append(r)
    return out


def knockout_cross_check(p_records: list, knockout: Path = KNOCKOUT) -> dict | None:
    """Our H_P zero ablation on P against Kush's grp_promoting knockout.

    Same 9 heads (D40 found exactly Lietzow's set), same objects, same
    ablation site; his run used a full-forward baseline, ours the cached path.
    """
    if knockout is None or not Path(knockout).exists():
        return None
    knock = {r["i"]: r for r in json.loads(Path(knockout).read_text())["result"]["records"]}
    ours = [r for r in p_records if r.get("status") == "ok" and r["i"] in knock
            and (r.get("a_prior"), r.get("a_visual")) == (knock[r["i"]]["a_prior"], knock[r["i"]]["a_visual"])]
    if not ours:
        return None
    theirs = [knock[r["i"]] for r in ours]
    diff = max(abs(a - b) for r, k in zip(ours, theirs) for run in ("prior", "visual")
               for a, b in zip(r[f"P/{run}"], k[f"grp_promoting/{run}"]))
    return {
        "n_compared": len(ours),
        "max_abs_logit_diff": float(diff),
        "source_flips_ours_vs_kush": [int(source_flips(ours, "P").sum()),
                                      int(source_flips(theirs, "grp_promoting").sum())],
        "target_flips_ours_vs_kush": [int(target_flips(ours, "P").sum()),
                                      int(target_flips(theirs, "grp_promoting").sum())],
        "flip_disagreements": int(np.sum(source_flips(ours, "P") != source_flips(theirs, "grp_promoting"))),
    }


def line(name: str, t: dict | None) -> str:
    if t is None:
        return f"  {name:10s} (not run)"
    return (f"  {name:10s} {t['count']:3d}/{t['n']:<3d} = {t['observed']:6.1%} "
            f"[{t['ci95'][0]:.1%}, {t['ci95'][1]:.1%}]   random {t['null_mean']:5.1%} "
            f"(p95 {t['null_p95']:5.1%}, max {t['null_max']:5.1%})   "
            f"excess {t['excess']:+.1%} [{t['excess_ci95'][0]:+.1%}, {t['excess_ci95'][1]:+.1%}]   "
            f"p = {t['p']:.4f}")


def main(summary_path: str, records_path: str, out_path: str | None = None,
         knockout: Path | None = KNOCKOUT, decision: str = "D73") -> dict:
    """``knockout``: the same model's knockout file, whose ``grp_promoting`` is
    the same head set as H_P (Qwen's by default; never mix models)."""
    summary = json.loads(Path(summary_path).read_text())
    plan = summary["result"]["plan"]
    recs = load_records(records_path)
    out = analyse_cross_ablation(plan, {"P": recs["P"], "T": recs["T"]}, recs["clean"],
                                 n_boot=N_BOOT, seed=SEED)
    out["knockout_cross_check"] = knockout_cross_check(recs["P"], knockout)

    print(f"objects: P {out['n_objects']['P']} (dropped {out['n_dropped']['P']}), "
          f"T {out['n_objects']['T']} (dropped {out['n_dropped']['T']})")
    for k, v in plan["sets"].items():
        print(f"  {k:7s} ({len(v):2d}): {', '.join(v)}")
    print(f"\nSource flips = the competing signal stops winning (D67 pairwise; D73). "
          f"alpha = {out['alpha']}, {plan['n_draws']} layer-matched draws.")

    print("\nPRIMARY (D73 (6))")
    for name, t in out["primary"].items():
        print(line(name, t))
    for name, t in out["primary"].items():
        if t:
            print(f"  VERDICT {name}: {t['wording']}.")

    print("\nSECONDARY: diagonal (in-sample)")
    for name, t in out["diagonal"].items():
        print(line(name, t))
    print("\nSECONDARY: decomposition")
    for name, t in out["decomposition"].items():
        print(line(name, t))
    for name, t in out["decomposition"].items():
        if t:
            print(f"  {name}: {t['wording']}")
    for name, o in out["own_condition_subsets"].items():
        print(f"  descriptive {name}: {o['count']}/{o['n']} = {o['observed']:.1%} (no null)")
    for name, r in out["transfer_ratio"].items():
        print(f"  transfer ratio {name} / own-circuit diagonal: {r:.2f}")

    print("\nROBUSTNESS: mean ablation (D73 (10))")
    for name, t in out["mean_ablation"]["tests"].items():
        print(line(name, t))
    print(f"  same wording as zero ablation: {out['mean_ablation']['same_wording_as_zero']}")

    print("\nCOLLATERAL (descriptive)")
    for cond, c in out["collateral_target_flips"].items():
        cells = "  ".join(f"{s} {v['target_flip_rate']:.1%}" for s, v in c.items())
        any_v = next(iter(c.values()))
        rnd = ", ".join(f"{p} {any_v[f'null_mean_{p}']:.1%}" for p in ("rP", "rT")
                        if any_v[f"null_mean_{p}"] is not None)
        print(f"  {cond} target-run flips: {cells}   (random: {rnd})")
    cl = out["clean_accuracy"]
    if cl:
        kept = "  ".join(f"{k} {v:.1%}" for k, v in cl["accuracy_kept"].items())
        print(f"  clean image, {cl['n_correct_unablated']} of {cl['n_objects']} correct unablated; "
              f"kept: {kept}   (random rP {cl['random_mean_rP']:.1%}, rT {cl['random_mean_rT']:.1%})")
    kc = out["knockout_cross_check"]
    if kc:
        print(f"\nCHECK vs the knockout of the same {len(plan['sets']['P'])} heads on P "
              f"({Path(knockout).name}, {kc['n_compared']} objects): "
              f"max |dlogit| {kc['max_abs_logit_diff']:.3f}, source flips ours/his "
              f"{kc['source_flips_ours_vs_kush']}, disagreements {kc['flip_disagreements']}")

    if out_path:
        print("\n", write_result(out_path, result_envelope(
            experiment="cross_ablation_analysis", payload=out,
            model=summary.get("model", "Qwen2.5-VL-3B-Instruct"),
            checkpoint=summary.get("checkpoint", "Qwen/Qwen2.5-VL-3B-Instruct"),
            seed=SEED, prompt=summary.get("prompt", ""),
            n=out["n_objects"]["P"] + out["n_objects"]["T"], repo_dir=ROOT, decision=decision,
            sources={"summary": summary_path, "records": records_path,
                     "run_commit": summary.get("commit")},
        )))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
