"""The V condition: J_PV, J_VT and dJ = J_VT - J_PT, read by D75.

Committed before any V-circuit result existed. Run it on the files that
notebooks/v_patching.ipynb saves, plus the committed P and T results:

    .venv/bin/python scripts/analyze_v_circuit.py \\
        results/replication_patching_qwen25vl3b.json \\
        results/t_patching_records_qwen25vl3b.jsonl.gz \\
        results/v_patching_records_qwen25vl3b.jsonl.gz \\
        results/split_half_ceiling_p_qwen25vl3b.json \\
        results/t_patching_qwen25vl3b.json \\
        results/v_circuit_overlap_qwen25vl3b.json

CPU only; a few minutes (three bootstraps and two split-half ceilings).
Every number it prints goes into RESULTS.md as is, and the verdict lines are
D75's fixed phrases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuit_overlap.provenance import result_envelope, write_result  # noqa: E402
from circuit_overlap.t_patching import finite_records, load_patching_records  # noqa: E402
from circuit_overlap.v_patching import analyse_v  # noqa: E402

# D75 (reusing D69/D72). Changing any of these after the V results exist
# breaks the pre-registration.
N_BOOT = 1000
SEED = 0
SIGMA = 2.0
N_SPLITS = 1000
N_PERM = 10_000
CORE_THRESHOLD = 0.9


def ci(pair) -> str:
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def report_pair(name: str, block: dict, first: str, second: str) -> None:
    pr = block["primary"]
    print(f"\n{name} (D75 (7), computed as D72 computes J_PT)")
    print(f"  {first} promoting ({len(block['p_promoting'])}): {', '.join(block['p_promoting'])}")
    print(f"  {second} promoting ({len(block['t_promoting'])}): "
          f"{', '.join(block['t_promoting']) or '(none)'}")
    print(f"  shared ({len(block['shared_promoting'])}): {', '.join(block['shared_promoting']) or '(none)'}")
    print(f"  {name} = {pr['j_pt']:.3f}   bootstrap 95% CI {ci(pr['ci95'])}")
    print(f"  chance floor at sizes {pr['set_sizes']}: mean {pr['chance_floor_mean']:.4f}, "
          f"95th pct {pr['chance_floor_p95_at_observed_sizes']:.4f}   (p = {pr['p_value_vs_chance']:.4f})")
    print(f"  binding ceiling ({block['ceiling_binding']}'s): median {pr['ceiling_median']:.3f}, "
          f"2.5th pct {pr['ceiling_p2_5']:.3f}")
    print(f"  VERDICT: {name} is {pr['wording']}.")
    sc = block["stable_core"]
    print(f"  stable-core J {sc['j_core']:.3f}   suppressing J {block['descriptive']['j_suppressing']:.3f}")
    print("  layer bands   " + "  ".join(f"{b['layers']}: {first}{b['p']} {second}{b['t']} both{b['shared']}"
                                         for b in block["descriptive"]["layer_profile"]))
    so = block["shared_objects"]
    if so:
        flag = "  <- full-sample J lies OUTSIDE this CI" if so["selection_effect_flag"] else ""
        print(f"  shared objects (n={so['n_objects']}): J {so['j_pt']:.3f} {ci(so['ci95'])}{flag}")


def main(p_path: str, t_path: str, v_path: str, ceiling_path: str, t_result_path: str,
         out_path: str | None = None) -> dict:
    p_records, p_bad = finite_records(load_patching_records(p_path))
    t_records, t_bad = finite_records(load_patching_records(t_path))
    v_records, v_bad = finite_records(load_patching_records(v_path))
    p_ceiling = json.loads(Path(ceiling_path).read_text())["result"]
    p_result = json.loads(Path(p_path).read_text())["result"]
    t_result = json.loads(Path(t_result_path).read_text())["result"]

    out = analyse_v(p_records, v_records, t_records, p_ceiling, p_result, t_result,
                    n_boot=N_BOOT, seed=SEED, sigma=SIGMA, n_splits=N_SPLITS, n_perm=N_PERM,
                    core_threshold=CORE_THRESHOLD)

    for tag, path, recs, bad in (("P", p_path, p_records, p_bad), ("T", t_path, t_records, t_bad),
                                 ("V", v_path, v_records, v_bad)):
        print(f"{tag}  {path}: {len(recs)} objects ({len(bad)} non-finite dropped)")
    print(f"V promoting ({len(out['v_promoting'])}): {', '.join(out['v_promoting']) or '(none)'}")
    print(f"V suppressing ({len(out['v_suppressing'])}): {', '.join(out['v_suppressing']) or '(none)'}")

    vc = out["v_ceiling"]
    print("\nV SPLIT-HALF CEILING (D75 (6), D69's parameters)")
    print(f"  halves of {vc['half_sizes']}: promoting J median {vc['jaccard_promoting']['median']:.3f} "
          f"{ci((vc['jaccard_promoting']['p2_5'], vc['jaccard_promoting']['p97_5']))}")
    tc = out["t_ceiling_summary"]["jaccard_promoting"]
    print(f"  (T, for the binding rule: median {tc['median']:.3f} {ci((tc['p2_5'], tc['p97_5']))})")

    report_pair("J_PV", out["j_pv"], "P", "V")
    report_pair("J_VT", out["j_vt"], "V", "T")

    dj = out["delta_j"]
    print("\nPRIMARY TEST OF H1-specific (D75 (8))")
    print(f"  dJ = J_VT - J_PT = {dj['j_vt']:.3f} - {dj['j_pt']:.3f} = {dj['delta_j']:+.3f}   "
          f"bootstrap 95% CI {ci(dj['ci95'])}   ({dj['n_boot']} resamples, T shared per replicate)")
    print(f"  VERDICT: {dj['wording']}.")

    ds = out["descriptive"]
    print("\nDESCRIPTIVE (D75 (9); not part of any verdict)")
    print(f"  containment V in P: promoting {ds['containment_v_in_p_promoting']:.2f}, "
          f"promoting+suppressing {ds['containment_v_in_p_classified']:.2f}   "
          "(Lietzow's 61% is a reference, not our result)")
    li = ds["lietzow_intersection_subset"]
    if li:
        print(f"  on Lietzow's intersection subset (n={li['n_objects']}): V promoting "
              f"{', '.join(li['v_promoting']) or '(none)'}; containment V in P "
              f"{li['containment_v_in_p_promoting']:.2f} / {li['containment_v_in_p_classified']:.2f}")
    for key, name in (("shared_heads_in_v", "shared P-T heads"), ("t_only_heads_in_v", "T-only heads")):
        b = ds[key]
        print(f"  {name} that are V-promoting: {b['n']} of {b['of']}  {', '.join(b['heads'])}")

    if out_path:
        p_env = json.loads(Path(p_path).read_text())
        print("\n", write_result(out_path, result_envelope(
            experiment="v_circuit_overlap", payload=out,
            model=p_env.get("model", "Qwen2.5-VL-3B-Instruct"),
            checkpoint=p_env.get("checkpoint", "Qwen/Qwen2.5-VL-3B-Instruct"),
            seed=SEED, prompt="V: Visual prompt, original vs counterfactual image",
            n=len(v_records), repo_dir=ROOT, decision="D75",
            sources={"p": p_path, "t": t_path, "v": v_path, "p_ceiling": ceiling_path,
                     "t_result": t_result_path},
        )))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    main(*sys.argv[1:7])
