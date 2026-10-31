"""The headline: P-T circuit overlap, read by D70 (implementation fixed in D72).

Committed before any T-circuit result existed. Run it on the files that
notebooks/t_patching.ipynb saves:

    .venv/bin/python scripts/analyze_t_circuit.py \\
        results/replication_patching_qwen25vl3b.json \\
        results/t_patching_records_qwen25vl3b.jsonl.gz \\
        results/split_half_ceiling_p_qwen25vl3b.json \\
        results/t_circuit_overlap_qwen25vl3b.json

CPU only, about a minute. Every number it prints goes into RESULTS.md as is,
and the verdict line is one of D70's four fixed phrases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuit_overlap.overlap import analyse_overlap  # noqa: E402
from circuit_overlap.provenance import result_envelope, write_result  # noqa: E402
from circuit_overlap.t_patching import finite_records, load_patching_records  # noqa: E402

# D72. Changing any of these after the T results exist breaks the pre-registration.
N_BOOT = 1000
SEED = 0
SIGMA = 2.0
T_SPLITS = 1000
N_PERM = 10_000
CORE_THRESHOLD = 0.9


def ci(pair) -> str:
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def main(p_path: str, t_path: str, ceiling_path: str, out_path: str | None = None) -> dict:
    p_records, p_bad = finite_records(load_patching_records(p_path))
    t_records, t_bad = finite_records(load_patching_records(t_path))
    ceiling = json.loads(Path(ceiling_path).read_text())["result"]

    out = analyse_overlap(p_records, t_records, ceiling, n_boot=N_BOOT, seed=SEED,
                          sigma=SIGMA, t_splits=T_SPLITS, n_perm=N_PERM,
                          core_threshold=CORE_THRESHOLD)
    pr, sc, rc, ds = out["primary"], out["stable_core"], out["rank_correlation"], out["descriptive"]

    print(f"P  {p_path}: {len(p_records)} objects ({len(p_bad)} non-finite dropped)")
    print(f"T  {t_path}: {len(t_records)} objects ({len(t_bad)} non-finite dropped)")
    print(f"P promoting ({len(out['p_promoting'])}): {', '.join(out['p_promoting'])}")
    print(f"T promoting ({len(out['t_promoting'])}): {', '.join(out['t_promoting']) or '(none)'}")
    print(f"shared     ({len(out['shared_promoting'])}): {', '.join(out['shared_promoting']) or '(none)'}")

    print("\nPRIMARY (D70)")
    print(f"  J_PT = {pr['j_pt']:.3f}   bootstrap 95% CI {ci(pr['ci95'])}   ({N_BOOT} resamples)")
    inside = "" if pr["point_inside_ci"] else "   <- point estimate OUTSIDE its CI"
    print(f"  bootstrap mean {pr['bootstrap_mean']:.3f}, bias {pr['bootstrap_bias']:+.3f}{inside}")
    print(f"  chance floor at sizes {pr['set_sizes']}: mean {pr['chance_floor_mean']:.4f}, "
          f"95th pct {pr['chance_floor_p95_at_observed_sizes']:.4f}   (p = {pr['p_value_vs_chance']:.4f})")
    print(f"  P self-agreement ceiling: median {pr['ceiling_median']:.3f}, 2.5th pct {pr['ceiling_p2_5']:.3f}")
    print(f"\n  VERDICT: J_PT is {pr['wording']}.")

    print("\nSTABLE CORE (heads in >= 90% of each condition's own split-half runs)")
    print(f"  P core ({len(sc['p_core'])}): {', '.join(sc['p_core'])}")
    print(f"  T core ({len(sc['t_core'])}): {', '.join(sc['t_core']) or '(none)'}")
    print(f"  J_core = {sc['j_core']:.3f}")
    t_sa = sc["t_self_agreement_jaccard"]
    print(f"  T's own split-half J: median {t_sa['median']:.3f} {ci((t_sa['p2_5'], t_sa['p97_5']))}")

    print("\nSECONDARY: all-head PC1 rank correlation (landscape similarity, NOT circuit overlap)")
    print(f"  rho_PT = {rc['rho']:.3f}   (no interval: its bootstrap is biased low, D72)")
    print(f"  P ceiling: {rc['p_ceiling_half_length']:.3f} half length, "
          f"{rc['p_ceiling_full_length']:.3f} full length")
    print(f"  attenuation bound sqrt(rel_P * rel_T) = {rc['attenuation_ceiling']:.3f} (descriptive)")

    print("\nDESCRIPTIVE (plan 3.5; not part of the verdict)")
    print(f"  suppressing J {ds['j_suppressing']:.3f}  "
          f"(P {len(ds['p_suppressing'])}, T {len(ds['t_suppressing'])})")
    print(f"  containment: P in T {ds['containment_p_in_t']:.2f}, T in P {ds['containment_t_in_p']:.2f}"
          "   (Lietzow's 61% is V-in-P containment)")
    print(f"  random-normalised J {pr['normalised_j']:.1f}x chance")
    print("  layer bands   " + "  ".join(f"{b['layers']}: P{b['p']} T{b['t']} both{b['shared']}"
                                         for b in ds["layer_profile"]))
    so = out["shared_objects"]
    if so:
        flag = "  <- full-sample J lies OUTSIDE this CI: object selection matters" \
            if so["selection_effect_flag"] else ""
        print(f"  shared objects (n={so['n_objects']}): J {so['j_pt']:.3f} {ci(so['ci95'])}{flag}")

    if out_path:
        payload = {k: v for k, v in out.items()}
        p_env = json.loads(Path(p_path).read_text()) if p_path.endswith(".json") else {}
        print("\n", write_result(out_path, result_envelope(
            experiment="t_circuit_overlap", payload=payload,
            model=p_env.get("model", "Qwen2.5-VL-3B-Instruct"),
            checkpoint=p_env.get("checkpoint", "Qwen/Qwen2.5-VL-3B-Instruct"),
            seed=SEED, prompt="P: Visual vs Prior prompt; T: incongruent vs neutral note, Visual prompt",
            n=len(t_records), repo_dir=ROOT, decision="D70, D72",
            sources={"p": p_path, "t": t_path, "ceiling": ceiling_path},
        )))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:5])
