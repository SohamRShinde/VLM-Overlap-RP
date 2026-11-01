"""H3 mechanism, read by D74. Committed before any H3 run existed.

Run it on the two files that notebooks/h3_attention.ipynb saves:

    .venv/bin/python scripts/analyze_h3.py \\
        results/h3_attention_qwen25vl3b.json \\
        results/h3_attention_records_qwen25vl3b.jsonl.gz \\
        results/h3_analysis_qwen25vl3b.json

CPU only. Every number it prints goes into RESULTS.md as is, and each
verdict line is one of D74's fixed phrases. The head sets and random draws
come from the plan saved in the summary file (D73's, rebuilt by the notebook).
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuit_overlap.h3 import analyse_h3  # noqa: E402
from circuit_overlap.provenance import result_envelope, write_result  # noqa: E402

# D74. Changing either after the H3 results exist breaks the pre-registration.
N_BOOT = 1000
SEED = 0


def load_records(path: str) -> dict[str, list]:
    opener = gzip.open if path.endswith(".gz") else open
    out: dict[str, list] = {"P": [], "T": []}
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["condition"]].append(r)
    return out


def line(name: str, t: dict | None) -> str:
    if t is None:
        return f"  {name:22s} (not run)"
    bound = "p95" if t["direction"] == "greater" else "p5"
    return (f"  {name:22s} {t['set']:6s} {t['measure']:8s} n={t['n']:<3d} "
            f"{t['observed']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]   random "
            f"{t['null_mean']:+.4f} ({bound} {t[f'null_{bound}']:+.4f})   "
            f"excess {t['excess']:+.4f} [{t['excess_ci95'][0]:+.4f}, {t['excess_ci95'][1]:+.4f}]   "
            f"p = {t['p']:.4f}")


def main(summary_path: str, records_path: str, out_path: str | None = None) -> dict:
    summary = json.loads(Path(summary_path).read_text())
    plan = summary["result"]["plan"]
    out = analyse_h3(plan, load_records(records_path), n_boot=N_BOOT, seed=SEED)

    print(f"objects: P {out['n_objects']['P']} (dropped {out['n_dropped']['P']}), "
          f"T {out['n_objects']['T']} (dropped {out['n_dropped']['T']}); "
          f"dropped as {out['dropped_status']}")
    for k, v in plan["sets"].items():
        print(f"  {k:7s} ({len(v):2d}): {', '.join(v)}")
    print(f"\nMasses are summed last-token attention, mean over the set's heads; dla is the "
          f"set's summed direct effect (logits, + = toward the competing answer). "
          f"alpha = {out['alpha']}, {plan['n_draws']} layer-matched draws (D73's).")

    for tier in ("primary", "secondary", "validation"):
        print(f"\n{tier.upper()} (D74)")
        for name, t in out[tier].items():
            print(line(name, t))
        for name, t in out[tier].items():
            if t:
                print(f"  VERDICT {name}: {t['wording']}.")
    print(f"\nMECHANISM (D74 (6)): {out['mechanism']}.")

    d = out["descriptive"]
    if "note_enrichment" in d:
        print(f"\nDESCRIPTIVE. Note's share of image tokens {d['note_token_share']:.4f}; enrichment "
              "(share of image attention on the note / that):")
        for k, v in d["note_enrichment"].items():
            print(f"  {k:40s} {v:.2f}" if v is not None else f"  {k:40s} n/a")
        print("  spatial (T-only note mass; fool rate on this path):")
        for layout, s in d["spatial"].items():
            f = s["fooled_minus_not_incongruent"]
            diff = (f"fooled - not {f['diff']:+.4f} [{f['ci95'][0]:+.4f}, {f['ci95'][1]:+.4f}] "
                    f"(n {f['n'][0]}/{f['n'][1]})") if "diff" in f else f"fooled - not n/a (n {f['n']})"
            print(f"  {layout:17s} fool {s['fool_rate']:.1%}  note tokens {s['note_tokens_mean']:.1f} "
                  f"({s['n_objects_without_note_tokens']} with none)  incongruent "
                  f"{s['T_only_note_mass_incongruent']:.4f}  neutral {s['T_only_note_mass_neutral']:.4f}  {diff}")
    for c in ("T", "P"):
        k = f"eager_vs_discovery_max_abs_logit_{c}"
        if k in d:
            print(f"  eager attention vs discovery, {c}: max |dlogit| {d[k]:.3f}")

    if out_path:
        print("\n", write_result(out_path, result_envelope(
            experiment="h3_analysis", payload=out,
            model=summary.get("model", "Qwen2.5-VL-3B-Instruct"),
            checkpoint=summary.get("checkpoint", "Qwen/Qwen2.5-VL-3B-Instruct"),
            seed=SEED, prompt=summary.get("prompt", ""),
            n=out["n_objects"]["P"] + out["n_objects"]["T"], repo_dir=ROOT, decision="D74",
            sources={"summary": summary_path, "records": records_path,
                     "run_commit": summary.get("commit")},
        )))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
