"""PaliGemma H3 mechanism, read by D74 as D80 fixes. Committed before any PaliGemma H3 run.

    .venv/bin/python scripts/analyze_pg_h3.py \\
        results/pg_h3_attention_paligemma2_3b.json \\
        results/pg_h3_attention_records_paligemma2_3b.jsonl.gz \\
        results/pg_h3_analysis_paligemma2_3b.json

CPU only. A thin wrapper: it reports D80 (7)'s drift rule, then runs Qwen's
committed ``analyze_h3.main`` unchanged with every layer stored (D80 (3)).
The "validation" line it prints (routing:P) is descriptive on PaliGemma
(D80 (8)): Lietzow report that PaliGemma's heads do not shift attention.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_h3 import load_records, main as h3_main  # noqa: E402

MAX_DRIFT = 0.20     # D80 (7)


def drift_report(records: dict) -> dict:
    out = {}
    for cond, recs in records.items():
        n = len(recs)
        drift = sum(r.get("status") == "drift" for r in recs)
        out[cond] = {"n": n, "drift": drift, "fraction": drift / n if n else 0.0,
                     "no_claim": bool(n) and drift / n > MAX_DRIFT}
    return out


def main(summary_path: str, records_path: str, out_path: str | None = None) -> dict:
    summary = json.loads(Path(summary_path).read_text())
    first = int(summary["result"]["layers_stored"][0])
    drift = drift_report(load_records(records_path))
    print(f"D80: PaliGemma H3, layers {summary['result']['layers_stored']} stored")
    for cond, d in drift.items():
        flag = ("  <- over 20%: NO H3 CLAIM for this condition until explained (D80 (7))"
                if d["no_claim"] else "")
        print(f"  {cond}: {d['drift']} of {d['n']} objects drifted under eager attention "
              f"({d['fraction']:.1%}){flag}")
    print("  routing:P below is DESCRIPTIVE on PaliGemma, not a validation (D80 (8))\n")
    out = h3_main(summary_path, records_path, out_path, first_layer=first)
    out["d80_drift"] = drift
    if out_path:      # add the drift report to the file analyze_h3 just wrote
        env = json.loads(Path(out_path).read_text())
        env["result"]["d80_drift"] = drift
        env["decision"] = "D74 as fixed by D80"
        Path(out_path).write_text(json.dumps(env, indent=2) + "\n")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
