"""PaliGemma V condition: J_PV, J_VT and dJ = J_VT - J_PT, read by D75 as D79 fixes.

    .venv/bin/python scripts/analyze_pg_v_circuit.py \\
        results/pg_v_patching_records_paligemma2_3b.jsonl.gz \\
        results/pg_v_circuit_overlap_paligemma2_3b.json

CPU only. A thin wrapper: it enforces D79 (4)'s minimum of 60 patched V
objects, maps V's soft-capped logits back to uncapped ones (D79b), and then runs Qwen's committed ``analyze_v_circuit.main`` unchanged on
PaliGemma's P, T and P-ceiling files, so PaliGemma's dJ is computed exactly as
Qwen's was. It then repeats the analysis on the logits as recorded and
says whether anything differs (D79b (1)). Written before any PaliGemma V
result existed.
"""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_v_circuit import main as v_main  # noqa: E402
from circuit_overlap.t_patching import finite_records, load_patching_records  # noqa: E402
from circuit_overlap.v_patching import PG_HEAD_SOFTCAP, PG_V_CAP, uncap_record  # noqa: E402

MIN_V_OBJECTS = 60   # D79 (4), D77 (3)'s minimum

# D79 (1): PaliGemma's committed P, T and P-ceiling files.
P_FILE = ROOT / "results" / "pg_replication_patching_paligemma2_3b.json"
T_RECORDS = ROOT / "results" / "pg_t_patching_records_paligemma2_3b.jsonl.gz"
P_CEILING = ROOT / "results" / "split_half_ceiling_p_paligemma2_3b.json"
T_RESULT = ROOT / "results" / "pg_t_patching_paligemma2_3b.json"


def main(v_path: str, out_path: str | None = None) -> dict | None:
    records = load_patching_records(v_path)
    v_records, _ = finite_records([r for r in records if r.get("status", "patched") == "patched"])
    fallback = sum(r.get("method") == "batched_fallback" for r in records)
    checks = [r["replay_check"] for r in records if "replay_check" in r]
    print(f"D79: {len(v_records)} PaliGemma V objects patched (minimum {MIN_V_OBJECTS}); "
          f"{fallback} by the batched fallback; worst replay check "
          f"{max(checks) if checks else float('nan'):.4f}\n")
    if len(v_records) < MIN_V_OBJECTS:
        print(f"fewer than {MIN_V_OBJECTS}: no D75 verdict (D79 (4))")
        return None
    print(f"D79b: V logits mapped back from the head's soft-cap at {PG_HEAD_SOFTCAP:g} "
          "(P and T were recorded without it)\n")
    with tempfile.TemporaryDirectory() as tmp:
        uncapped = Path(tmp) / "v_uncapped.jsonl.gz"
        with gzip.open(uncapped, "wt") as f:
            for r in records:
                f.write(json.dumps(uncap_record(r) if r.get("status", "patched") == "patched" else r) + "\n")
        out = v_main(str(P_FILE), str(T_RECORDS), str(uncapped), str(P_CEILING), str(T_RESULT),
                     out_path, cap=PG_V_CAP)
    with contextlib.redirect_stdout(io.StringIO()):
        raw = v_main(str(P_FILE), str(T_RECORDS), v_path, str(P_CEILING), str(T_RESULT), None, cap=PG_V_CAP)
    same = (raw["v_promoting"] == out["v_promoting"] and raw["v_suppressing"] == out["v_suppressing"]
            and raw["delta_j"]["verdict"] == out["delta_j"]["verdict"])
    print(f"\nD79b check, logits as recorded (capped): V promoting {', '.join(raw['v_promoting'])}; "
          f"dJ {raw['delta_j']['delta_j']:+.3f} [{raw['delta_j']['ci95'][0]:+.3f}, {raw['delta_j']['ci95'][1]:+.3f}] -> {raw['delta_j']['wording']}")
    print("  same head sets and dJ verdict as the primary analysis" if same
          else "  DIFFERS from the primary analysis: report both")
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:3])
