"""PaliGemma P-T circuit overlap, read by D70 and computed per D72, as D77 fixes.

    .venv/bin/python scripts/analyze_pg_t_circuit.py \\
        results/pg_replication_patching_paligemma2_3b.json \\
        results/pg_t_patching_records_paligemma2_3b.jsonl.gz \\
        results/split_half_ceiling_p_paligemma2_3b.json \\
        results/pg_t_circuit_overlap_paligemma2_3b.json

CPU only. A thin wrapper: it enforces D77 (3)'s minimum of 60 patched T
objects and then runs Qwen's committed ``analyze_t_circuit.main`` unchanged,
so PaliGemma's J_PT is computed exactly as Qwen's was. Written before any
PaliGemma T patching result existed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_t_circuit import main as overlap_main  # noqa: E402
from circuit_overlap.t_patching import finite_records, load_patching_records  # noqa: E402

MIN_T_OBJECTS = 60   # D77 (3), D76a's minimum


def main(p_path: str, t_path: str, ceiling_path: str, out_path: str | None = None) -> dict | None:
    t_records, _ = finite_records(load_patching_records(t_path))
    print(f"D77: {len(t_records)} PaliGemma T objects patched (minimum {MIN_T_OBJECTS})\n")
    if len(t_records) < MIN_T_OBJECTS:
        print(f"fewer than {MIN_T_OBJECTS}: no D70 verdict (D77 (3))")
        return None
    return overlap_main(p_path, t_path, ceiling_path, out_path)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:5])
