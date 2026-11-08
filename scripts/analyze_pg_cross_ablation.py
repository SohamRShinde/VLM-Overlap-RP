"""PaliGemma H2 cross-ablation, read by D73 as D78 fixes. Written before any run.

    .venv/bin/python scripts/analyze_pg_cross_ablation.py \\
        results/pg_cross_ablation_paligemma2_3b.json \\
        results/pg_cross_ablation_records_paligemma2_3b.jsonl.gz \\
        results/pg_cross_ablation_analysis_paligemma2_3b.json

CPU only. A thin wrapper around Qwen's ``analyze_cross_ablation.main``
(unchanged tests, wording and parameters) that

* rebuilds D78's plan from the committed PaliGemma discovery files and
  refuses a run whose saved plan differs (heads, draws or phases);
* refuses a run the notebook marked incomplete (a phase not finished);
* cross-checks H_P on P against **PaliGemma's own** knockout of the same 8
  heads (``pg_replication_knockout``), never Qwen's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_cross_ablation import main as h2_main  # noqa: E402
from circuit_overlap.cross_ablation import build_plan  # noqa: E402

P_FILE = ROOT / "results" / "pg_replication_patching_paligemma2_3b.json"
T_FILE = ROOT / "results" / "pg_t_patching_paligemma2_3b.json"
KNOCKOUT = ROOT / "results" / "pg_replication_knockout_paligemma2_3b.json"
N_HEADS_PER_LAYER = 8


def expected_plan() -> dict:
    p = json.loads(P_FILE.read_text())["result"]
    t = json.loads(T_FILE.read_text())["result"]
    return build_plan(p, t, n_heads_per_layer=N_HEADS_PER_LAYER)


def main(summary_path: str, records_path: str, out_path: str | None = None) -> dict | None:
    summary = json.loads(Path(summary_path).read_text())["result"]
    if summary["plan"] != json.loads(json.dumps(expected_plan())):
        print("REFUSED: the run's plan differs from D78's plan rebuilt from the committed files")
        return None
    if not summary.get("complete"):
        print(f"REFUSED: the run is incomplete ({summary.get('status')}, clean {summary.get('clean_status')}); "
              "attach its output and run the notebook again")
        return None
    print(f"replay check, worst per-object |dlogit|: {summary.get('replay_check_max')}\n")
    return h2_main(summary_path, records_path, out_path, knockout=KNOCKOUT, decision="D78")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
