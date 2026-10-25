"""The replication analysis script feeds RESULTS.md, so it must not rot."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_replication import fisher_exact_two_sided, wilson_interval  # noqa: E402


def test_wilson_interval_hand_checked():
    lo, hi = wilson_interval(5, 73)
    assert abs(lo - 0.0296) < 1e-3 and abs(hi - 0.1508) < 1e-3
    assert wilson_interval(0, 10)[0] == 0.0


def test_fisher_exact_two_sided():
    # 5/73 vs 1/73: hypergeometric sum computed independently, p = 0.209.
    assert abs(fisher_exact_two_sided(5, 73, 1, 73) - 0.209) < 1e-3
    assert fisher_exact_two_sided(3, 20, 3, 20) == 1.0
    assert fisher_exact_two_sided(10, 10, 0, 10) < 1e-4


def test_script_runs_on_a_result_file(tmp_path):
    path = tmp_path / "res.json"
    path.write_text(json.dumps({
        "commit": "abc1234", "model": "test", "quantisation": "4-bit",
        "compute_dtype": "float16",
        "result": {
            "prior_flip": 66 / 73, "visual_flip": 5 / 73,
            "n_correctly_conflicting": 73, "n_top1_differs": 105,
            "n_objects": 469, "dropped": {"name": 22, "overlap": 2},
            "heads_ablated": [[26, 5], [31, 3]],
            "reported": {"prior_flip": 0.849, "visual_flip": 0.014,
                         "n": 73, "n_objects": 467},
        },
    }))
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analyze_replication.py"), str(path)],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout
    assert "prior  flip  66/73 = 90.4%" in out
    assert "visual flip  5/73 = 6.8%" in out
    assert "Fisher p = 0.21" in out
