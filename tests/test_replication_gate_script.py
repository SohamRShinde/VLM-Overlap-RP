"""The D40 gate script decides pass/fail, so it is tested on synthetic files."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
THEIRS = [[26, 0], [26, 5], [26, 6], [27, 1], [27, 4], [28, 3], [31, 3], [31, 7], [34, 14]]


def knockout_file(path: Path, prior_flips: int, visual_flips: int, n: int = 10) -> None:
    configs = {"grp_promoting": THEIRS, "grp_suppressing": [[30, 3]]}
    configs.update({f"ind_L{l}H{h}": [[l, h]] for l, h in THEIRS})
    records = []
    for k in range(n):
        rec = {"i": k, "none/prior": [5.0, 1.0], "none/visual": [1.0, 5.0],
               "grp_suppressing/prior": [5.0, 1.0], "grp_suppressing/visual": [1.0, 5.0]}
        rec["grp_promoting/prior"] = [1.0, 5.0] if k < prior_flips else [5.0, 1.0]
        rec["grp_promoting/visual"] = [5.0, 1.0] if k < visual_flips else [1.0, 5.0]
        for l, h in THEIRS:
            # L26H5 flips 8 objects, L31H3 7, every other head 1.
            flips = {(26, 5): 8, (31, 3): 7}.get((l, h), 1)
            rec[f"ind_L{l}H{h}/prior"] = [1.0, 5.0] if k < flips else [5.0, 1.0]
            rec[f"ind_L{l}H{h}/visual"] = [1.0, 5.0]
        records.append(rec)
    path.write_text(json.dumps({"commit": "abc", "result": {"configs": configs, "records": records}}))


def patching_file(path: Path, planted) -> None:
    rng = np.random.default_rng(0)
    records = []
    for _ in range(5):
        p2v_patched = rng.normal(0, 0.05, (36, 16))       # restoration ~0 everywhere
        for l, h in planted:
            p2v_patched[l, h] = 0.6
        # Delta_prior_run = 4, Delta_visual_run = -3: P2V score r gives Delta = -3 + 7r,
        # V2P score r gives Delta = 4 - 7r. Store logit_prior = Delta, logit_visual = 0.
        records.append({
            "prior_token_logit_prior_run": 4.0, "visual_token_logit_prior_run": 0.0,
            "prior_token_logit_visual_run": -3.0, "visual_token_logit_visual_run": 0.0,
            "p2v_logit_prior": (-3 + 7 * p2v_patched).tolist(), "p2v_logit_visual": np.zeros((36, 16)).tolist(),
            "v2p_logit_prior": (4 - 7 * p2v_patched).tolist(), "v2p_logit_visual": np.zeros((36, 16)).tolist(),
        })
    path.write_text(json.dumps({"commit": "abc", "result": {
        "records": records, "sigma": 2.0, "reference_promoting": THEIRS,
        "promoting": [list(h) for h in sorted(map(tuple, planted))]}}))


def run(*paths):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_replication_gate.py"),
                           *map(str, paths)], capture_output=True, text=True, check=True, cwd=ROOT).stdout


def test_gate_passes_when_every_criterion_holds(tmp_path):
    knockout_file(tmp_path / "k.json", prior_flips=9, visual_flips=0)
    patching_file(tmp_path / "p.json", planted=THEIRS)
    out = run(tmp_path / "k.json", tmp_path / "p.json")
    assert "D40 verdict: PASS" in out, out
    assert "their 9 found: 9" in out


def test_gate_fails_on_a_low_prior_flip_and_reports_missing_files(tmp_path):
    knockout_file(tmp_path / "k.json", prior_flips=5, visual_flips=0)
    out = run(tmp_path / "k.json", tmp_path / "missing.json")
    assert "missing:" in out and "incomplete" in out
    patching_file(tmp_path / "p.json", planted=THEIRS)
    out = run(tmp_path / "k.json", tmp_path / "p.json")
    assert "(a) group prior flip" in out and "FAIL -- stop and debug" in out


def test_gate_is_partial_when_rediscovery_overlaps_only_partly(tmp_path):
    knockout_file(tmp_path / "k.json", prior_flips=9, visual_flips=1)
    planted = THEIRS[:3] + [[20, 1], [21, 2], [22, 3], [23, 4], [24, 5], [25, 6]]
    patching_file(tmp_path / "p.json", planted=planted)
    out = run(tmp_path / "k.json", tmp_path / "p.json")
    assert "their 9 found: 3" in out
    assert "D40 verdict: PARTIAL" in out, out
