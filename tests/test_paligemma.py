"""CPU pieces of the PaliGemma port (D76) and its analysis script."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from circuit_overlap.paligemma import CHECKPOINT, square_image_block

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_is_d11s():
    assert CHECKPOINT == json.loads((ROOT / "configs" / "heads_lietzow_paligemma2_3b.json").read_text())["checkpoint"]


def test_square_image_block_finds_offset_and_side():
    ids = [7] * 4 + [1] * 9 + [2, 3, 4]
    assert square_image_block(ids, image_token_id=1) == (4, 3)


@pytest.mark.parametrize("ids, msg", [
    ([2, 3], "no image tokens"),
    ([1, 1, 2, 1, 1], "not contiguous"),
    ([1] * 8 + [2], "not a square"),
])
def test_square_image_block_rejects_bad_layouts(ids, msg):
    with pytest.raises(ValueError, match=msg):
        square_image_block(ids, image_token_id=1)


def _envelope(result, commit="test"):
    return {"commit": commit, "result": result}


def test_selection_counts_recompute_from_records(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_pg_replication as pg

    color_ids = {"red": [1, 2], "blue": [3, 4], "green": [5, 6]}
    recs = []
    for i in range(60):
        p_ok, t_ok = i < 30, i % 2 == 0
        recs.append({"i": i, "name": f"o{i}", "status": "ok", "correct": ["red"], "attack": "blue",
                     "top1_visual": 3 if p_ok else 1, "top1_prior": 1,
                     "top1_neutral": 2, "top1_incongruent": 4 if t_ok else 2,
                     "p_conflicting": p_ok, "t_conflicting": t_ok})
    recs.append({"i": 60, "name": "x", "status": "numerics_failed", "correct": ["red"], "attack": "blue"})
    # an object using a colliding colour never counts (D26)
    recs.append({"i": 61, "name": "y", "status": "ok", "correct": ["green"], "attack": "blue",
                 "top1_visual": 3, "top1_prior": 5, "top1_neutral": 5, "top1_incongruent": 3,
                 "p_conflicting": True, "t_conflicting": True})
    f = tmp_path / "sel.json"
    f.write_text(json.dumps(_envelope({"color_ids": color_ids, "ambiguous_colours": ["green"], "records": recs})))
    out = pg.selection_counts(f)
    assert (out["n_p"], out["n_t"]) == (30, 30)
    assert out["t_viable"] is False                 # 30 < 50
    assert 61 not in out["p_ids"] and 60 not in out["t_ids"]


def test_gate_script_runs_on_the_qwen_files_in_d76_mode():
    """Format check only: the Qwen files are the same shape the notebook writes."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_pg_replication.py"),
                        "missing_selection.json",
                        str(ROOT / "results" / "replication_knockout_qwen25vl3b.json"),
                        str(ROOT / "results" / "replication_patching_qwen25vl3b.json")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "D76 criteria" in r.stdout and "L15H7 and L20H3" in r.stdout
    assert "D76 verdict:" in r.stdout


def test_run_cached_budget_stops_and_resumes(tmp_path):
    from circuit_overlap.replication import run_cached

    rows = [{"i": i} for i in range(5)]
    calls = []

    def slow(r):
        calls.append(r["i"])
        import time
        time.sleep(0.05)
        return {"v": r["i"] * 2}

    first = run_cached(tmp_path / "c.jsonl", rows, slow, progress_every=0, budget_s=0.07)
    assert 1 <= len(first) < 5 and [r["i"] for r in first] == calls     # stopped early, in order
    rest = run_cached(tmp_path / "c.jsonl", rows, slow, progress_every=0)
    assert [r["i"] for r in rest] == list(range(5)) and sorted(calls) == list(range(5))   # resumed, no repeats


def test_run_cached_without_budget_is_unchanged(tmp_path):
    from circuit_overlap.replication import run_cached

    out = run_cached(tmp_path / "c.jsonl", [{"i": 1}, {"i": 2}], lambda r: {"v": 1}, progress_every=0)
    assert out == [{"i": 1, "v": 1}, {"i": 2, "v": 1}]


def test_quiet_processor_warning_drops_only_that_message(caplog):
    import logging

    from circuit_overlap.paligemma import quiet_processor_warning

    quiet_processor_warning()
    log = logging.getLogger("transformers.models.paligemma.processing_paligemma")
    with caplog.at_level(logging.WARNING):
        log.warning("You are passing both `text` and `images` to `PaliGemmaProcessor`. blah")
        log.warning("something else")
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs == ["something else"]


def test_too_few_patched_objects_leave_c_to_e_not_run(tmp_path, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_pg_replication as pg

    qwen = json.loads((ROOT / "results" / "replication_patching_qwen25vl3b.json").read_text())
    qwen["result"]["records"] = qwen["result"]["records"][:30]
    f = tmp_path / "patch.json"
    f.write_text(json.dumps(qwen))
    out = pg.main(None, None, str(f))
    assert out["gate"]["criteria"]["c"] is None and "not run" in capsys.readouterr().out
