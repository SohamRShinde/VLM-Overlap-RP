"""CPU tests for the V condition (D75): selection, order, and the analysis.

The analysis is checked on known-answer synthetic conditions, as D72 checked
the J_PT code: a V that equals T's circuit must read dJ > 0, a V that equals
P's must read dJ < 0, and a V between them must read no difference.
"""

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from circuit_overlap.ceiling import split_half_ceiling
from circuit_overlap.v_patching import (
    DJ_WORDING,
    V_CAP,
    V_ROLES,
    V_VERDICTS,
    analyse_v,
    binding_ceiling,
    delta_j_bootstrap,
    dj_verdict,
    is_v_conflicting,
    v_sweep_order,
)

ROOT = Path(__file__).resolve().parents[1]
COLOR_IDS = {"red": [1, 2], "blue": [3, 4], "white": [5, 6], "brown": [7, 8]}
ROW = {"i": 0, "name": "strawberry", "correct": ["red"], "attack": "blue"}


# --- selection and order ---------------------------------------------------


def test_selection_needs_both_images_answered_from_the_image():
    assert is_v_conflicting(ROW, top1_original=1, top1_counterfactual=4, color_ids=COLOR_IDS)
    # original image answered with the counterfactual colour: not V-correct
    assert not is_v_conflicting(ROW, top1_original=3, top1_counterfactual=4, color_ids=COLOR_IDS)
    # counterfactual image answered with the real colour (prior wins): not V-correct
    assert not is_v_conflicting(ROW, top1_original=1, top1_counterfactual=2, color_ids=COLOR_IDS)
    # a third colour on either side
    assert not is_v_conflicting(ROW, top1_original=5, top1_counterfactual=4, color_ids=COLOR_IDS)


def test_selection_excludes_ambiguous_colours():
    assert not is_v_conflicting(ROW, 1, 4, COLOR_IDS, ambiguous={"blue"})
    assert not is_v_conflicting(ROW, 1, 4, COLOR_IDS, ambiguous={"red"})
    assert is_v_conflicting(ROW, 1, 4, COLOR_IDS, ambiguous={"white"})


def test_sweep_order_puts_p_objects_first_and_caps():
    selected = list(range(400))
    p_ids = [5, 17, 399, 1000]                    # 1000 was not V-selected
    order = v_sweep_order(selected, p_ids, seed=0)
    assert len(order) == V_CAP == 244
    assert order[:3] == [5, 17, 399]
    assert len(set(order)) == len(order)
    # the rest is a seeded shuffle, not the first ids
    assert order[3:13] != list(range(10))
    assert v_sweep_order(selected, p_ids, seed=0) == order
    assert v_sweep_order(selected, p_ids, cap=10) == order[:10]
    with pytest.raises(ValueError):
        v_sweep_order(selected, p_ids, cap=0)


def test_roles_name_the_lietzow_contrast():
    assert "original" in V_ROLES["source"] and "_prior" in V_ROLES["source"]
    assert "counterfactual" in V_ROLES["target"] and "_visual" in V_ROLES["target"]


# --- reading rules -----------------------------------------------------------


def test_dj_verdict_three_branches_and_wording():
    assert dj_verdict(0.01, 0.2) == "greater"
    assert dj_verdict(-0.2, -0.01) == "less"
    assert dj_verdict(-0.1, 0.1) == "none"
    assert dj_verdict(0.0, 0.1) == "none"          # touching 0 is not above it
    with pytest.raises(ValueError):
        dj_verdict(0.2, 0.1)
    assert DJ_WORDING["none"] == "no difference between J_VT and J_PT detectable at this n"
    assert V_VERDICTS["ceiling"] == "not distinguishable from self-agreement"
    assert V_VERDICTS["between"] == "above chance, below self-agreement"


def test_binding_ceiling_takes_the_lower_p2_5():
    a = {"jaccard_promoting": {"median": 0.9, "p2_5": 0.9}, "selection_frequency_promoting": {"L1H1": 1.0}}
    b = {"jaccard_promoting": {"median": 0.8, "p2_5": 0.6}, "selection_frequency_promoting": {"L2H2": 1.0}}
    out = binding_ceiling(a, b)
    assert out["jaccard_promoting"]["p2_5"] == 0.6 and out["binding"] == "second"
    assert out["selection_frequency_promoting"] == {"L1H1": 1.0}   # still the first's
    assert a["jaccard_promoting"]["p2_5"] == 0.9                   # input untouched
    assert binding_ceiling(b, a)["binding"] == "first"


# --- synthetic conditions ----------------------------------------------------


def make_record(i, p2v, v2p):
    p2v, v2p = np.asarray(p2v, float), np.asarray(v2p, float)
    return {
        "i": i, "status": "patched",
        "prior_token_logit_prior_run": 1.0, "visual_token_logit_prior_run": 0.0,
        "prior_token_logit_visual_run": 0.0, "visual_token_logit_visual_run": 0.0,
        "p2v_logit_prior": p2v.tolist(), "p2v_logit_visual": np.zeros_like(p2v).tolist(),
        "v2p_logit_prior": (1.0 - v2p).tolist(), "v2p_logit_visual": np.zeros_like(v2p).tolist(),
    }


def condition(circuit, n=24, ids=None, seed=0, noise=0.02, shape=(12, 8)):
    rng = np.random.default_rng(seed)
    ids = list(range(n)) if ids is None else ids
    out = []
    for i in ids:
        p2v, v2p = rng.normal(0, noise, shape), rng.normal(0, noise, shape)
        for l, h in circuit:
            p2v[l, h] += 0.6
            v2p[l, h] += 0.6
        out.append(make_record(i, p2v, v2p))
    return out


SHARED = [(8, 1), (9, 6)]
P_ONLY = [(4, 2), (5, 3)]
T_ONLY = [(10, 2), (11, 7)]
P_CIRCUIT, T_CIRCUIT = SHARED + P_ONLY, SHARED + T_ONLY


def as_result(circuit):
    return {"promoting": [list(h) for h in circuit], "suppressing": []}


def run_v(v_circuit, **kw):
    p = condition(P_CIRCUIT, seed=0)
    t = condition(T_CIRCUIT, ids=list(range(100, 124)), seed=1)
    v = condition(v_circuit, ids=list(range(12, 36)), seed=2)   # 12 objects shared with P
    ceil = split_half_ceiling(p, n_splits=20, seed=0, n_perm=200)
    return analyse_v(p, v, t, ceil, as_result(P_CIRCUIT), as_result(T_CIRCUIT),
                     n_boot=40, n_splits=20, n_perm=200, **kw)


def test_v_equal_to_t_reads_dj_greater():
    out = run_v(T_CIRCUIT)
    dj = out["delta_j"]
    assert dj["j_vt"] == pytest.approx(1.0) and dj["j_pt"] == pytest.approx(2 / 6)
    assert dj["verdict"] == "greater" and dj["wording"] == "J_VT > J_PT"
    assert out["j_vt"]["primary"]["verdict"] == "ceiling"
    assert out["j_vt"]["primary"]["wording"] == "not distinguishable from self-agreement"
    assert out["descriptive"]["t_only_heads_in_v"]["n"] == 2
    assert out["descriptive"]["shared_heads_in_v"]["n"] == 2


def test_v_equal_to_p_reads_no_difference():
    # V = P makes J_VT = J_PT by construction, so dJ is exactly 0.
    out = run_v(P_CIRCUIT)
    dj = out["delta_j"]
    assert dj["delta_j"] == pytest.approx(0.0)
    assert dj["verdict"] == "none"
    assert out["j_pv"]["primary"]["j_pt"] == pytest.approx(1.0)
    assert out["descriptive"]["containment_v_in_p_promoting"] == pytest.approx(1.0)
    assert out["descriptive"]["t_only_heads_in_v"]["n"] == 0


def test_v_disjoint_from_t_reads_dj_less():
    # V shares nothing with T: J_VT = 0 against J_PT = 2/6.
    out = run_v(P_ONLY + [(2, 0), (3, 3)])
    dj = out["delta_j"]
    assert dj["j_vt"] == 0.0 and dj["delta_j"] == pytest.approx(-2 / 6)
    assert dj["verdict"] == "less" and dj["wording"] == "J_VT < J_PT"
    assert out["j_vt"]["primary"]["verdict"] == "chance"


def test_intersection_subset_and_shared_object_check_use_p_objects():
    out = run_v(T_CIRCUIT)
    li = out["descriptive"]["lietzow_intersection_subset"]
    assert li["n_objects"] == 12
    assert out["j_pv"]["shared_objects"]["n_objects"] == 12
    assert out["j_vt"]["shared_objects"] is None          # V and T share no objects here


def test_t_is_resampled_once_per_replicate():
    """With V identical to T record-for-record, sharing T's resample makes
    J_VT exactly 1 in every replicate; independent draws would not."""
    p = condition(P_CIRCUIT, seed=0)
    t = condition(T_CIRCUIT, ids=list(range(100, 124)), seed=1, noise=0.2)
    out = delta_j_bootstrap(p, t, t, n_boot=30, seed=0)
    assert out["j_vt"] == 1.0
    # J_VT is not stored per replicate, so check through dJ: with T shared,
    # dJ = 1 - J_PT_b, which is never below 0.
    assert min(out["bootstrap"]) >= 0.0


def test_output_is_json_serialisable():
    json.dumps(run_v(T_CIRCUIT))


# --- the script, end to end, on files in the notebooks' formats --------------


def test_script_reads_the_notebook_output_formats(tmp_path):
    p = condition(P_CIRCUIT, n=16, seed=0)
    t = condition(T_CIRCUIT, ids=list(range(100, 116)), seed=1)
    v = condition(T_CIRCUIT, ids=list(range(8, 24)), seed=2)
    (tmp_path / "p.json").write_text(json.dumps(
        {"model": "m", "checkpoint": "c",
         "result": {"records": p, "promoting": [list(h) for h in P_CIRCUIT], "suppressing": []}}))
    (tmp_path / "t_res.json").write_text(json.dumps(
        {"result": {"promoting": [list(h) for h in T_CIRCUIT], "suppressing": []}}))
    for name, recs in (("t", t), ("v", v)):
        with gzip.open(tmp_path / f"{name}.jsonl.gz", "wt") as f:
            for r in recs + [{"i": 999, "status": "not_conflicting"}]:
                f.write(json.dumps(r) + "\n")
    (tmp_path / "ceil.json").write_text(json.dumps(
        {"result": split_half_ceiling(p, n_splits=20, seed=0, n_perm=200)}))

    # The script's own parameters are D75's (1,000 bootstraps), so this takes
    # a little while even on 16 small synthetic objects.
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analyze_v_circuit.py"),
         str(tmp_path / "p.json"), str(tmp_path / "t.jsonl.gz"), str(tmp_path / "v.jsonl.gz"),
         str(tmp_path / "ceil.json"), str(tmp_path / "t_res.json"), str(tmp_path / "out.json")],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout
    assert "V  " in out and "16 objects" in out
    assert "VERDICT: J_VT > J_PT." in out
    saved = json.loads((tmp_path / "out.json").read_text())
    assert saved["result"]["delta_j"]["verdict"] == "greater"
    assert saved["decision"] == "D75" and saved["experiment"] == "v_circuit_overlap"
