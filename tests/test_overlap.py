"""P-T overlap analysis (D70, D72). Written before any T-circuit result.

Besides synthetic cases, two controls on the committed P records show the
instrument can return BOTH answers: P against itself must read "self-
agreement", and P against a head-scrambled copy of itself must read "chance".
An instrument that cannot say both is not measuring anything.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from circuit_overlap.ceiling import split_half_ceiling
from circuit_overlap.overlap import (
    VERDICTS,
    analyse_overlap,
    d70_verdict,
    layer_profile,
    stable_core,
)

ROOT = Path(__file__).resolve().parents[1]
P_FILE = ROOT / "results" / "replication_patching_qwen25vl3b.json"


# --- the wording rule ------------------------------------------------------


def test_verdict_four_branches():
    floor, ceil = 0.06, 0.90
    assert d70_verdict(0.00, 0.03, floor, ceil) == "chance"
    assert d70_verdict(0.92, 1.00, floor, ceil) == "ceiling"
    assert d70_verdict(0.30, 0.60, floor, ceil) == "between"
    assert d70_verdict(0.02, 0.95, floor, ceil) == "inconclusive"


def test_verdict_boundaries_count_as_reaching():
    floor, ceil = 0.06, 0.90
    assert d70_verdict(0.06, 0.40, floor, ceil) == "chance"     # lower end == floor p95
    assert d70_verdict(0.50, 0.90, floor, ceil) == "ceiling"    # upper end == ceiling p2.5


def test_verdict_rejects_malformed_inputs():
    with pytest.raises(ValueError, match="not ordered"):
        d70_verdict(0.5, 0.4, 0.06, 0.9)
    with pytest.raises(ValueError, match="no length"):
        d70_verdict(0.1, 0.2, 0.9, 0.9)


def test_verdict_phrases_are_d70_verbatim():
    decisions = (ROOT / "docs" / "DECISIONS.md").read_text()
    d70 = next(l for l in decisions.splitlines() if l.startswith("| D70 "))
    for phrase in VERDICTS.values():
        assert phrase in d70, phrase


# --- synthetic records -----------------------------------------------------


def make_record(i, p2v, v2p):
    p2v, v2p = np.asarray(p2v, float), np.asarray(v2p, float)
    return {
        "i": i, "status": "patched",
        "prior_token_logit_prior_run": 1.0, "visual_token_logit_prior_run": 0.0,
        "prior_token_logit_visual_run": 0.0, "visual_token_logit_visual_run": 0.0,
        "p2v_logit_prior": p2v.tolist(), "p2v_logit_visual": np.zeros_like(p2v).tolist(),
        "v2p_logit_prior": (1.0 - v2p).tolist(), "v2p_logit_visual": np.zeros_like(v2p).tolist(),
    }


def condition(circuit, n=30, ids=None, seed=0, noise=0.02, shape=(12, 8)):
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


A = [(8, 1), (9, 6), (10, 2), (11, 7)]
B = [(2, 0), (3, 3), (4, 5), (5, 1)]


def run(p, t, **kw):
    ceil = split_half_ceiling(p, n_splits=40, seed=0, n_perm=500)
    return analyse_overlap(p, t, ceil, n_boot=60, t_splits=40, n_perm=500, **kw)


def test_same_circuit_reads_as_self_agreement():
    out = run(condition(A, seed=0), condition(A, seed=1))
    assert out["primary"]["j_pt"] == pytest.approx(1.0)
    assert out["primary"]["verdict"] == "ceiling"
    assert out["stable_core"]["j_core"] == pytest.approx(1.0)


def test_disjoint_circuits_read_as_chance():
    out = run(condition(A, seed=0), condition(B, seed=1))
    assert out["primary"]["j_pt"] == 0.0
    assert out["primary"]["verdict"] == "chance"
    assert out["shared_promoting"] == []


def test_partial_overlap_reads_between():
    half = A[:2] + B[:2]                          # shares 2 of 4 heads with A
    out = run(condition(A, seed=0), condition(half, seed=1))
    assert out["primary"]["j_pt"] == pytest.approx(2 / 6)
    assert out["primary"]["verdict"] == "between"
    assert out["descriptive"]["containment_p_in_t"] == pytest.approx(0.5)


def test_chance_floor_uses_the_observed_set_sizes():
    """D72: plan 3.5 defines the floor at the sizes of the two sets, so it
    moves when T's set is bigger than P's."""
    big = A + B                                   # 8 heads
    small = run(condition(A, seed=0), condition(A, seed=1))["primary"]
    large = run(condition(A, seed=0), condition(big, seed=1))["primary"]
    assert small["set_sizes"] == [4, 4] and large["set_sizes"] == [4, 8]
    assert large["chance_floor_mean"] > small["chance_floor_mean"]


def test_shared_object_check_uses_the_intersection():
    p = condition(A, ids=list(range(0, 30)), seed=0)
    t = condition(A, ids=list(range(20, 50)), seed=1)
    out = run(p, t)
    assert out["shared_objects"]["n_objects"] == 10
    assert out["shared_objects"]["j_pt"] == pytest.approx(1.0)
    assert out["shared_objects"]["selection_effect_flag"] is False


def test_layer_profile_and_stable_core():
    prof = layer_profile({(26, 0), (31, 3)}, {(26, 0), (2, 1)}, n_layers=36, band=6)
    assert [b["layers"] for b in prof][:2] == ["0-5", "6-11"]
    band = next(b for b in prof if b["layers"] == "24-29")
    assert (band["p"], band["t"], band["shared"]) == (1, 1, 1)
    assert stable_core({"L26H5": 1.0, "L31H1": 0.484, "L2H2": 0.9}) == {(26, 5), (2, 2)}


def test_output_is_json_serialisable():
    json.dumps(run(condition(A, n=12), condition(A, n=12, seed=1)))


# --- controls on the committed P data --------------------------------------


def p_records():
    if not P_FILE.exists():
        pytest.skip("P patching results not present")
    payload = json.loads(P_FILE.read_text())
    return payload["result"]["records"]


def p_ceiling():
    path = ROOT / "results" / "split_half_ceiling_p_qwen25vl3b.json"
    if not path.exists():
        pytest.skip("P ceiling not present")
    return json.loads(path.read_text())["result"]


def scramble_heads(records, seed=0):
    """Same records, every head moved to a random new position (one fixed
    permutation for all objects): the signal strength is unchanged, only its
    location is wrong."""
    perm = np.random.default_rng(seed).permutation(36 * 16)
    out = []
    for r in records:
        r = copy.deepcopy(r)
        for key in ("p2v_logit_prior", "p2v_logit_visual", "v2p_logit_prior", "v2p_logit_visual"):
            r[key] = np.asarray(r[key], float).ravel()[perm].reshape(36, 16).tolist()
        out.append(r)
    return out


def test_positive_control_p_against_itself_reads_self_agreement():
    p = p_records()
    out = analyse_overlap(p, p, p_ceiling(), n_boot=40, t_splits=20, n_perm=500)
    assert out["primary"]["j_pt"] == pytest.approx(1.0)
    assert out["primary"]["verdict"] == "ceiling", out["primary"]


def test_negative_control_scrambled_heads_read_chance():
    p = p_records()
    out = analyse_overlap(p, scramble_heads(p), p_ceiling(), n_boot=40, t_splits=20, n_perm=500)
    assert out["primary"]["j_pt"] < 0.2
    assert out["primary"]["verdict"] == "chance", out["primary"]


# --- known-answer controls: partial overlap built from the real P data -----

P_PROM = [(26, 0), (26, 5), (26, 6), (27, 1), (27, 4), (28, 3), (31, 3), (31, 7), (34, 14)]


def move_heads(records, k, seed=0):
    """Swap k of P's 9 circuit heads with k random inactive heads, the same
    swap for every object, so the true overlap is 9 - k shared heads."""
    rng = np.random.default_rng(seed)
    src = [P_PROM[j] for j in rng.choice(9, k, replace=False)]
    others = [(l, h) for l in range(36) for h in range(16) if (l, h) not in P_PROM]
    dst = [others[j] for j in rng.choice(len(others), k, replace=False)]
    out = []
    for r in records:
        r = copy.deepcopy(r)
        for key in ("p2v_logit_prior", "p2v_logit_visual", "v2p_logit_prior", "v2p_logit_visual"):
            g = np.asarray(r[key], float)
            for (a, b), (c, d) in zip(src, dst):
                g[a, b], g[c, d] = g[c, d].copy(), g[a, b].copy()
            r[key] = g.tolist()
        out.append(r)
    return out


def test_partial_overlap_control_reads_between_with_an_unbiased_bootstrap():
    """3 of 9 heads moved: 6 shared of 12 in the union, J = 0.5 exactly.
    D70 must read 'between', and the bootstrap must be centred on it."""
    p = p_records()
    out = analyse_overlap(p, move_heads(p, 3), p_ceiling(), n_boot=60, t_splits=20, n_perm=500)
    pr = out["primary"]
    assert pr["j_pt"] == pytest.approx(0.5)
    assert pr["verdict"] == "between", pr
    assert pr["point_inside_ci"]
    assert abs(pr["bootstrap_bias"]) < 0.05


def test_rank_correlation_is_insensitive_to_circuit_identity():
    """Why the rank correlation is not an overlap measure (D72): with 7 of the
    9 circuit heads moved, J falls to 0.125 but rho stays near 1, because the
    other ~567 heads dominate it."""
    p = p_records()
    out = analyse_overlap(p, move_heads(p, 7), p_ceiling(), n_boot=20, t_splits=10, n_perm=500)
    assert out["primary"]["j_pt"] == pytest.approx(2 / 16)
    assert out["rank_correlation"]["rho"] > 0.9
    assert "not an overlap measure" in out["rank_correlation"]["reading"]


# --- the script, end to end, on files in the notebook's exact formats ------


def test_script_reads_the_notebooks_output_formats(tmp_path):
    """P as a result envelope (.json), T as gzipped JSON lines with a status
    per object (what t_patching.ipynb writes), the ceiling as an envelope."""
    import gzip
    import subprocess
    import sys

    from circuit_overlap.ceiling import split_half_ceiling

    p = condition(A, n=16, seed=0)
    t = condition(A, n=16, ids=list(range(8, 24)), seed=1)
    t_lines = t + [{"i": 99, "status": "not_conflicting"},
                   {"i": 98, "status": "numerics_failed"}]
    (tmp_path / "p.json").write_text(json.dumps(
        {"model": "m", "checkpoint": "c", "result": {"records": p}}))
    with gzip.open(tmp_path / "t.jsonl.gz", "wt") as f:
        for r in t_lines:
            f.write(json.dumps(r) + "\n")
    (tmp_path / "ceil.json").write_text(json.dumps(
        {"result": split_half_ceiling(p, n_splits=20, seed=0, n_perm=200)}))

    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analyze_t_circuit.py"),
         str(tmp_path / "p.json"), str(tmp_path / "t.jsonl.gz"),
         str(tmp_path / "ceil.json"), str(tmp_path / "overlap.json")],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout
    assert "16 objects" in out                      # non-patched T lines filtered out
    assert "VERDICT: J_PT is not distinguishable from the P circuit's own self-agreement" in out
    saved = json.loads((tmp_path / "overlap.json").read_text())
    assert saved["result"]["primary"]["verdict"] == "ceiling"
    assert saved["result"]["shared_objects"]["n_objects"] == 8
