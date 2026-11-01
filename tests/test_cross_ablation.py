"""H2 cross-ablation (D73). Written before any H2 run.

Synthetic known-answer cases show the tests can say both things -- "defeats
the override more than random" and "not distinguishable from random" -- and
the real P knockout records pin the flip definition to Kush's (D67).
"""

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from circuit_overlap.cross_ablation import (
    H2_VERDICTS,
    SETS,
    TESTS,
    ablate_heads_last_token,
    analyse_cross_ablation,
    build_plan,
    head_sets,
    layer_matched_draws,
    null_test,
    plan_cost,
    source_flips,
    target_flips,
)
from circuit_overlap.heads import parse_head_label
from circuit_overlap.replication import knockout_flip_counts

ROOT = Path(__file__).resolve().parents[1]
P_FILE = ROOT / "results" / "replication_patching_qwen25vl3b.json"
T_FILE = ROOT / "results" / "t_patching_qwen25vl3b.json"
KNOCK_FILE = ROOT / "results" / "replication_knockout_qwen25vl3b.json"


def result(path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    return json.loads(path.read_text())["result"]


# --- sets and draws ----------------------------------------------------------


def test_head_sets_decompose():
    s = head_sets([[1, 0], [1, 1], [2, 0]], [[1, 1], [2, 0], [3, 3]])
    assert s["shared"] == [(1, 1), (2, 0)]
    assert s["P_only"] == [(1, 0)] and s["T_only"] == [(3, 3)]


def test_draws_match_layers_and_avoid_excluded_heads():
    heads = [(26, 0), (26, 5), (31, 3)]
    exclude = {(26, 0), (26, 5), (26, 6), (31, 3)}
    draws = layer_matched_draws(heads, exclude, 16, n_draws=200, seed=0, stream=0)
    assert len(draws) == 200
    for d in draws:
        assert Counter(l for l, _ in d) == Counter({26: 2, 31: 1})
        assert not set(d) & exclude
        assert len(set(d)) == len(d)
    # every allowed head in layer 26 turns up eventually: the draw is not stuck
    assert {h for d in draws for l, h in d if l == 26} == set(range(16)) - {0, 5, 6}


def test_draws_are_seeded_and_streams_independent():
    heads = [(26, 0), (31, 3)]
    a = layer_matched_draws(heads, set(), 16, 10, seed=0, stream=0)
    assert a == layer_matched_draws(heads, set(), 16, 10, seed=0, stream=0)
    assert a != layer_matched_draws(heads, set(), 16, 10, seed=0, stream=1)


def test_draws_refuse_an_impossible_layer():
    with pytest.raises(ValueError, match="layer 3"):
        layer_matched_draws([(3, 0), (3, 1)], {(3, h) for h in range(15)}, 16, 1, 0, 0)


# --- the plan on the real discovery results ----------------------------------


def real_plan():
    return build_plan(result(P_FILE), result(T_FILE))


def test_plan_sets_come_from_the_result_files():
    plan, p, t = real_plan(), result(P_FILE), result(T_FILE)
    as_set = lambda labels: {parse_head_label(h) for h in labels}
    assert as_set(plan["sets"]["P"]) == {tuple(h) for h in p["promoting"]}
    assert as_set(plan["sets"]["T"]) == {tuple(h) for h in t["promoting"]}
    assert as_set(plan["sets"]["shared"]) == as_set(plan["sets"]["P"]) & as_set(plan["sets"]["T"])


def test_no_draw_contains_a_classified_head():
    plan = real_plan()
    excluded = {parse_head_label(h) for h in plan["excluded_from_draws"]}
    assert len(excluded) == len(plan["excluded_from_draws"])
    for name, heads in plan["configs"].items():
        if name.startswith("r"):
            assert not {tuple(h) for h in heads} & excluded, name


def test_every_test_has_its_set_and_null_on_its_condition():
    plan = real_plan()
    for test, s, cond, tier in TESTS:
        entries = dict(plan["zero"][cond])
        assert s in entries, test
        nulls = [k for k in entries if k.startswith(SETS[s] + "_")]
        assert len(nulls) == plan["n_draws"], test
    # primary/diagonal nulls also run the target run, for collateral
    for cond in ("P", "T"):
        entries = dict(plan["zero"][cond])
        assert all(entries[k] == "both" for k in entries if k.startswith(("rP_", "rT_")))


def test_mean_plan_is_the_primary_tests_only():
    plan = real_plan()
    assert sorted(k for k, _ in plan["mean"]["T"] if k.startswith("r")) == \
        [f"rP_{k:03d}" for k in range(plan["n_mean_draws"])]
    assert sorted(k for k, _ in plan["mean"]["P"] if k.startswith("r")) == \
        [f"rT_{k:03d}" for k in range(plan["n_mean_draws"])]


def test_plan_is_json_and_its_cost_is_sane():
    plan = real_plan()
    json.dumps(plan)
    cost = plan_cost(plan, {"P": 73, "T": 244}, 277)
    hours = (sum(cost[k] for k in ("zero", "mean", "clean")) * 0.083 + cost["prefills"] * 2.9) / 3600
    assert 5 < hours < 9                      # D73 budgets ~7 h on a T4


# --- the hook, on a fake model ----------------------------------------------


class FakeModule:
    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))


def fake_model(n_layers=4):
    layers = [SimpleNamespace(self_attn=SimpleNamespace(o_proj=FakeModule())) for _ in range(n_layers)]
    return SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers))), layers


def test_ablation_hooks_one_per_head_and_removes_them():
    model, layers = fake_model()
    values = {(1, 0): "v10", (1, 3): "v13", (3, 2): "v32"}
    with ablate_heads_last_token(model, [[1, 0], [3, 2], [1, 3]], head_dim=8, values=values):
        assert [len(l.self_attn.o_proj.hooks) for l in layers] == [0, 2, 0, 1]
    assert all(not l.self_attn.o_proj.hooks for l in layers)


def test_empty_ablation_registers_nothing_and_needs_no_torch():
    model, layers = fake_model()
    with ablate_heads_last_token(model, [], head_dim=8):
        assert all(not l.self_attn.o_proj.hooks for l in layers)


# --- flip definition pinned to Kush's knockout (D67) -------------------------


def test_flips_match_the_replication_knockout_counts():
    knock = result(KNOCK_FILE)
    for config in ("none", "grp_promoting", "grp_suppressing", "ind_L26H5"):
        c = knockout_flip_counts(knock["records"], config)
        assert source_flips(knock["records"], config).sum() == c["prior_flips"]
        assert target_flips(knock["records"], config).sum() == c["visual_flips"]
    assert source_flips(knock["records"], "grp_promoting").sum() == 65   # RESULTS.md, D40


# --- the test statistic -------------------------------------------------------


def test_null_test_p_value_is_hand_checked():
    obs = np.array([1, 1, 0, 0])                         # rate 0.5
    null = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0]])  # 0, .5, .75, .25
    out = null_test(obs, null, n_boot=50, rng=np.random.default_rng(0))
    assert out["observed"] == 0.5 and out["count"] == 2
    assert out["p"] == pytest.approx((1 + 2) / 5)        # draws >= 0.5: two of four
    assert out["null_mean"] == pytest.approx(0.375)
    assert not out["passes"]


def test_mean_draws_cannot_outnumber_zero_draws():
    with pytest.raises(ValueError, match="first n_mean_draws"):
        build_plan({"promoting": [[2, 0]], "suppressing": []},
                   {"promoting": [[2, 1]], "suppressing": []}, 8, n_draws=10, n_mean_draws=20)


def test_null_test_smallest_p_with_100_draws_passes_alpha():
    out = null_test(np.ones(10), np.zeros((100, 10)), n_boot=20, rng=np.random.default_rng(0))
    assert out["p"] == pytest.approx(1 / 101) and out["passes"]
    out = null_test(np.ones(10), np.zeros((50, 10)), n_boot=20, rng=np.random.default_rng(0))
    assert out["p"] == pytest.approx(1 / 51) and out["passes"]    # D73 (10)


# --- end to end, synthetic records with a known answer -----------------------


def toy_plan():
    p = {"promoting": [[2, 0], [2, 1], [3, 0]], "suppressing": [[2, 5]]}
    t = {"promoting": [[2, 0], [3, 1], [3, 2]], "suppressing": []}
    return build_plan(p, t, n_heads_per_layer=8, n_draws=60, n_mean_draws=50, n_clean_draws=5)


def make_records(plan, cond, n, flips, seed=0):
    """flips(config_name, k) -> (source flips?, target flips?) for object k."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        r = {"i": k, "status": "ok"}
        names = [c for c, _ in plan["zero"][cond]] + ["mean:" + c for c, _ in plan["mean"][cond]]
        for name in names:
            sf, tf = flips(name, k)
            base = rng.uniform(1, 3)
            r[f"{name}/prior"] = [10.0, 10.0 + base] if sf else [10.0 + base, 10.0]
            r[f"{name}/visual"] = [10.0 + base, 10.0] if tf else [10.0, 10.0 + base]
        out.append(r)
    return out


def circuit_flips(strong_sets, rate=0.9, noise=0.05):
    rng = np.random.default_rng(1)
    table = {}

    def flips(name, k):
        key = (name, k)
        if key not in table:
            core = name.split(":")[-1]
            p = rate if core in strong_sets else (0.0 if core == "none" else noise)
            table[key] = (bool(rng.random() < p), False)
        return table[key]
    return flips


def test_transfer_is_detected_and_worded():
    plan = toy_plan()
    flips = circuit_flips({"P", "T", "shared"})
    recs = {"P": make_records(plan, "P", 60, flips), "T": make_records(plan, "T", 60, flips, seed=1)}
    out = analyse_cross_ablation(plan, recs, n_boot=100)
    for test in ("P->T", "T->P"):
        assert out["primary"][test]["passes"], out["primary"][test]
        assert out["primary"][test]["p"] == pytest.approx(1 / 61)
    assert out["primary"]["P->T"]["wording"] == \
        "ablating H_P defeats the T override more than layer-matched random heads"
    assert out["mean_ablation"]["same_wording_as_zero"] is True
    # the subsets that do not carry the effect read as null
    assert not out["decomposition"]["P_only->T"]["passes"]
    assert "not distinguishable" in out["decomposition"]["P_only->T"]["wording"]


def test_no_transfer_reads_null():
    plan = toy_plan()
    flips = circuit_flips({"__nothing__"}, noise=0.1)
    recs = {"P": make_records(plan, "P", 60, flips), "T": make_records(plan, "T", 60, flips, seed=1)}
    out = analyse_cross_ablation(plan, recs, n_boot=100)
    for test in ("P->T", "T->P"):
        assert not out["primary"][test]["passes"]
        assert out["primary"][test]["wording"].endswith(H2_VERDICTS["null"])
        assert abs(out["primary"][test]["excess"]) < 0.1


def test_dropped_records_do_not_count():
    plan = toy_plan()
    flips = circuit_flips({"P", "T", "shared"})
    p = make_records(plan, "P", 30, flips)
    p[0]["status"] = "drift"
    out = analyse_cross_ablation(plan, {"P": p, "T": make_records(plan, "T", 30, flips, 1)}, n_boot=20)
    assert out["n_objects"]["P"] == 29 and out["n_dropped"]["P"] == 1


def test_clean_accuracy_and_json():
    plan = toy_plan()
    flips = circuit_flips({"P", "T", "shared"})
    recs = {"P": make_records(plan, "P", 20, flips), "T": make_records(plan, "T", 20, flips, 1)}
    clean = [{"i": k, "status": "ok", **{f"{c}/correct": (c != "P" or k % 2 == 0) for c in plan["clean"]}}
             for k in range(10)]
    out = analyse_cross_ablation(plan, recs, clean, n_boot=20)
    assert out["clean_accuracy"]["accuracy_kept"]["P"] == 0.5
    assert out["clean_accuracy"]["accuracy_kept"]["T"] == 1.0
    json.dumps(out)


def test_wording_is_d73_verbatim():
    decisions = (ROOT / "docs" / "DECISIONS.md").read_text()
    d73 = next(l for l in decisions.splitlines() if l.startswith("| D73 "))
    for phrase in H2_VERDICTS.values():
        assert phrase in d73, phrase


def test_script_reads_the_notebooks_output_format(tmp_path):
    """Summary envelope with the plan, records gzipped with a condition per
    line and clean lines mixed in -- what cross_ablation.ipynb writes."""
    import gzip
    import subprocess
    import sys

    plan = toy_plan()
    flips = circuit_flips({"P", "T", "shared"})
    p = make_records(plan, "P", 40, flips)
    t = make_records(plan, "T", 40, flips, seed=1)
    p.append({"i": 999, "status": "drift"})
    clean = [{"i": k, "status": "ok", **{f"{c}/correct": True for c in plan["clean"]}} for k in range(5)]
    (tmp_path / "s.json").write_text(json.dumps({"commit": "abc", "result": {"plan": plan}}))
    with gzip.open(tmp_path / "r.jsonl.gz", "wt") as f:
        for cond, rs in (("P", p), ("T", t), ("clean", clean)):
            for r in rs:
                f.write(json.dumps({**r, "condition": cond}) + "\n")
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analyze_cross_ablation.py"),
         str(tmp_path / "s.json"), str(tmp_path / "r.jsonl.gz"), str(tmp_path / "a.json")],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout
    assert "objects: P 40 (dropped 1), T 40 (dropped 0)" in out
    assert "VERDICT P->T: ablating H_P defeats the T override more than layer-matched random heads." in out
    saved = json.loads((tmp_path / "a.json").read_text())["result"]
    assert saved["primary"]["T->P"]["passes"]
    assert saved["clean_accuracy"]["n_correct_unablated"] == 5
