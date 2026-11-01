"""CPU tests for the H3 analysis (D74): token positions, statistics, wording.

The GPU helpers (attention capture, direct effects, the reconstruction
checks) need the real model; the notebook's section 2 runs them there.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from circuit_overlap.cross_ablation import build_plan
from circuit_overlap.h3 import (
    FIRST_LAYER,
    H3_TESTS,
    H3_VERDICTS,
    MECHANISM,
    P_RUNS,
    SMALL_INC,
    T_RUNS,
    analyse_h3,
    attention_masses,
    directed_null_test,
    image_token_layout,
    round_grid,
    set_values,
    token_sets,
)
from circuit_overlap.notes import NoteSpec, object_bbox_from_pair, object_mask_from_pair
from circuit_overlap.t_patching import render_t_pair
from circuit_overlap.t_screen import ScreenConfig

ROOT = Path(__file__).resolve().parents[1]
N_LAYERS, N_HEADS = 36 - FIRST_LAYER, 16


def real_plan(**kw):
    p = json.loads((ROOT / "results/replication_patching_qwen25vl3b.json").read_text())["result"]
    t = json.loads((ROOT / "results/t_patching_qwen25vl3b.json").read_text())["result"]
    return build_plan(p, t, n_heads_per_layer=N_HEADS, **kw)


# --- token positions -----------------------------------------------------------


def _prompt(rows, cols, before=5, after=7, image_id=99):
    ids = np.array([1] * before + [image_id] * (rows * cols) + [2] * after)
    n = ids.size
    t = np.arange(n)
    h, w = t.copy(), t.copy()
    k = np.arange(rows * cols)
    h[before:before + rows * cols] = before + k // cols
    w[before:before + rows * cols] = before + k % cols
    return ids, np.stack([t, t, h, w])[:, : n - 1]       # 4 rows, prefix only


def test_image_token_layout_accepts_row_major_and_returns_offset():
    ids, pos = _prompt(3, 4)
    assert image_token_layout(ids, pos, 99, 3, 4) == 5
    assert image_token_layout(ids, pos[1:], 99, 3, 4) == 5     # 3-row M-RoPE form


def test_image_token_layout_rejects_wrong_count_gap_and_column_major():
    ids, pos = _prompt(3, 4)
    with pytest.raises(ValueError, match="grid"):
        image_token_layout(ids, pos, 99, 4, 4)
    gap = ids.copy()
    gap[6] = 1
    gap = np.append(gap, 99)
    with pytest.raises(ValueError):
        image_token_layout(gap, np.concatenate([pos, pos[:, -1:]], axis=1), 99, 3, 4)
    col_major = pos.copy()
    k = np.arange(12)
    col_major[2, 5:17], col_major[3, 5:17] = 5 + k % 3, 5 + k // 3
    with pytest.raises(ValueError, match="row-major"):
        image_token_layout(ids, col_major, 99, 3, 4)


def test_token_sets_separate_note_object_and_edge_tokens():
    rows, cols, cell = 4, 4, 10
    note = np.zeros((rows * cell, cols * cell), bool)
    note[10:20, 10:20] = True            # exactly token (1, 1)
    note[10:16, 20:30] = True            # 60% of token (1, 2): note >= 0.5 (exactly half
    #                                      reads 127/255 after coverage_on_grid's 8-bit resize)
    note[0:2, 0:10] = True               # a sliver of token (0, 0): neither note nor object
    obj = np.zeros_like(note)
    obj[0:30, 0:30] = True               # tokens (0..2, 0..2)
    s = token_sets(note, obj, rows, cols, offset=100)
    assert list(s["image"]) == list(range(100, 116))
    assert sorted(s["note"]) == [105, 106]
    # object: the 3x3 block minus anything the note touches at all
    assert sorted(s["object"]) == [101, 102, 104, 108, 109, 110]
    none = token_sets(None, obj, rows, cols, offset=0)
    assert none["note"].size == 0 and none["object"].size == 9


def test_attention_masses_sum_weights_per_head():
    w = np.array([[0.5, 0.25, 0.25, 0.0], [0.1, 0.2, 0.3, 0.4]])
    m = attention_masses(w, {"a": np.array([1, 2]), "b": np.array([3])})
    assert np.allclose(m["a"], [0.5, 0.5]) and np.allclose(m["b"], [0.0, 0.4])


def test_round_grid_keeps_four_significant_figures():
    assert round_grid([[0.000123456, 12.34567]]) == [[0.0001235, 12.35]]


# --- shared-code additions -------------------------------------------------------


def _pair():
    orig = Image.new("RGB", (112, 112), (200, 200, 200))
    cf = orig.copy()
    cf.paste((0, 0, 255), (30, 40, 70, 90))
    orig.paste((255, 0, 0), (30, 40, 70, 90))
    return orig, cf


def test_object_mask_is_what_the_bbox_boxes():
    orig, cf = _pair()
    m = object_mask_from_pair(orig, cf)
    assert m.shape == (112, 112) and m.sum() == 40 * 50
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    assert object_bbox_from_pair(orig, cf) == (cols[0] / 112, rows[0] / 112,
                                               (cols[-1] + 1) / 112, (rows[-1] + 1) / 112)
    assert object_mask_from_pair(orig, orig.resize((56, 56))) is None


def test_render_t_pair_returns_the_note_mask_of_both_renders():
    orig, cf = _pair()
    out = render_t_pair(orig, cf, "blue", ["table"], NoteSpec(rel_width=0.2, mode="on"))
    assert out["note_mask"].shape == (112, 112) and out["note_mask"].any()
    assert out["same_note_footprint"]
    assert abs(out["note_mask"].mean() - out["note_area_frac"]) < 1e-12


def test_screen_config_keeps_the_default_attention_unless_asked():
    assert ScreenConfig().attn_implementation is None
    assert ScreenConfig(attn_implementation="eager").attn_implementation == "eager"


# --- statistics ----------------------------------------------------------------


def test_directed_null_test_p_values_hand_checked():
    rng = np.random.default_rng(0)
    obs = np.full(10, 0.5)
    null = np.array([[0.1] * 10, [0.6] * 10, [0.5] * 10, [0.2] * 10])   # 2 of 4 >= 0.5
    g = directed_null_test(obs, null, "greater", 50, rng)
    assert g["p"] == pytest.approx(3 / 5) and not g["passes"]
    l = directed_null_test(obs, null, "less", 50, rng)                  # 3 of 4 <= 0.5
    assert l["p"] == pytest.approx(4 / 5)
    best = directed_null_test(np.full(10, -1.0), null, "less", 50, rng)
    assert best["p"] == pytest.approx(1 / 5) and best["min_attainable_p"] == pytest.approx(1 / 5)


def _grid(rng, scale=0.01):
    return (rng.random((N_LAYERS, N_HEADS)) * scale).tolist()


def _records(plan, n_t=40, n_p=20, seed=1, bump=None):
    """Toy records; ``bump`` = {(run, measure): (heads, amount)} adds to those heads."""
    rng = np.random.default_rng(seed)
    bump = bump or {}

    def run(cond, name):
        g = {m: np.array(_grid(rng)) for m in ("m_image", "m_object", "m_note", "dla")}
        g["dla"] -= 0.005
        for (r, m), (heads, amount) in bump.items():
            if r == name:
                for l, h in heads:
                    g[m][l - FIRST_LAYER, h] += amount
        out = {k: v.tolist() for k, v in g.items()}
        if cond == "P":
            out["m_note"] = None
        out.update(top1=1, pair=[0.0, 0.0], finite=True,
                   n_tokens={"image": 1280, "object": 100, **({"note": 20} if cond == "T" else {})})
        return out

    t = [{"i": k, "condition": "T", "status": "ok", "runs": {n: run("T", n) for n in T_RUNS},
          "fooled": {"on_object_small": True, "on_object_medium": k % 2 == 0, "beside_object": k % 3 == 0}}
         for k in range(n_t)]
    p = [{"i": k, "condition": "P", "status": "ok", "runs": {n: run("P", n) for n in P_RUNS}}
         for k in range(n_p)]
    return {"T": t, "P": p}


def test_set_values_mean_for_masses_sum_for_direct_effects():
    plan = real_plan(n_draws=5, n_mean_draws=5)
    recs = _records(plan, n_t=3, n_p=1)["T"]
    heads = plan["configs"]["T_only"]
    grid = np.array(recs[0]["runs"][SMALL_INC]["m_note"])
    expect = np.mean([grid[l - FIRST_LAYER, h] for l, h in heads])
    assert set_values(recs, SMALL_INC, "m_note", heads)[0] == pytest.approx(expect)
    dla = np.array(recs[0]["runs"][SMALL_INC]["dla"])
    assert set_values(recs, SMALL_INC, "dla", heads)[0] == pytest.approx(
        sum(dla[l - FIRST_LAYER, h] for l, h in heads))


def test_known_answer_readers_pass_both_primary_tests():
    plan = real_plan()
    t_only = [tuple(h) for h in plan["configs"]["T_only"]]
    recs = _records(plan, bump={(SMALL_INC, "m_note"): (t_only, 0.2), (SMALL_INC, "dla"): (t_only, 0.5)})
    out = analyse_h3(plan, recs, n_boot=100)
    assert out["primary"]["A"]["passes"] and out["primary"]["W"]["passes"]
    assert out["primary"]["A"]["p"] == pytest.approx(1 / 101)
    assert out["mechanism"] == MECHANISM[(True, True)]
    assert out["primary"]["A"]["wording"] == "T-only heads attend to the note more than layer-matched random heads"
    assert out["primary"]["W"]["wording"] == "T-only heads write the note's colour more than layer-matched random heads"
    # the bump is on the incongruent run only, so attention also "rises" vs neutral
    assert out["secondary"]["A_rise:T_only"]["passes"]


def test_known_answer_null_and_the_less_direction():
    plan = real_plan()
    t_only = [tuple(h) for h in plan["configs"]["T_only"]]
    recs = _records(plan, bump={("clean", "dla"): (t_only, -0.5), ("visual", "m_object"): (t_only, 0.2)})
    out = analyse_h3(plan, recs, n_boot=100)
    assert not out["primary"]["A"]["passes"] and not out["primary"]["W"]["passes"]
    assert out["mechanism"] == MECHANISM[(False, False)]
    assert out["primary"]["A"]["wording"] == "T-only heads: not distinguishable from layer-matched random heads at this n"
    assert out["secondary"]["image_colour:clean"]["passes"]          # predicted negative, is negative
    assert out["secondary"]["image_colour:clean"]["direction"] == "less"
    assert out["secondary"]["object:P_visual"]["passes"]
    assert not out["validation"]["routing:P"]["passes"]


def test_routing_validation_reads_visual_minus_prior():
    plan = real_plan()
    hp = [tuple(h) for h in plan["configs"]["P"]]
    out = analyse_h3(plan, _records(plan, bump={("visual", "m_image"): (hp, 0.3)}), n_boot=100)
    assert out["validation"]["routing:P"]["passes"]
    out = analyse_h3(plan, _records(plan, bump={("prior", "m_image"): (hp, 0.3)}), n_boot=100)
    assert not out["validation"]["routing:P"]["passes"]


def test_dropped_records_are_reported_not_used():
    plan = real_plan(n_draws=20, n_mean_draws=20)
    recs = _records(plan, n_t=10, n_p=5)
    recs["T"][0]["status"] = "drift"
    recs["P"][1]["status"] = "numerics_failed"
    out = analyse_h3(plan, recs, n_boot=20)
    assert out["n_objects"] == {"T": 9, "P": 4} and out["n_dropped"] == {"T": 1, "P": 1}
    assert out["dropped_status"] == {"T": ["drift"], "P": ["numerics_failed"]}


def test_descriptives_cover_spatial_enrichment_and_layers():
    plan = real_plan(n_draws=10, n_mean_draws=10)
    d = analyse_h3(plan, _records(plan, n_t=12, n_p=4), n_boot=20)["descriptive"]
    assert set(d["spatial"]) == {"on_object_small", "on_object_medium", "beside_object"}
    assert d["spatial"]["on_object_small"]["fool_rate"] == 1.0
    assert d["spatial"]["on_object_medium"]["fooled_minus_not_incongruent"]["n"] == [6, 6]
    assert d["note_token_share"] == pytest.approx(20 / 1280)
    assert set(d["layer_profile_note_mass"]) == {str(l) for l in range(FIRST_LAYER, 36)}
    assert "L33H1" in d["per_head_T"] and "L26H6" in d["per_head_P"]


def test_tests_cover_d74_and_every_wording_is_in_decisions():
    decisions = (ROOT / "docs/DECISIONS.md").read_text()
    row = next(l for l in decisions.splitlines() if l.startswith("| D74 |"))
    assert [t[0] for t in H3_TESTS if t[-1] == "primary"] == ["A", "W"]
    for phrase in ("T-only heads attend to the note more than layer-matched random heads",
                   "T-only heads write the note's colour more than layer-matched random heads",
                   H3_VERDICTS["null"], *MECHANISM.values()):
        assert phrase in row, phrase
    assert "alpha = 0.025" in row and "layers 20-35" in row


def test_script_end_to_end(tmp_path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("analyze_h3", ROOT / "scripts/analyze_h3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plan = real_plan(n_draws=20, n_mean_draws=20)
    recs = _records(plan, n_t=8, n_p=4)
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"result": {"plan": plan}, "commit": "abc"}))
    rec_path = tmp_path / "r.jsonl.gz"
    with gzip.open(rec_path, "wt") as f:
        for c in ("T", "P"):
            for r in recs[c]:
                f.write(json.dumps(r) + "\n")
    out = mod.main(str(summary), str(rec_path), str(tmp_path / "a.json"))
    text = capsys.readouterr().out
    assert "VERDICT A:" in text and "MECHANISM" in text and (tmp_path / "a.json").exists()
    assert out["n_objects"] == {"T": 8, "P": 4}
