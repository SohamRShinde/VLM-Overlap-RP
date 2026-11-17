"""CPU checks for PaliGemma H3 (D80): the notebook's structure and the analysis wrapper.

The GPU helpers in ``pg_h3`` need the real model; the notebook's section 3
runs their checks there and stops if any fails.
"""

import ast
import builtins
import gzip
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from circuit_overlap.cross_ablation import build_plan
from circuit_overlap.h3 import P_RUNS, T_RUNS
from circuit_overlap.pg_h3 import FIRST_LAYER_PG, PROBE_CELL
from test_t_patching_notebook import bound_names

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "pg_h3_attention.ipynb"
N_LAYERS, N_HEADS = 26, 8


def cells():
    return json.loads(NB.read_text())["cells"]


def code_cells():
    for cell in cells():
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            yield cell, "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("!"))


def all_code():
    return "\n".join(s for _, s in code_cells())


def pg_plan(**kw):
    p = json.loads((ROOT / "results/pg_replication_patching_paligemma2_3b.json").read_text())["result"]
    t = json.loads((ROOT / "results/pg_t_patching_paligemma2_3b.json").read_text())["result"]
    return build_plan(p, t, n_heads_per_layer=N_HEADS, **kw)


# --- the notebook --------------------------------------------------------------


def test_every_code_cell_parses():
    for i, (_, src) in enumerate(code_cells()):
        ast.parse(src, filename=f"cell {i}")


def test_no_name_is_used_before_a_cell_defines_it():
    defined = set(dir(builtins)) | {"display", "get_ipython"}
    problems = []
    for i, (_, src) in enumerate(code_cells()):
        tree = ast.parse(src)
        local = bound_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in defined and node.id not in local:
                    problems.append(f"code cell {i}, line {node.lineno}: {node.id!r}")
        defined |= local
    assert not problems, "used before defined:\n  " + "\n  ".join(problems)


def test_outputs_are_cleared():
    for cell, _ in code_cells():
        assert cell["outputs"] == [] and cell["execution_count"] is None


def test_install_cell_is_the_one_that_ran():
    ran = "".join(json.loads((ROOT / "notebooks/pg_cross_ablation.ipynb").read_text())["cells"][3]["source"])
    assert "".join(cells()[3]["source"]) == ran


def test_d80_pieces():
    src = all_code()
    assert 'pathlib.Path("/kaggle/working/pg_h3_attention")' in src
    assert 'load_paligemma(attn_implementation="eager")' in src and 'ATTN == "eager"' in src   # (2)
    assert "FullForwardRunner(model, inputs)" in src and "LastTokenRunner" not in src         # (2)
    assert "LAYERS = list(range(FIRST_LAYER_PG, n_layers))" in src                            # (3)
    assert "square_image_block(" in src and "check_patch_order(" in src                        # (4)
    assert '["L20H7", "L22H4", "L22H5", "L23H7"]' in src                                       # (1)
    assert "(302, 121)" in src
    for check in ("check_attention_reconstructs_heads_full(", "check_direct_effect_formula_gemma(",
                  "max_weight_sum_error", 'm0["top1"] == tok0[0]'):                            # (6)
        assert check in src, check
    assert "analyse_h3(plan, h3, n_boot=200, first_layer=FIRST_LAYER_PG)" in src
    assert "L15H" not in src.replace("layer=15", "")        # no head typed in by hand


def test_checks_come_before_the_loops():
    src = all_code()
    loop = src.index('run_cached(CKPT / "h3_T.jsonl"')
    for check in ("check_patch_order(", "check_attention_reconstructs_heads_full(",
                  "check_direct_effect_formula_gemma("):
        assert src.index(check) < loop, check


def test_the_runs_are_d74s():
    src = all_code()
    assert 'LAYOUTS = ("on_object_small", "on_object_medium", "beside_object")' in src
    assert 'out["runs"][f"{layout}/{variant}"]' in src and 'out["runs"]["clean"]' in src
    assert 'prompts = {"prior": PRIOR_PROMPT' in src
    names = {f"{l}/{v}" for l in ("on_object_small", "on_object_medium", "beside_object")
             for v in ("incongruent", "neutral")} | {"clean"}
    assert names == set(T_RUNS) and set(P_RUNS) == {"prior", "visual"}


def test_the_freshness_check_names_exist_in_src():
    cell1 = next(s for _, s in code_cells())
    table = next(n for n in ast.walk(ast.parse(cell1)) if isinstance(n, ast.Dict)
                 and any(isinstance(k, ast.Constant) and k.value == "circuit_overlap.pg_h3" for k in n.keys))
    for key, value in zip(table.keys, table.values):
        mod = importlib.import_module(key.value)
        for elt in value.elts:
            assert hasattr(mod, elt.value), f"{key.value}.{elt.value}"


def test_output_prefix_and_owner_are_in_claude_md():
    claude = (ROOT / "CLAUDE.md").read_text()
    assert "`pg_h3_attention.ipynb` uses" in claude and "notebooks/pg_h3_attention.ipynb" in claude


def test_probe_cell_catches_a_transposed_grid():
    r, c = PROBE_CELL
    assert r != c and FIRST_LAYER_PG == 0


def test_plan_is_d80s():
    sets = pg_plan()["sets"]
    assert sets["T_only"] == ["L20H7", "L22H4", "L22H5", "L23H7"]
    assert len(sets["shared"]) == 4 and len(sets["P_only"]) == 4
    decisions = (ROOT / "docs/DECISIONS.md").read_text()
    row = next(l for l in decisions.splitlines() if l.startswith("| D80 |"))
    assert "L20H7, L22H4, L22H5, L23H7" in row


# --- the wrapper ---------------------------------------------------------------


def _toy(plan, n_t=8, n_p=4, drift_t=0, seed=1):
    rng = np.random.default_rng(seed)

    def run(cond):
        g = {m: (rng.random((N_LAYERS, N_HEADS)) * 0.01).tolist() for m in ("m_image", "m_object", "m_note", "dla")}
        if cond == "P":
            g["m_note"] = None
        return {**g, "top1": 1, "pair": [0.0, 0.0], "finite": True,
                "n_tokens": {"image": 1024, "object": 80, **({"note": 20} if cond == "T" else {})}}

    t = [{"i": k, "condition": "T", "status": "drift" if k < drift_t else "ok",
          "runs": {n: run("T") for n in T_RUNS},
          "fooled": {"on_object_small": True, "on_object_medium": k % 2 == 0, "beside_object": False}}
         for k in range(n_t)]
    p = [{"i": k, "condition": "P", "status": "ok", "runs": {n: run("P") for n in P_RUNS}} for k in range(n_p)]
    return t + p


def _write(tmp_path, plan, recs):
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"result": {"plan": plan, "layers_stored": [0, 25]}, "commit": "abc"}))
    rec = tmp_path / "r.jsonl.gz"
    with gzip.open(rec, "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return summary, rec


def test_wrapper_end_to_end_reads_all_26_layers(tmp_path):
    plan = pg_plan(n_draws=20, n_mean_draws=20)
    summary, rec = _write(tmp_path, plan, _toy(plan, n_t=10, drift_t=1))
    out = tmp_path / "a.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_pg_h3.py"), str(summary), str(rec), str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "T: 1 of 10 objects drifted" in r.stdout and "NO H3 CLAIM" not in r.stdout
    assert "DESCRIPTIVE on PaliGemma" in r.stdout and "VERDICT A:" in r.stdout
    saved = json.loads(out.read_text())["result"]
    assert saved["parameters"]["first_layer"] == 0
    assert saved["d80_drift"]["T"]["drift"] == 1 and saved["n_objects"] == {"T": 9, "P": 4}


def test_wrapper_flags_more_than_20_percent_drift(tmp_path):
    plan = pg_plan(n_draws=20, n_mean_draws=20)
    summary, rec = _write(tmp_path, plan, _toy(plan, n_t=10, drift_t=3))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_pg_h3.py"), str(summary), str(rec)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "T: 3 of 10 objects drifted under eager attention (30.0%)  <- over 20%: NO H3 CLAIM" in r.stdout
