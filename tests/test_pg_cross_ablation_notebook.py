"""Static checks on notebooks/pg_cross_ablation.ipynb (D78), which cannot run on the CPU."""

import ast
import builtins
import importlib
import json
import subprocess
import sys
from pathlib import Path

from test_t_patching_notebook import bound_names

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "pg_cross_ablation.ipynb"


def cells():
    return json.loads(NB.read_text())["cells"]


def code_cells():
    for cell in cells():
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            yield cell, "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("!"))


def all_code():
    return "\n".join(s for _, s in code_cells())


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


def test_setup_cells_are_the_ones_that_ran():
    pgt = ["".join(c["source"]) for c in json.loads((ROOT / "notebooks" / "pg_t_patching.ipynb").read_text())["cells"]]
    ours = ["".join(c["source"]) for c in cells()]
    assert ours[3] == pgt[3] and ours[5] == pgt[5]


def test_d78_plan_and_objects():
    src = all_code()
    assert "build_plan(p_res, t_res, n_heads_per_layer=n_heads)" in src
    assert 'assert (len(todo["P"]), len(todo["T"]), len(clean_ids)) == (121, 302, 354)' in src
    assert "L15H" not in src and "L20H" not in src and "(15," not in src      # heads are data
    assert "LastTokenRunner" not in src


def test_replay_is_checked_against_independent_full_passes_before_any_loop():
    src = all_code()
    first_loop = src.index('run_cached(CKPT / f"zero_{cond}.jsonl"')
    for check in ('all_zero_configs_', 'values_path', 'unablated_row_in_ablated_batch',
                  'batch_vs_alone', 'batch_vs_reversed', 'ablated_rows_differ', '"clean"'):
        assert src.index(check) < first_loop, check
    # the reference is D73's own hook in a full forward pass, not the replay's hook
    assert "with ablate_heads_last_token(model, heads, head_dim, values):" in src
    assert "n_flip_mismatch == 0 and n_argmax_mismatch == 0 and worst < 0.5" in src
    # every config of the plan is compared, on both runs where the plan asks for both
    assert 'names = [n for n, runs in plan["zero"][cond] if kind == "prior" or runs == "both"]' in src


def test_every_object_checks_its_own_replay():
    src = all_code()
    assert 'out["replay_check"] = round(replay_check(reps), 4)' in src
    assert '"replay_mismatch"' in src and "REPLAY_TOL = 0.1" in src
    assert src.count("softcap=SOFTCAP") >= 6      # one head decision, reused everywhere after VERIFY


def test_every_long_loop_checkpoints_and_respects_the_budget():
    src = all_code()
    for name in ('f"zero_{cond}.jsonl"', 'f"mean_{cond}.jsonl"', '"clean.jsonl"'):
        assert f"run_cached(CKPT / {name}" in src, name
    assert src.count("budget_s=max(left(), 0)") == 3


def test_output_prefix_and_owner_are_in_claude_md():
    claude = (ROOT / "CLAUDE.md").read_text()
    assert "`pg_cross_ablation.ipynb` uses" in claude and "notebooks/pg_cross_ablation.ipynb" in claude


def test_the_freshness_check_names_exist_in_src():
    cell1 = next(s for _, s in code_cells())
    table = next(n for n in ast.walk(ast.parse(cell1)) if isinstance(n, ast.Dict)
                 and any(isinstance(k, ast.Constant) and k.value == "circuit_overlap.replay"
                         for k in n.keys))
    for key, value in zip(table.keys, table.values):
        mod = importlib.import_module(key.value)
        for elt in value.elts:
            assert hasattr(mod, elt.value), f"{key.value}.{elt.value}"


def test_saved_files_are_what_the_wrapper_reads():
    src = all_code()
    doc = (ROOT / "scripts" / "analyze_pg_cross_ablation.py").read_text()
    for name in ("pg_cross_ablation_paligemma2_3b.json", "pg_cross_ablation_records_paligemma2_3b.jsonl.gz"):
        assert name in src and name in doc
    assert '"plan": plan' in src and '"complete": complete' in src


def test_wrapper_refuses_a_plan_that_is_not_d78s():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_pg_cross_ablation.py"),
                        str(ROOT / "results" / "cross_ablation_qwen25vl3b.json"),
                        str(ROOT / "results" / "cross_ablation_records_qwen25vl3b.jsonl.gz")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "REFUSED: the run's plan differs" in r.stdout and "VERDICT" not in r.stdout
