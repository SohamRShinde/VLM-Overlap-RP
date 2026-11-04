"""Static checks on notebooks/v_patching.ipynb, which cannot run on the CPU.

Same checks as tests/test_t_patching_notebook.py: the costly Kaggle failure is
a name used in one cell that no earlier cell defines, found hours into a run.
This walks the cells in order and fails on any such name, and pins D75's
roles, cap and checkpoints.
"""

import ast
import builtins
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "v_patching.ipynb"


def code_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            # IPython shell escapes (!pip ...) are not Python.
            yield cell, "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("!"))


def bound_names(tree) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def all_source() -> str:
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
    """CLAUDE.md: notebooks are committed with outputs cleared."""
    for cell, _ in code_cells():
        assert cell["outputs"] == [] and cell["execution_count"] is None


def test_it_patches_with_the_d75_roles():
    src = all_source()
    assert "patch_example(model, source, target, head_dim)" in src
    assert "source, target = LastTokenRunner(model, src_in), LastTokenRunner(model, tgt_in)" in src
    # src_in is the original image (pair_inputs returns it first), both with the Visual prompt
    assert "return build_inputs(original, prompt), build_inputs(counterfact, prompt), (original, counterfact)" in src
    assert "prompt = VISUAL_PROMPT.format(object=r[\"name\"])" in src
    assert "PRIOR_PROMPT" not in src and "render_note" not in src


def test_it_selects_with_d75_and_caps_the_sweep():
    src = all_source()
    assert "is_v_conflicting(r, top1_orig, top1_cf, color_ids, ambiguous)" in src
    assert "v_sweep_order(selected, p_ids, seed=0, cap=V_CAP)" in src
    assert "shape_mismatch" in src and "numerics_failed" in src


def test_it_checkpoints_both_long_loops_under_its_own_prefix():
    """D44, and CLAUDE.md's one-prefix-per-notebook rule."""
    src = all_source()
    assert 'WORK = pathlib.Path("/kaggle/working/v_patching")' in src
    assert 'run_cached(WORK / "selection.jsonl"' in src
    assert 'run_cached(WORK / "records.jsonl"' in src
    assert "/kaggle/working/t_patching" not in src


def test_it_saves_the_files_the_analysis_script_reads():
    src = all_source()
    for name in ("v_patching_qwen25vl3b.json", "v_patching_records_qwen25vl3b.jsonl.gz",
                 "v_patching_selection_qwen25vl3b.jsonl"):
        assert name in src
    assert 'decision="D75"' in src


def test_the_freshness_check_names_exist_in_src():
    """Cell 1 refuses to run on a stale checkout; its list must be real names."""
    import importlib

    setup = next(s for _, s in code_cells())
    for mod, names in {
        "circuit_overlap.v_patching": ["is_v_conflicting", "v_sweep_order", "V_ROLES", "V_CAP"],
        "circuit_overlap.t_patching": ["finite_records", "has_nonfinite"],
        "circuit_overlap.patching": ["LastTokenRunner", "patch_example", "mean_restoration_grids",
                                     "classify_heads", "capture_last_token_o_proj_inputs",
                                     "patch_head_last_token"],
        "circuit_overlap.replication": ["lietzow_rows", "run_cached"],
    }.items():
        assert f'"{mod}"' in setup, mod
        m = importlib.import_module(mod)
        assert all(hasattr(m, n) for n in names), mod
