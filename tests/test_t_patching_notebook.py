"""Static checks on notebooks/t_patching.ipynb, which cannot run on the CPU.

The GPU cells only ever run on Kaggle, hours into a session. The failure that
costs the most there is a name used in one cell that no earlier cell defines
(the T screen's save cell died on exactly that, ``NameError: EXCLUDE``). This
walks the cells in order and fails on any such name, so it is caught here in
a second instead of after a night of compute.
"""

import ast
import builtins
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "t_patching.ipynb"


def code_cells():
    nb = json.loads(NB.read_text())
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


def test_it_patches_with_the_d71_roles_and_layout():
    src = "\n".join(s for _, s in code_cells())
    assert "patch_example(model, source, target, head_dim)" in src
    assert "source, target = LastTokenRunner(model, src_in), LastTokenRunner(model, tgt_in)" in src
    # src_in is the incongruent render (pair_inputs returns it first)
    assert "return build_inputs(pr[\"incongruent\"], prompt), build_inputs(pr[\"neutral\"], prompt), pr" in src
    assert "LAYOUT" in src and "sweep_order(selected, p_ids, seed=0)" in src


def test_it_checkpoints_both_long_loops():
    """D44: a run this long must resume, not restart."""
    src = "\n".join(s for _, s in code_cells())
    assert 'run_cached(WORK / "selection.jsonl"' in src
    assert 'run_cached(WORK / "records.jsonl"' in src


def test_the_freshness_check_names_exist_in_src():
    """Cell 1 refuses to run on a stale checkout; its list must be real names."""
    import importlib

    for mod, names in {
        "circuit_overlap.t_patching": ["render_t_pair", "is_t_conflicting", "sweep_order",
                                       "finite_records", "screen_cross_check", "LAYOUT", "T_ROLES"],
        "circuit_overlap.patching": ["LastTokenRunner", "patch_example", "mean_restoration_grids",
                                     "classify_heads", "capture_last_token_o_proj_inputs",
                                     "patch_head_last_token"],
        "circuit_overlap.replication": ["lietzow_rows", "run_cached"],
    }.items():
        m = importlib.import_module(mod)
        assert all(hasattr(m, n) for n in names), mod
