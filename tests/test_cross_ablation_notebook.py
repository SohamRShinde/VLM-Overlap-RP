"""Static checks on notebooks/cross_ablation.ipynb, which cannot run on the CPU.

Same checks as tests/test_t_patching_notebook.py -- above all, no name used
in a cell before an earlier cell defines it -- plus the pieces of D73 that
live in the notebook rather than in src/.
"""

import ast
import builtins
import importlib
import json
from pathlib import Path

from test_t_patching_notebook import bound_names

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "cross_ablation.ipynb"


def code_cells():
    for cell in json.loads(NB.read_text())["cells"]:
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


def test_setup_cells_match_t_patching():
    """Install, model load and image-size cells are the ones that ran on Kaggle."""
    t_cells = ["".join(c["source"]) for c in json.loads((ROOT / "notebooks" / "t_patching.ipynb")
                                                        .read_text())["cells"]]
    ours = ["".join(c["source"]) for c in json.loads(NB.read_text())["cells"]]
    for k in (3, 4, 5, 6):
        assert t_cells[k] in ours, f"t_patching cell {k} not copied verbatim"


def test_d73_pieces():
    src = all_code()
    # sets and draws come from the plan, never typed
    assert "build_plan(p_res, t_res, n_heads_per_layer=n_heads)" in src
    assert "L26H" not in src and "(26," not in src
    # same inputs as discovery: P source = Prior prompt, T source = incongruent note
    assert 'build_inputs(pr["incongruent"], prompt), build_inputs(pr["neutral"], prompt)' in src
    assert "PRIOR_PROMPT.format(a_object=_eng.a(r[\"name\"]))" in src
    # drift check against discovery answers (D73 (2))
    assert 'if a != tuple(disc[cond][r["i"]]):' in src
    # mean ablation over both runs of the condition (D73 (10))
    assert 'for run in ("source", "target"):' in src and 'key_prefix="mean:"' in src
    # the ablation is checked against Kush's hook and his record
    assert "stacked_vs_kush_hook" in src and "grp_promoting/prior" in src


def test_every_long_loop_checkpoints():
    src = all_code()
    for name in ('f"zero_{cond}.jsonl"', 'f"mean_{cond}.jsonl"', '"clean.jsonl"'):
        assert f"run_cached(CKPT / {name}" in src, name


def test_saved_records_are_what_the_analysis_script_reads(tmp_path):
    src = all_code()
    assert '"cross_ablation_records_qwen25vl3b.jsonl.gz"' in src
    assert '"plan": plan' in src
    script = (ROOT / "scripts" / "analyze_cross_ablation.py").read_text()
    assert 'summary["result"]["plan"]' in script and 'r["condition"]' in script


def test_the_freshness_check_names_exist_in_src():
    cell1 = next(s for _, s in code_cells())
    tree = ast.parse(cell1)
    table = next(n for n in ast.walk(tree) if isinstance(n, ast.Dict)
                 and any(isinstance(k, ast.Constant) and k.value == "circuit_overlap.cross_ablation"
                         for k in n.keys))
    for key, value in zip(table.keys, table.values):
        mod = importlib.import_module(key.value)
        for elt in value.elts:
            assert hasattr(mod, elt.value), f"{key.value}.{elt.value}"
