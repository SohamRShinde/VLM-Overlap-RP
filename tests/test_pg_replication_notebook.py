"""Static checks on notebooks/pg_replication.ipynb (D76), which cannot run on the CPU."""

import ast
import builtins
import importlib
import json
from pathlib import Path

from test_t_patching_notebook import bound_names

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "pg_replication.ipynb"


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


def test_setup_cells_match_the_ones_that_ran():
    h3 = ["".join(c["source"]) for c in json.loads((ROOT / "notebooks" / "h3_attention.ipynb").read_text())["cells"]]
    ours = ["".join(c["source"]) for c in cells()]
    assert ours[3] == h3[3], "install cell not copied verbatim"
    head = h3[2][:h3[2].index("for _mod, _names in {")]
    tail = h3[2][h3[2].index("if _need:\n    raise RuntimeError"):]
    assert ours[2].startswith(head) and ours[2].endswith(tail)


def test_d76_method_pieces():
    src = all_code()
    assert "FullForwardRunner(model" in src and "LastTokenRunner" not in src    # D76 (2)
    assert "last_token_reaches_earlier_positions" in src
    assert "configs/heads_lietzow_paligemma2_3b.json" in src
    assert "L15H" not in src and "L20H" not in src and "(15," not in src      # heads are data
    assert "is_correctly_conflicting(r" in src and "is_t_conflicting(r" in src
    assert "text_geometry(model)" in src and "square_image_block" in src
    for check in ("empty_ablation_max_dlogit", "promoting_ablation_max_dlogit", "same_note_footprint"):
        assert check in src
    # every check runs before the first long loop
    assert src.index("promoting_ablation_max_dlogit") < src.index('run_cached(CKPT / "select.jsonl"')


def test_every_long_loop_checkpoints():
    src = all_code()
    for name in ("select.jsonl", "knockout.jsonl", "patching.jsonl"):
        assert f'run_cached(CKPT / "{name}"' in src, name


def test_output_prefix_is_this_notebooks_own():
    src = all_code()
    assert 'pathlib.Path("/kaggle/working/pg_replication")' in src
    claude = (ROOT / "CLAUDE.md").read_text()
    assert "`pg_replication.ipynb` uses" in claude and "notebooks/pg_replication.ipynb" in claude


def test_saved_files_are_what_the_analysis_script_reads():
    src = all_code()
    doc = (ROOT / "scripts" / "analyze_pg_replication.py").read_text()
    for name in ("pg_selection_paligemma2_3b.json", "pg_replication_knockout_paligemma2_3b.json",
                 "pg_replication_patching_paligemma2_3b.json"):
        assert name in src and name in doc, name
    for key in ('"color_ids"', '"ambiguous_colours"', '"reference_promoting"', '"configs"', '"records"'):
        assert key in src, key


def test_the_freshness_check_names_exist_in_src():
    cell1 = next(s for _, s in code_cells())
    table = next(n for n in ast.walk(ast.parse(cell1)) if isinstance(n, ast.Dict)
                 and any(isinstance(k, ast.Constant) and k.value == "circuit_overlap.paligemma"
                         for k in n.keys))
    for key, value in zip(table.keys, table.values):
        mod = importlib.import_module(key.value)
        for elt in value.elts:
            assert hasattr(mod, elt.value), f"{key.value}.{elt.value}"
