"""Static checks on notebooks/h3_attention.ipynb, which cannot run on the CPU.

Same checks as tests/test_cross_ablation_notebook.py -- above all, no name
used in a cell before an earlier cell defines it -- plus the pieces of D74
that live in the notebook rather than in src/.
"""

import ast
import builtins
import importlib
import inspect
import json
from pathlib import Path

from test_t_patching_notebook import bound_names

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "h3_attention.ipynb"


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


def test_setup_cells_match_the_ones_that_ran():
    """Install and image-size cells verbatim; the model load differs only in eager attention."""
    t_cells = ["".join(c["source"]) for c in json.loads((ROOT / "notebooks" / "t_patching.ipynb")
                                                        .read_text())["cells"]]
    ours = ["".join(c["source"]) for c in json.loads(NB.read_text())["cells"]]
    for k in (3, 5, 6):
        assert t_cells[k] == ours[k], f"t_patching cell {k} not copied verbatim"
    load = (ours[4].replace('ScreenConfig(attn_implementation="eager")   # H3 needs attention weights (D74 (3))',
                            "ScreenConfig()")
            .replace('\nprint("attention implementation:", getattr(model.config, "_attn_implementation", "?"))', ""))
    assert load == t_cells[4]


def test_d74_pieces():
    src = all_code()
    # sets and draws from D73's plan, never typed
    assert "build_plan(p_res, t_res, n_heads_per_layer=n_heads)" in src
    assert "L26H" not in src and "L33H" not in src and "(26," not in src
    # eager attention, layers 20-35, three layouts plus no note, P's two prompts
    assert 'attn_implementation="eager"' in src
    assert "LAYERS = list(range(FIRST_LAYER, n_layers))" in src
    assert 'LAYOUTS = ("on_object_small", "on_object_medium", "beside_object")' in src
    assert 'out["runs"]["clean"]' in src and "token_sets(small_mask, obj_mask" in src
    assert "PRIOR_PROMPT.format(a_object=_eng.a(r[\"name\"]))" in src
    assert "token_sets(None, obj_mask" in src and "with_note=False" in src
    # same answers as discovery, on the discovery layout (D74 (2))
    assert "finish(out, d, SMALL_INC, SMALL_NEU)" in src and 'finish(out, d, "prior", "visual")' in src
    # every D74 (10) check runs before the loops
    for check in ("check_attention_reconstructs_heads", "check_direct_effect_formula",
                  "max_weight_sum_error", "check_against_full_forward", "image_token_layout"):
        assert check in src, check
    assert src.index("check_attention_reconstructs_heads(model, runner0") < src.index('run_cached(CKPT / "h3_T.jsonl"')


def test_every_long_loop_checkpoints():
    src = all_code()
    for name in ('"h3_T.jsonl"', '"h3_P.jsonl"'):
        assert f"run_cached(CKPT / {name}" in src, name


def test_output_prefix_is_this_notebooks_own():
    src = all_code()
    assert 'pathlib.Path("/kaggle/working/h3_attention")' in src
    claude = (ROOT / "CLAUDE.md").read_text()
    assert "`h3_attention.ipynb` uses" in claude and "notebooks/h3_attention.ipynb" in claude


def test_saved_records_are_what_the_analysis_script_reads():
    src = all_code()
    assert '"h3_attention_records_qwen25vl3b.jsonl.gz"' in src and '"plan": plan' in src
    script = (ROOT / "scripts" / "analyze_h3.py").read_text()
    assert 'summary["result"]["plan"]' in script and 'r["condition"]' in script


def test_the_freshness_check_names_exist_in_src():
    cell1 = next(s for _, s in code_cells())
    tree = ast.parse(cell1)
    table = next(n for n in ast.walk(tree) if isinstance(n, ast.Dict)
                 and any(isinstance(k, ast.Constant) and k.value == "circuit_overlap.h3"
                         for k in n.keys))
    for key, value in zip(table.keys, table.values):
        mod = importlib.import_module(key.value)
        for elt in value.elts:
            assert hasattr(mod, elt.value), f"{key.value}.{elt.value}"
    # the attribute checks the cell does by source inspection must hold here too
    from circuit_overlap.patching import LastTokenRunner
    from circuit_overlap.t_patching import render_t_pair
    from circuit_overlap.t_screen import ScreenConfig
    assert hasattr(ScreenConfig, "attn_implementation")
    assert "prefix_position_ids" in inspect.getsource(LastTokenRunner)
    assert '"note_mask"' in inspect.getsource(render_t_pair)
