"""Static checks on notebooks/pg_t_patching.ipynb (D77), which cannot run on the CPU."""

import ast
import builtins
import gzip
import importlib
import json
import subprocess
import sys
from pathlib import Path

from test_t_patching_notebook import bound_names

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "pg_t_patching.ipynb"


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


def test_setup_and_checks_are_the_ones_that_ran():
    pg = ["".join(c["source"]) for c in json.loads((ROOT / "notebooks" / "pg_replication.ipynb").read_text())["cells"]]
    ours = ["".join(c["source"]) for c in cells()]
    assert ours[3] == pg[3] and ours[5] == pg[5], "install / model cells not copied verbatim"
    # the checks cell differs only in its folder, resume file and estimate
    for piece in ("labels_max_dlogit", "promoting_ablation_max_dlogit", "logits_to_keep_max_dlogit",
                  "same_note_footprint", "square_image_block", "BATCH = min(per_row"):
        assert piece in ours[7], piece


def test_d77_pieces():
    src = all_code()
    assert 'pathlib.Path("/kaggle/working/pg_t_patching")' in src
    assert "results/pg_selection_paligemma2_3b.json" in src and "assert len(selected) == 302" in src
    assert "sweep_order(selected, p_ids, seed=0)" in src                    # D71 (4)
    assert "patch_example_batched(model, src_in, tgt_in, head_dim" in src  # D76a
    assert "LastTokenRunner" not in src                                    # full forward only
    assert "selection_drift" in src and "has_nonfinite(full)" in src
    assert src.index("batched_vs_unbatched_max_dlogit") < src.index('run_cached(CKPT / "records.jsonl"')
    assert "budget_s=PATCH_BUDGET_H * 3600" in src
    assert "L15H" not in src and "L20H" not in src and "(15," not in src
    for name in ("pg_t_patching_paligemma2_3b.json", "pg_t_patching_records_paligemma2_3b.jsonl.gz"):
        assert name in src


def test_output_prefix_and_owner_are_in_claude_md():
    claude = (ROOT / "CLAUDE.md").read_text()
    assert "`pg_t_patching.ipynb` uses" in claude and "notebooks/pg_t_patching.ipynb" in claude


def test_the_freshness_check_names_exist_in_src():
    cell1 = next(s for _, s in code_cells())
    table = next(n for n in ast.walk(ast.parse(cell1)) if isinstance(n, ast.Dict)
                 and any(isinstance(k, ast.Constant) and k.value == "circuit_overlap.paligemma"
                         for k in n.keys))
    for key, value in zip(table.keys, table.values):
        mod = importlib.import_module(key.value)
        for elt in value.elts:
            assert hasattr(mod, elt.value), f"{key.value}.{elt.value}"


def test_overlap_wrapper_refuses_fewer_than_60_objects(tmp_path):
    """D77 (3). Qwen's files stand in for the format; only the count matters here."""
    with gzip.open(ROOT / "results" / "t_patching_records_qwen25vl3b.jsonl.gz", "rt") as f:
        lines = [l for l in f if l.strip()][:30]
    t = tmp_path / "t.jsonl"
    t.write_text("".join(lines))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_pg_t_circuit.py"),
                        str(ROOT / "results" / "replication_patching_qwen25vl3b.json"), str(t),
                        str(ROOT / "results" / "split_half_ceiling_p_qwen25vl3b.json")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "no D70 verdict" in r.stdout and "VERDICT" not in r.stdout
