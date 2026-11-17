"""Static checks on notebooks/pg_v_patching.ipynb (D79), which cannot run on the CPU."""

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
NB = ROOT / "notebooks" / "pg_v_patching.ipynb"


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


def test_install_and_model_cells_are_the_ones_that_ran():
    t = ["".join(c["source"]) for c in json.loads((ROOT / "notebooks" / "pg_t_patching.ipynb").read_text())["cells"]]
    ours = ["".join(c["source"]) for c in cells()]
    assert ours[3] == t[3] and ours[5] == t[5], "install / model cells not copied verbatim"
    for piece in ("labels_max_dlogit", "promoting_ablation_max_dlogit", "logits_to_keep_max_dlogit",
                  "square_image_block", "BATCH = min(per_row", "last_token_reaches_earlier_positions"):
        assert piece in ours[7], piece


def test_d79_pieces():
    src = all_code()
    assert 'pathlib.Path("/kaggle/working/pg_v_patching")' in src
    assert 'run_cached(CKPT / "selection.jsonl", rows, select_object' in src           # all 469
    assert "is_v_conflicting(r, top1_orig, top1_cf, color_ids, ambiguous)" in src
    assert "v_sweep_order(selected, p_ids, seed=0, cap=V_CAP)" in src                  # D75 (4)
    assert 'build_inputs(original, prompt), build_inputs(counterfact, prompt)' in src  # source, target
    assert "LastTokenRunner" not in src                                                # full forward only
    assert "selection_drift" in src and "has_nonfinite(full)" in src
    assert "budget_s=PATCH_BUDGET_H * 3600" in src
    assert "L15H" not in src and "L20H" not in src and "(15," not in src


def test_replay_is_checked_against_the_batched_path_before_the_sweep():
    src = all_code()
    sweep = src.index('run_cached(CKPT / "records.jsonl"')
    assert src.index('VERIFY["replay_vs_batched"]') < sweep
    assert "USE_REPLAY = same_pair and diff < 0.5" in src
    # every head, both directions: no layers= restriction on either side of the check
    assert "fast = patch_example_replay(model, src1, tgt1, head_dim, batch_size=n_heads)" in src
    assert "slow = patch_example_batched(model, src1, tgt1, head_dim, batch_size=BATCH)" in src
    # the per-object guard and the fallback
    assert "REPLAY_TOL = 0.1" in src and 'record["replay_check"] > REPLAY_TOL' in src
    assert '"batched_fallback"' in src


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
    doc = (ROOT / "scripts" / "analyze_pg_v_circuit.py").read_text()
    assert "pg_v_patching_records_paligemma2_3b.jsonl.gz" in src and "pg_v_patching_records_paligemma2_3b.jsonl.gz" in doc
    assert "pg_v_patching_paligemma2_3b.json" in src
    for name in ("pg_replication_patching_paligemma2_3b.json", "pg_t_patching_records_paligemma2_3b.jsonl.gz",
                 "split_half_ceiling_p_paligemma2_3b.json", "pg_t_patching_paligemma2_3b.json"):
        assert name in doc and (ROOT / "results" / name).exists(), name


def test_output_prefix_and_owner_are_in_claude_md():
    claude = (ROOT / "CLAUDE.md").read_text()
    assert "`pg_v_patching.ipynb` uses" in claude and "notebooks/pg_v_patching.ipynb" in claude


def test_wrapper_gives_no_verdict_below_60(tmp_path):
    t_path = ROOT / "results" / "pg_t_patching_records_paligemma2_3b.jsonl.gz"
    with gzip.open(t_path, "rt") as f:
        some = [json.loads(l) for l, _ in zip(f, range(80))]
    some = [r for r in some if r.get("status") == "patched"][:10]
    fake = tmp_path / "v.jsonl.gz"
    with gzip.open(fake, "wt") as f:
        for r in some:
            f.write(json.dumps({**r, "replay_check": 0.0, "method": "replay"}) + "\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_pg_v_circuit.py"), str(fake)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "10 PaliGemma V objects patched" in r.stdout
    assert "no D75 verdict" in r.stdout and "VERDICT" not in r.stdout
