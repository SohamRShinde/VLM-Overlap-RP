"""scripts/make_figures.py draws every figure from the committed results."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]


def test_every_figure_is_written_from_the_committed_results(tmp_path):
    spec = importlib.util.spec_from_file_location("make_figures", ROOT / "scripts" / "make_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    written = mod.main(str(tmp_path))
    assert len(written) == 2 * len(mod.FIGURES)
    for p in written:
        assert p.exists() and p.stat().st_size > 1000, p


def test_no_head_index_is_typed_into_the_figure_script():
    src = (ROOT / "scripts" / "make_figures.py").read_text()
    assert "L26H" not in src and "L31H" not in src and "L33H" not in src
