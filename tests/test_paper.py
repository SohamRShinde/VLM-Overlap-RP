"""The paper's numbers come from results/, and its text uses only defined macros."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def _paper_sources() -> list[Path]:
    """Every LaTeX source in paper/ except the generated numbers (draft and Prism's paper.tex)."""
    srcs = sorted(f for f in PAPER.glob("*.tex") if f.name != "numbers.tex")
    assert srcs, "no paper source in paper/"
    return srcs


def _load_numbers_module():
    spec = importlib.util.spec_from_file_location("paper_numbers", ROOT / "scripts" / "paper_numbers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_numbers_tex_is_current(tmp_path):
    out = _load_numbers_module().main(str(tmp_path / "numbers.tex"))
    assert out.read_text() == (PAPER / "numbers.tex").read_text(), \
        "paper/numbers.tex is stale: run scripts/paper_numbers.py and commit it"


def test_every_macro_the_text_uses_is_defined():
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", (PAPER / "numbers.tex").read_text()))
    tex = "\n".join(f.read_text() for f in _paper_sources())
    defined |= set(re.findall(r"\\newcommand\{\\(\w+)\}", tex))
    used = set(re.findall(r"\\([A-Za-z]+)", tex))
    # anything shaped like a generated macro must be generated
    ours = {u for u in used if re.match(
        r"^(n[A-Z]|hT[a-z]|J[A-Z]|chance|ceiling|rank[A-Z]|contain|topSingle|rep[A-Z]|lietzow[A-Z]"
        r"|attack|congruent|screen[A-Z]|enrich|noteShare|fool|prom[A-Z]|layers|paired|TselfJ"
        r"|pg[A-Z]|dJ|Vceiling)", u)}
    assert not ours - defined, f"undefined macros: {sorted(ours - defined)}"


def test_no_digits_typed_into_result_sentences():
    """Percentages in the text must be macros: no literal '12.3\\%' outside comments."""
    tex = "\n".join(l.split("%")[0] if not l.lstrip().startswith("%") else ""
                    for f in _paper_sources() for l in f.read_text().splitlines())
    assert not re.findall(r"\d+\.\d+\\%", tex)


def _load_flatten_module():
    spec = importlib.util.spec_from_file_location("flatten_paper", ROOT / "scripts" / "flatten_paper.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flatten_inlines_numbers_verbatim():
    fl = _load_flatten_module()
    numbers = "\\newcommand{\\JPT}{0.312}  % src\n\\newcommand{\\nP}{73}  % src\n"
    paper = "\\documentclass{article}\n\\input{numbers}\n\\begin{document}\\JPT\\end{document}\n"
    out = fl.flatten(paper, numbers)
    assert numbers.rstrip() in out and "\\input" not in out
    assert out.index("\\newcommand{\\JPT}") < out.index("\\begin{document}")


def test_flatten_refuses_retyped_or_extra_inputs():
    import pytest
    fl = _load_flatten_module()
    numbers = "\\newcommand{\\JPT}{0.312}\n"
    with pytest.raises(ValueError, match="redefines"):
        fl.flatten("\\newcommand{\\JPT}{0.31}\n\\input{numbers}\n", numbers)
    with pytest.raises(ValueError, match="exactly one"):
        fl.flatten("\\begin{document}\n", numbers)
    with pytest.raises(ValueError, match="another"):
        fl.flatten("\\input{numbers}\n\\input{intro}\n", numbers)
