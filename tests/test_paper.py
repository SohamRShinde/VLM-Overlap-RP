"""The paper's numbers come from results/, and its text uses only defined macros."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


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
    tex = (PAPER / "main.tex").read_text()
    defined |= set(re.findall(r"\\newcommand\{\\(\w+)\}", tex))
    used = set(re.findall(r"\\([A-Za-z]+)", tex))
    # anything shaped like a generated macro must be generated
    ours = {u for u in used if re.match(
        r"^(n[A-Z]|hT[a-z]|J[A-Z]|chance|ceiling|rank[A-Z]|contain|topSingle|rep[A-Z]|lietzow[A-Z]"
        r"|attack|congruent|screen[A-Z]|enrich|noteShare|fool|prom[A-Z]|layers|paired|TselfJ)", u)}
    assert not ours - defined, f"undefined macros: {sorted(ours - defined)}"


def test_no_digits_typed_into_result_sentences():
    """Percentages in the text must be macros: no literal '12.3\\%' outside comments."""
    tex = "\n".join(l.split("%")[0] if not l.lstrip().startswith("%") else ""
                    for l in (PAPER / "main.tex").read_text().splitlines())
    assert not re.findall(r"\d+\.\d+\\%", tex)
