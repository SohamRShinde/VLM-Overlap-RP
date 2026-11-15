# Paper source (first draft started 4 Nov 2026)

```bash
# 1. numbers: every result in the text is a macro generated from results/
.venv/bin/python scripts/paper_numbers.py        # writes paper/numbers.tex
# 2. figures
.venv/bin/python scripts/make_figures.py         # writes figures/
# 3. single submission file: inlines numbers.tex (AAAI wants one .tex)
.venv/bin/python scripts/flatten_paper.py paper/paper.tex   # writes paper/paper_flat.tex
```

- **Never type a result into `paper.tex`.** Add a macro to `scripts/paper_numbers.py`
  (with the file and key it comes from) and use it. `tests/test_paper.py` fails if
  `numbers.tex` is stale or the text uses an undefined macro.
- Verdict sentences are the pre-registered phrases (D70, D73, D74); they come in
  as macros too.
- `\todo{...}` marks anything unverified or unfinished; red in the PDF.
- `refs.bib`: every entry is checked against its source before submission;
  unconfirmed titles carry `[VERIFY]`.
- `paper.tex` (15 Nov) uses the AAAI-27 Author Kit (`AuthorKit27/`) for the
  Deployable AI workshop (D01); `draft_old.tex` is the 4 Nov draft it replaced.
  It is drafted in Prism with `paper/prism_prompts.md`. The local TeX install
  lacks `newtxtext.sty`, which `aaai2027.sty` needs, so it builds in Prism.
- Build outputs (`main.pdf`, `.aux`, ...) are git-ignored; build locally.
