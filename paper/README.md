# Paper source (first draft started 4 Nov 2026)

```bash
# 1. numbers: every result in the text is a macro generated from results/
.venv/bin/python scripts/paper_numbers.py        # writes paper/numbers.tex
# 2. figures
.venv/bin/python scripts/make_figures.py         # writes figures/
# 3. PDF (run from paper/)
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

- **Never type a result into `main.tex`.** Add a macro to `scripts/paper_numbers.py`
  (with the file and key it comes from) and use it. `tests/test_paper.py` fails if
  `numbers.tex` is stale or the text uses an undefined macro.
- Verdict sentences are the pre-registered phrases (D70, D73, D74); they come in
  as macros too.
- `\todo{...}` marks anything unverified or unfinished; red in the PDF.
- `refs.bib`: every entry is checked against its source before submission;
  unconfirmed titles carry `[VERIFY]`.
- Layout is a plain two-column article sized like AAAI. Swap in the venue's
  template once the workshop is confirmed (D01). Times is not installed locally
  (`texlive-fontsrecommended`), so the local preview uses Computer Modern.
- Build outputs (`main.pdf`, `.aux`, ...) are git-ignored; build locally.
