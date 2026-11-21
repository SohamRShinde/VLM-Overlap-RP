# Paper source (first draft started 4 Nov 2026)

**Submitted 21 Nov 2026** to the Deployable AI workshop from commit `95ba7cd`.
Edits after that are for a camera-ready or a later version.

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
- `\todo{...}` marked anything unverified or unfinished. All were resolved on 16 Nov
  and the `\todo` / `\figplaceholder` definitions removed for submission; re-add
  them to the preamble if drafting resumes.
- `refs.bib`: every entry is checked against its source before submission;
  unconfirmed titles carry `[VERIFY]`.
- `paper.tex` (15 Nov) uses the AAAI-27 Author Kit (`AuthorKit27/`) for the
  Deployable AI workshop (D01); `draft_old.tex` is the 4 Nov draft it replaced.
  It is drafted in Prism with `paper/prism_prompts.md`. The local TeX install
  lacks `newtxtext.sty`, which `aaai2027.sty` needs, so it builds in Prism.
- Build outputs (`main.pdf`, `.aux`, ...) are git-ignored; build locally.
- **Local build (16 Nov, MiKTeX 25.12, packages auto-installed):** copy `paper.tex`,
  `numbers.tex`, `refs.bib`, `AuthorKit27/aaai2027.sty`, `AuthorKit27/aaai2027.bst` and
  `figures/*.pdf` (into a `figures/` subfolder) into one folder, then
  `pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`. The same folder
  layout is what Prism/Overleaf needs: if those files are missing there, every result
  macro prints as `[JPT]`-style text and citations as `[?]`. `\graphicspath` also finds
  `../figures/` when building inside `paper/`.
- Page budget (Deployable AI long track): 7 pages of main text; references and the
  appendices after them do not count. As of 16 Nov the main text ends on page 7; in the submitted version (21 Nov) too.
- Sections are unnumbered (`secnumdepth` 0, the AAAI default), so `Section~\ref{...}`
  prints an empty number; refer to a section by name instead.
