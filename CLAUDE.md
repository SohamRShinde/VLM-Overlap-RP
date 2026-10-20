# Circuit overlap project

Research code for a mechanistic interpretability paper.
Question: do the heads that implement prompted prior override (Lietzow et al.
2606.28273) overlap with the heads that let in-image text override vision?

**Plan of record: @docs/plan_v2.md** (v2.1, 16 Oct 2026). Read it before
proposing experiments; it supersedes any earlier plan.

## Read these first, every session
- `docs/STATE.md`     current status, blockers, open questions, assumptions in code
- `docs/DECISIONS.md` dated decisions and pre-registered thresholds (D01-D66)
- `docs/RESULTS.md`   every reported number, with model, n, seed, prompt, commit

They are living documents: keep them current, never overwrite wholesale, and
date what you add. STATE.md has a table of everything Lietzow's own repo
settled (prompt strings, checkpoints, indexing, loader filters) -- consult it
before re-deriving any of that.

**Venue: undecided.** AIR-FM was the AAAI-2026 workshop and is gone; plan v2.1
section 0.8 proposes GSR-VLM but that was never agreed (DECISIONS.md D01,
reopened). Do not state a venue, page limit, or framing as settled.

## Status, 20 Oct 2026
- **T condition: done.** Behavioural screen passed its gate (286 usable objects
  of 436 eligible, 65.6%). Per-record data in `results/`. See RESULTS.md.
- **Next: the replication gate** (plan section 4 step 2), starting with
  `notebooks/replication_flip_rate.ipynb`, which still has `[FILL IN]` cells
  for building the conflicting subset. Pass criteria: DECISIONS.md D40; target
  numbers in `configs/heads_lietzow_qwen3b.json` and plan Appendix B.
- No circuit discovery has been run. Nothing has been patched or ablated.

## Two people, one repo
Two of us work on this from separate machines through the same GitHub repo.
- `git pull` before starting work, and **push every commit** -- the Kaggle
  notebooks clone from `origin/main`, so an unpushed commit means the GPU runs
  stale code. This has caused several confusing failures.
- Both of us edit `docs/`. Keep edits additive and commit doc changes on their
  own where practical, so conflicts stay trivial.

## Setup
- Models: Qwen2.5-VL-3B-Instruct (primary), PaliGemma-3B. **4-bit nf4**,
  T4-compatible (D10 -- matches Lietzow's default; not fp16, despite what plan
  section 3.3 says).
- No local GPU. Real experiments run on Kaggle; here we only write code and
  test on CPU.
- Local env: Python 3.11 in `.venv/` (git-ignored). numpy, pillow, pytest only.
  torch/transformers/nnsight are deliberately NOT installed locally -- they run
  on Kaggle.
- `src/` is not pip-installed. pytest puts it on the import path via
  `pyproject.toml`; Kaggle notebooks do `sys.path.insert(0, ".../src")`.

## Commands
```bash
# create the environment (once)
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

# run all tests - do this before suggesting any Kaggle run
.venv/bin/python -m pytest

# one file, or one test, with verbose output
.venv/bin/python -m pytest tests/test_metrics.py -v
.venv/bin/python -m pytest tests/test_metrics.py::test_jaccard_partial_overlap_hand_checked -v

# regenerate every reported T-screen number from the saved results
.venv/bin/python scripts/analyze_t_screen.py results/t_screen_qwen25vl3b.json
```

## Layout
- `src/circuit_overlap/`  metrics, head sets, note generator, mask->token
  mapping, screen scoring, manifest, provenance, Kaggle driver
- `configs/`    experiment configs; `heads_lietzow_*.json` are the published
  head lists, transcribed from the paper's Tables 5-7
- `tests/`      CPU-only unit tests (no model, no network)
- `results/`    JSON from Kaggle runs, committed
- `notebooks/`  thin Kaggle wrappers that call `src/`
- `scripts/`    analysis that reads `results/` and prints reportable numbers
- `docs/`       plan and living documents

## Kaggle workflow
- Notebook cell 1 resets the checkout to `origin/main`, purges cached
  `circuit_overlap` modules, prints the commit, and **fails loudly** if the
  checkout is older than the notebook expects. Do not weaken that.
- Needs a Kaggle secret `HF_TOKEN` (the dataset is token-gated) and
  `pip install bitsandbytes` for the 4-bit load.
- **Long runs must checkpoint** one JSON line per item and resume by skipping
  completed ids (D44). A 469-object screen takes hours; a lost session must not
  cost the run. Run them as **Save Version -> Save & Run All (Commit)**, not
  interactively.
- Results come back by hand: download the JSONs from the Kaggle version output,
  put them in `results/`, commit. There is a disabled token-based auto-push
  cell at the end of `notebooks/t_screen.ipynb`; it is off on purpose.

## Rules
- Zero-ablate head outputs at the last token position (matches Lietzow:
  `o_proj` input slice `[head*head_dim : (head+1)*head_dim]`).
- Never hard-code head indices from memory; load from `configs/heads_*.json`
  via `heads.load_head_set`. Heads are 0-based `(layer, head)` (D30, D60).
- Never change the prompt strings casually. The verified Visual prompt is
  `What color is this {object} here? Respond with the color name only.` -- the
  answer-format suffix appears in neither the paper's section 3.1 nor its
  Table 1 caption and was only found in their repo (D22).
- Run `.venv/bin/python -m pytest` before suggesting a Kaggle run.
- Never invent results. Numbers reach RESULTS.md only from a file in
  `results/`, via a script. Reference numbers from other papers are not
  results (see the RESULTS.md header).
- Large artifacts (weights, activations, data, PDFs) stay out of git; small
  JSON results and configs stay in.
- Pre-register thresholds in DECISIONS.md **before** looking at the data they
  judge (D38).

## Gotchas that have already cost a session each
- **Stale Kaggle checkout.** Symptom: `ImportError` or `TypeError` for a name
  that exists locally. Cause: unpushed commits. Push, then re-run cell 1.
- **bf16 on a T4.** `torch.cuda.is_bf16_supported()` returns True on Turing
  because it counts *emulation*, which is slow. Use `t_screen.native_bf16`,
  which requires Ampere or newer.
- **The processor resizes images.** Map masks onto the grid the model actually
  sees: read `image_grid_thw` and use `tokens.coverage_on_grid` /
  `mask_to_token_indices_on_grid`. `token_grid_shape` only applies to an
  already-resized image.
- **Half-precision overflow is real** (plan risk 7). 5 objects returned
  non-finite logits, one generating `'!!!!'`. `screen.has_numerics_failure`
  excludes them from every rate; never let a numerics failure count as a wrong
  answer.
- **The dataset's own loader silently drops rows.** Lietzow's pydantic model
  rejects plural and parenthetical object names and rows whose counterfactual
  colour is also correct, which is how 493 Hub rows become their 469/467.
  `data.lietzow_object_name` reproduces it. Which revision to pin is still open
  (D20a).
