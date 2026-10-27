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
- **Replication gate (D40), in progress.** Part 1 (group ablation of
  Lietzow's heads) ran on 25 Oct with an argmax flip definition: prior 90.4%,
  visual 6.8% (RESULTS.md). On 27 Oct D40 got numeric thresholds (PROPOSED,
  awaiting @SohamRShinde) and two runs were prepared, **not yet run**:
  `replication_flip_rate.ipynb` section 5 (Lietzow's pairwise flips, group and
  single heads, D67) and `replication_patching.ipynb` (re-discovery of the P
  heads over all 576). `scripts/analyze_replication_gate.py` applies D40 to
  their result files.
- No circuit discovery has produced a result yet.

## Two people, one repo
Two of us work on this from separate machines through the same GitHub repo.
- `git pull` before starting work, and **push every commit** -- the Kaggle
  notebooks clone from `origin/main`, so an unpushed commit means the GPU runs
  stale code. This has caused several confusing failures.
- Both of us edit `docs/`. Keep edits additive and commit doc changes on their
  own where practical, so conflicts stay trivial.

Each of us also runs a Claude Code session against this repo. These rules
apply to both people and both sessions:

- **Start every session with `git fetch && git status -sb`.** If the branch is
  behind, `git pull --ff-only` before reading or editing anything. A session
  that works on a stale checkout re-diagnoses bugs the other person already
  fixed, and its edits conflict on push (this happened on 20 Oct: 10 commits
  behind, see gotchas).
- **Before every push:** `git pull --rebase`, then `.venv/bin/python -m pytest`,
  then push. Never force-push `main`, and never rewrite pushed history.
- **One owner per notebook at a time.** `.ipynb` files are JSON and merge
  badly. Do not edit the other person's notebook; ask them, or change `src/`
  instead. @SohamRShinde owns the repo; kushviradiya is the
  partner on the other machine. Who runs each notebook:

  | Notebook | Run by |
  |---|---|
  | `notebooks/t_screen.ipynb` | @SohamRShinde |
  | `notebooks/replication_flip_rate.ipynb` | kushviradiya (the partner) |
  | `notebooks/replication_patching.ipynb` | kushviradiya (the partner) |

  Update this table in the same commit when ownership changes. Commit
  notebooks with **outputs cleared**; results go to `results/`, not cell
  output.
- **Shared code in `src/` is used by both notebooks**, so a change there can
  silently change the other person's next run (cell 1 pulls `origin/main`).
  Before renaming, removing, or changing the behaviour of a public function,
  grep `notebooks/` for its callers. If the other person's notebook needs a
  new name, add it to that notebook's cell-1 freshness check. Behaviour
  changes to code the other person is running go on a branch with a PR they
  review; tested additions and fixes can go straight to `main`.
- **Never push to `main` during the other person's Save & Run All without
  telling them** if it touches code their notebook imports -- a re-run of
  cell 1 picks it up. Record the commit hash cell 1 prints with every result.
- **`DECISIONS.md` IDs collide.** Pull immediately before taking the next
  D-number, and push the decision promptly. On a conflict, the later commit
  renumbers and fixes its references.
- **Results files are per experiment and never overwritten by the other
  person.** Name them `results/<experiment>_<model>.json`; append rows to
  `RESULTS.md`, never reorder or rewrite another person's rows.
- **Kaggle output paths are per notebook.** `t_screen.ipynb` uses
  `/kaggle/working/vc_color` and `/kaggle/working/t_screen_*`;
  `replication_flip_rate.ipynb` uses `/kaggle/working/replication/`. A new
  notebook gets its own prefix.

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

## Claude's own mistakes (20 Oct 2026, replication notebook)
Recorded so neither session repeats them.
- **Worked on a checkout 10 commits behind `origin/main`.** The session
  diagnosed a 16-hour Kaggle run from stale code and missed the likely cause
  (emulated bf16, already fixed upstream as `native_bf16`), then wrote cells
  using `torch.cuda.is_bf16_supported()` despite the gotcha above. Fix: fetch
  first, and read the gotchas before writing GPU code.
- **Wrong runtime estimate, no checkpointing.** It said "a few minutes" for
  ~940 generations and gave a loop that kept results only in memory. The run
  took 16+ h, the session died, and everything was lost. Fix: time one
  forward pass first (the replication notebook's section 1b does), multiply
  out, say the total, and checkpoint every long loop (D44) from the start.
- **Gave untested code as if finished.** The first conflicting-subset cell
  matched generated text against colour names, which breaks on answers like
  "The color is red" and costs 8 decode steps per answer. Top-1 comparisons
  use the argmax of the first answer-token logits from one forward pass.
  Label any cell that has not run on the real model as untested.
- **Notebooks drifted apart.** `replication_flip_rate.ipynb` had no
  `pip install bitsandbytes` cell while `t_screen.ipynb` did, and installing
  it into a kernel that had already imported transformers does nothing until
  the kernel restarts (transformers caches the check). New notebooks copy the
  install cell, and the fix for a fresh install is a restart, not a re-install.
- **`head_knockout` is re-implemented, not Lietzow's file** (D65 says reuse
  theirs). Diffed on 27 Oct against their commit 145b453: same slice, same
  position.
- **Did not check the flip definition against their code.** The 25 Oct run
  counted any top-1 change as a flip; Lietzow's flip is pairwise (logit of the
  prior answer vs the visual answer, D67), and 4 of the 5 "visual flips" went
  to a third colour. Before reporting a number "comparable to Lietzow", read
  the code that produced theirs -- the paper's wording is not enough.
