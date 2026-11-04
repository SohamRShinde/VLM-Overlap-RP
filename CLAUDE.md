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
- `docs/HANDOVER.md`  3 Nov handover: what is done, the two parallel tracks
  (V condition; paper and figures) and who owns each

They are living documents: keep them current, never overwrite wholesale, and
date what you add. STATE.md has a table of everything Lietzow's own repo
settled (prompt strings, checkpoints, indexing, loader filters) -- consult it
before re-deriving any of that.

**Paper: 6 pages, due 20 Nov 2026** (DECISIONS.md D01a, fixed 31 Oct). The
workshop itself is still unconfirmed: AIR-FM was the AAAI-2026 edition and is
gone, and GSR-VLM is plan v2.1's candidate but was never agreed (D01). Do not
state a workshop name or framing as settled.

## Status, 2 Nov 2026
- **T condition: done.** Behavioural screen passed its gate (286 usable objects
  of 436 eligible, 65.6%). Per-record data in `results/`. See RESULTS.md.
- **Replication gate (D40): PASSED** on all six criteria (thresholds
  proposed 27 Oct, approved 31 Oct by @SohamRShinde). Both runs at `90286fa`:
  patching over all 576 heads finds exactly Lietzow's 9 promoting and 6
  suppressing heads; pairwise knockout (D67) flips 89.0% of prior answers
  and 2.7% of visual ones, and L26H5 and L31H3 are the top single heads.
  Numbers and caveats in RESULTS.md; `scripts/analyze_replication_gate.py`
  reproduces the verdict. Next: T-circuit discovery.
- **P split-half ceiling: done** (31 Oct). Promoting-set J between halves
  median 0.900 vs chance 0.059. D70, the rule for reading J_PT against it,
  approved 31 Oct before any T-circuit result existed.
- **T-circuit discovery: done** (1 Nov, notebook at `3b73bb5`, design D71):
  244 objects, 12 promoting heads. **Headline J_PT = 0.312 [0.278, 0.375],
  D70 verdict "above chance, below self-agreement"** (5 of 16 heads shared).
  See RESULTS.md.
- **H2 cross-ablation: done** (1 Nov, run at `8208422`, design D73). Both
  primary tests null (H_P on T 2.5%, H_T on P 2.7%, random ~4-5%). Shared
  heads carry P (83.6%) but not T (0%); T's own heads carry T (70.5%).
  Overlap in selection, not in cause. See RESULTS.md.
- **H3 mechanism: done** (2 Nov, run at `49f0152`, design D74, eager
  attention). Both primary tests pass: the 7 T-only heads attend to the note
  and write its colour; with no note they attend to the recoloured object and
  write the image's colour. Shared heads do neither. "Object" tokens are the
  recoloured pixels only. See RESULTS.md.
- **V condition (track A): built, not yet run** (4 Nov, kushviradiya).
  Design D75 (pushed as `8aa9307` before any code); `notebooks/v_patching.ipynb`
  and `scripts/analyze_v_circuit.py` are CPU-tested only.

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
  | `notebooks/t_patching.ipynb` | @SohamRShinde |
  | `notebooks/cross_ablation.ipynb` | @SohamRShinde |
  | `notebooks/h3_attention.ipynb` | @SohamRShinde |
  | `notebooks/v_patching.ipynb` | kushviradiya (the partner) |

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
  `replication_flip_rate.ipynb` uses `/kaggle/working/replication/`;
  `replication_patching.ipynb` uses `/kaggle/working/patching/`;
  `t_patching.ipynb` uses `/kaggle/working/t_patching/`;
  `cross_ablation.ipynb` uses `/kaggle/working/cross_ablation/` (results at
  its top level, checkpoints under `ckpt/`); `h3_attention.ipynb` uses
  `/kaggle/working/h3_attention/` (same layout); `v_patching.ipynb` uses
  `/kaggle/working/v_patching/`. A new notebook gets its own prefix.

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

# P split-half ceiling (D69), CPU only
.venv/bin/python scripts/analyze_split_half.py results/replication_patching_qwen25vl3b.json

# THE HEADLINE: P-T circuit overlap read by D70 (computed per D72), CPU only.
# Written before the T results existed; do not change its parameters.
.venv/bin/python scripts/analyze_t_circuit.py \
    results/replication_patching_qwen25vl3b.json \
    results/t_patching_records_qwen25vl3b.jsonl.gz \
    results/split_half_ceiling_p_qwen25vl3b.json \
    results/t_circuit_overlap_qwen25vl3b.json

# H2 cross-ablation read by D73, CPU only. Written before any H2 run.
.venv/bin/python scripts/analyze_cross_ablation.py \
    results/cross_ablation_qwen25vl3b.json \
    results/cross_ablation_records_qwen25vl3b.jsonl.gz \
    results/cross_ablation_analysis_qwen25vl3b.json

# H3 mechanism read by D74, CPU only. Written before any H3 run.
.venv/bin/python scripts/analyze_h3.py \
    results/h3_attention_qwen25vl3b.json \
    results/h3_attention_records_qwen25vl3b.jsonl.gz \
    results/h3_analysis_qwen25vl3b.json

# V condition read by D75: J_PV, J_VT and dJ = J_VT - J_PT, CPU only.
# Written before any V run; do not change its parameters.
.venv/bin/python scripts/analyze_v_circuit.py \
    results/replication_patching_qwen25vl3b.json \
    results/t_patching_records_qwen25vl3b.jsonl.gz \
    results/v_patching_records_qwen25vl3b.jsonl.gz \
    results/split_half_ceiling_p_qwen25vl3b.json \
    results/t_patching_qwen25vl3b.json \
    results/v_circuit_overlap_qwen25vl3b.json
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
- **CPU test inputs did not match the real processor** (27 Oct). The tiny
  Qwen2.5-VL test omitted `mm_token_type_ids`, so the model took a simpler
  position path and the cached last-token code passed locally, then crashed
  on Kaggle (`A inner dim (2805760) does not match weight (2048)`). Test
  inputs for model code must carry every field the real processor emits.
