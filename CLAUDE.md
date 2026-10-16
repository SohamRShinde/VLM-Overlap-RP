# Circuit overlap project

Research code for a mechanistic interpretability paper.
Question: do heads that implement prompted prior override (Lietzow et al.
2606.28273) overlap with heads that let in-image text override vision?

**Plan of record: @docs/plan_v2.md** (v2.1, 16 Oct 2026). Read it before
proposing experiments; it supersedes any earlier plan. Venue is AAAI-27
**GSR-VLM** (not AIR-FM; see plan section 0 item 8).

## Living documents (keep current, never overwrite wholesale)
- `docs/STATE.md`     status, blockers, open questions, next steps
- `docs/DECISIONS.md` dated decisions and pre-registered thresholds (D01-D51)
- `docs/RESULTS.md`   every reported number, with model, n, seed, prompt, commit

## Setup
- Models: Qwen2.5-VL-3B (primary), PaliGemma-3B. fp16, T4-compatible.
- No local GPU. Real experiments run on Kaggle; here we only write code and test on CPU.
- Local env: Python 3.11 in `.venv/` (git-ignored). Currently installs numpy and
  pytest only. torch/transformers/nnsight are deliberately NOT installed locally;
  they arrive with the hook and smoke-test work, and run on Kaggle.
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
```

## Layout
- src/circuit_overlap/  harness, hooks, metrics, head sets
- configs/              experiment configs and head lists (heads_*.json)
- tests/                CPU unit and smoke tests (tiny random-init model)
- results/              JSON outputs from Kaggle (include model, n, seed, prompt, commit)
- notebooks/            thin Kaggle notebooks that call src/
- docs/                 plan and living documents

## Rules
- Zero-ablate head outputs at the last token position (matches Lietzow).
- Never hard-code head indices from memory; load from configs/heads_*.json.
  Heads are 0-based `(layer, head)` tuples (see DECISIONS.md D30).
- Run `.venv/bin/python -m pytest` before suggesting a Kaggle run.
- Never invent results; analysis scripts read from results/ only. Reference
  numbers from other papers are not results (see docs/RESULTS.md header).
- Large artifacts (weights, activations, data, PDFs) stay out of git; small
  JSON results and configs stay in.
