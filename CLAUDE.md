# Circuit overlap project

Research code for a mechanistic interpretability paper (AAAI-27 AIR-FM workshop).
Question: do heads that implement prompted prior override (Lietzow et al.
2606.28273) overlap with heads that let in-image text override vision?

## Setup
- Models: Qwen2.5-VL-3B (primary), PaliGemma-3B. fp16, T4-compatible.
- No local GPU. Real experiments run on Kaggle; here we only write code and test on CPU.
- Python 3.11, PyTorch, transformers, nnsight.

## Layout
- src/        harness, hooks, metrics, head sets
- configs/    experiment configs
- tests/      CPU unit and smoke tests (tiny random-init model)
- results/    JSON outputs from Kaggle (include model, n, seed, prompt, commit)
- notebooks/  thin Kaggle notebooks that call src/

## Rules
- Zero-ablate head outputs at the last token position (matches Lietzow).
- Never hard-code head indices from memory; load from configs/heads_*.json.
- Run `pytest tests/` before suggesting a Kaggle run.
- Never invent results; analysis scripts read from results/ only.
