# STATE.md — One Circuit or Two? (VLM circuit overlap)

**Last updated:** 2026-10-19 (chat 2: Lietzow repo read; prompt, checkpoints, loader filters and Δ convention pinned)
**Phase:** Pre-experiment. Code written and CPU-tested; **no GPU run, no results.** Plan of record: `docs/plan_v2.md` (v2.1, 16 Oct 2026), except its venue claim (see blocker 1).

## Status
- Audit of plan_v2 against uploaded PDFs: **done** (chat 1). Lietzow head lists (plan App. A) match Lietzow Tables 5–7 exactly for Qwen-VL 3B, PaliGemma 3B, Qwen-VL 7B. All App. B numbers match.
- Venue requirements: **BLOCKED.** Uploaded `AIR-FM.md` is the **AAAI 2026** edition (deadline 23 Oct 2025, 4-page limit). No AAAI-27 CFP in project files. Plan's "6 pages, non-archival" is unsupported.
- **Repo: done** (chat 2). git initialised, `.gitignore` excludes weights/activations/data/PDFs, Python 3.11 `.venv` with numpy + pytest + pillow. PR #1 merged; `main` at `1fd2b2d`.
- **Section-3 metrics: done and CPU-tested.** `src/circuit_overlap/metrics.py` -- logit difference, restoration score, flip rate, Jaccard, containment, permutation null for J.
- **T-screen code: written, CPU-tested, NOT RUN.** `notes.py` (overlay + exact mask), `tokens.py` (mask -> image tokens, 14px patch + 2x2 merge), `screen.py` (D27 rule, D41 gate, continuous fallback), `data.py` (manifest), `provenance.py` (result envelope), `t_screen.py` (GPU driver, never executed), `notebooks/t_screen.ipynb`. 104 CPU tests pass.
- **Visual-Counterfact schema: verified on the Hub** (2026-10-18, user, dataset viewer). `mgolov/Visual-Counterfact`, `split="color"`, 493 rows. Columns are `object`, `correct_answer` (**a stringified list** of acceptable colours), `incorrect_answer`, `original_image`, `counterfact_image` -- not the names plan_v2 section 3.1 assumed. Handled in code (D29).
- Replication gate, T screen on GPU: not started.

## Blockers
1. **No AAAI-27 venue chosen.** The uploaded `AIR-FM.md` is the **AAAI-2026** edition of that workshop (deadline 23 Oct 2025, 4 pages) and AIR-FM is not on the AAAI-27 list. plan_v2.1 section 0.8 proposed GSR-VLM, but that is **a candidate, not a decision** (D01, reopened 2026-10-18). Until a workshop is picked, its deadline, page limit, template, anonymity rules and archival status are all unknown, and plan section 9's 4-vs-6-page branch cannot be resolved. The AAAI general workshop deadline (20 Nov 2026) is the planning date in the meantime.
2. ~~Lietzow repo not yet inspected~~ **RESOLVED 2026-10-19.** Cloned `github.com/nlietzow/vision-default-prior-override` (MIT, Copyright 2026 Niclas Lietzow, commit `145b453`). Answers below. One sub-item remains: **their head lists are not in the repo** -- `data/` is git-ignored and `knockout_targets.py` reads `data/classify_attn_heads.json`, which is not published. Head lists must still be transcribed from the paper's Tables 5-7 into `configs/heads_*.json` by hand.

## What the Lietzow repo settled (2026-10-19)
| Question | Answer | Source in their repo |
|---|---|---|
| Visual prompt | `What color is this {object} here? Respond with the color name only.` -- "here" **is** present, and there is an **answer-format suffix** the paper's prose omits | `src/vdpo/runner/core.py::_get_question` |
| Prior prompt | `What color is {a/an object} usually? Respond with the color name only.` (article from `inflect`) | same |
| Checkpoints | `Qwen/Qwen2.5-VL-{3B,7B}-Instruct`, `google/paligemma2-{size}-mix-448`, `llava-hf/llava-v1.6-mistral-7b-hf` | `src/vdpo/vlm/*.py` |
| Precision | **4-bit by default** (`LOAD_IN_4BIT=true`); README offers full precision as an opt-in | `settings.py`, README "Hardware" |
| Answer tokens | First token of **both** case variants ("Blue", "blue"), **no leading space**; a colour's logit is the max over variants | `core.py::get_answer_token_ids` |
| Scoring | `max over correct_answer tokens` vs `max over incorrect_answer tokens` | `runner/inference.py` |
| Δ for patching | **Not** the colour pair: the logit difference between the **argmax token of the source run and of the target run**, and examples whose two argmax tokens coincide are skipped | `runner/patching_last_token_attn_heads.py` |
| Conflicting subset | Examples whose top-1 prediction differs between the two prompts | README, `--stage inference` |
| Head indexing | `{"layer": int, "head": int}`, 0-based, ablation zeroes `o_proj` input slice `[head*head_dim : (head+1)*head_dim]` at the last token via a pre-hook | `interventions.py::head_knockout` |
| 493 vs 467 | Their loader silently drops rows that fail pydantic validation: **parenthetical names** ("apple (fruit)"), **plural names**, and **counterfactual colour also in `correct_answer`** (this last is exactly radish and spider, per their README). So the paper's counts are post-filter | `types/models.py`, `utils/load_examples.py`, README "Dataset" |
| Dataset access | Needs an `HF_TOKEN` | README "Dataset" |
| Licence | MIT -- their code may be reused with attribution | `LICENSE` |

## Open questions (from audit)
- [x] ~~Exact Visual prompt~~ **resolved 2026-10-19** (see table above; note the answer-format suffix, absent from both the paper's §3.1 and its Table 1 caption).
- [x] ~~Visual-Counterfact column names~~ **resolved 2026-10-18:** `object`, `correct_answer` (stringified list), `incorrect_answer`, `original_image`, `counterfact_image`.
- [ ] **Visual-Counterfact size: mechanism found, numbers not yet confirmed.** It was explanation (c): undocumented filtering. Their loader drops invalid rows silently, so 493 Hub rows become 469 valid objects and 467 after the colour-overlap rule. `notebooks/t_screen.ipynb` cell 2 now applies the same three rules and prints the drop counts -- **confirm the cascade actually lands on 469/467** on the current Hub revision before the replication gate.
- [ ] Dataset licence: not shown on the viewer page. Check the dataset card before publishing.
- [x] ~~Exact checkpoints~~ **resolved 2026-10-19** (see table above).
- [x] ~~Layer index base~~ **resolved 2026-10-19:** 0-based layer and head indices, matching D30/D60.
- [ ] V contrast: their `ContrastSpec`s are in `src/vdpo/types/contrast.py` (`PRIOR_CIRCUIT_CONTRAST`, visual variant) -- read it to settle which prompt is held fixed. Not yet read.
- [ ] Lietzow "61% of V-classified heads also classified under P": pooled over models? promoting+suppressing? It is a containment fraction, not Jaccard.
- [ ] Is Qwen2.5-VL-3B susceptible to typographic notes at all? No uploaded paper tests Qwen2.5-VL or PaliGemma on typographic attacks (SCAM evaluates neither).
- [ ] Last-token risk for image-varying contrasts (V, T): Nooralahzadeh et al. report last-token residual patching leaves 90–98% of outputs unchanged on image-variant contrasts. Does last-token head patching give usable restoration for T?
- [ ] Object-set confound: P "correctly conflicting" objects (model succeeds at override) and T fooled objects may be different subsets. Analyze on shared objects too?
- [ ] Novelty vs Hua et al. 2025 (image-vs-caption router/promotion heads in Qwen2.5-VL) and Nikankin et al. 2025 (random-normalized IoU circuit overlap). Not-uploaded related work (Dyslexify, Ortu, Golovanevsky 2025a/b) still unread.

## Next steps (proposed, not yet agreed)
1. Email organizers; re-scope to confirmed page limit.
2. Clone Lietzow repo; resolve checkpoint / prompt / indexing questions; write `configs/heads_lietzow_*.json` from Tables 5–6.
3. **T behavioral screen first** (inference only, cheap): does a note flip Qwen2.5-VL-3B on ≥ ~50 objects?
4. CPU smoke tests, then replication gate (P on Qwen-3B).
5. Apply plan edits from chat-1 audit (§2 table, novelty wording, 61% definition, last-token risk, object-matched analysis).
