# STATE.md — One Circuit or Two? (VLM circuit overlap)

**Last updated:** 2026-10-19 (chat 2: Lietzow repo read; head lists transcribed; 4-bit decided)
**Phase:** T condition **established and analysed**. D41 gate passed (286 usable objects of 469, 65.6% of eligible), per-record data saved and committed to `results/`, D38 settled by the congruent baseline, and the plan 3.7 position-vs-size question answered. No circuit work yet. Plan of record: `docs/plan_v2.md` (v2.1, 16 Oct 2026), except its venue claim (see blocker 1).

## Status
- Audit of plan_v2 against uploaded PDFs: **done** (chat 1). Lietzow head lists (plan App. A) match Lietzow Tables 5–7 exactly for Qwen-VL 3B, PaliGemma 3B, Qwen-VL 7B. All App. B numbers match.
- Venue requirements: **BLOCKED.** Uploaded `AIR-FM.md` is the **AAAI 2026** edition (deadline 23 Oct 2025, 4-page limit). No AAAI-27 CFP in project files. Plan's "6 pages, non-archival" is unsupported.
- **Repo: done** (chat 2). git initialised, `.gitignore` excludes weights/activations/data/PDFs, Python 3.11 `.venv` with numpy + pytest + pillow. PR #1 merged; `main` at `1fd2b2d`.
- **Section-3 metrics: done and CPU-tested.** `src/circuit_overlap/metrics.py` -- logit difference, restoration score, flip rate, Jaccard, containment, permutation null for J.
- **T-screen code: written, CPU-tested, NOT RUN.** `notes.py` (overlay + exact mask), `tokens.py` (mask -> image tokens, 14px patch + 2x2 merge), `screen.py` (D27 rule, D41 gate, continuous fallback), `data.py` (manifest), `provenance.py` (result envelope), `t_screen.py` (GPU driver, never executed), `notebooks/t_screen.ipynb`. 104 CPU tests pass.
- **Visual-Counterfact schema: verified on the Hub** (2026-10-18, user, dataset viewer). `mgolov/Visual-Counterfact`, `split="color"`, 493 rows. Columns are `object`, `correct_answer` (**a stringified list** of acceptable colours), `incorrect_answer`, `original_image`, `counterfact_image` -- not the names plan_v2 section 3.1 assumed. Handled in code (D29).
- **T screen: DONE** (2026-10-20). Gate passed, 286 usable objects of 469. Per-record data in `results/`, recovered from the checkpoint after the save cell failed. Analysis: `scripts/analyze_t_screen.py`. All numbers in RESULTS.md.
- **D38 settled empirically:** congruent notes shift Delta by -2.10 (away from the counterfactual), the attack by +8.50, with **zero overlap** between the distributions. The attack's effect is note content, not note presence.
- **plan 3.7 answered:** position beats size. Paired within object, the note fools on-object-only 99 times vs beside-only 20; the small-vs-medium comparison is near-symmetric despite double the area. Not an occlusion effect.
- **plan risk 7 confirmed:** 5 objects returned non-finite logits in 4-bit/half precision (one generated `'!!!!'`). Excluded and reported; re-run those in fp32.
- Plan **risk 2 is retired**: Qwen2.5-VL-3B does not resist typographic notes.
- **Replication gate: part 1 of 3 done** (2026-10-25). Group ablation of Lietzow's 9 promoting heads: prior flip 66/73 = 90.4% (theirs 84.9%), visual flip 5/73 = 6.8% (theirs 1.4%, not distinguishable at n = 73). See RESULTS.md. Still to do: re-discover the heads by patching and compare with their Table 5, the single-head checks (L26H5, L31H3), and numeric pass thresholds for D40.
- **Replication gate: part 2 done** (run 2026-10-27, recorded 2026-10-31). Re-discovery by last-token patching over all 576 heads on the 73 correctly-conflicting objects finds **exactly** Lietzow's 9 promoting and 6 suppressing heads (Jaccard 1.000, their 9 are our PC1 ranks 1-9). D40 (c), (d), (e) pass. Both boundaries sit within 0.001 of the 2 sigma cut (L31H1 just out, L27H2 just in). Still to do: the pairwise knockout re-run on `90286fa` or later for (a), (b), (f); Soham's approval of D40/D67/D68.
- **Replication gate: PASSED** (recorded 2026-10-31, against the D40 thresholds proposed on 27 Oct). Pairwise knockout at `90286fa`: group promoting prior flip 65/73 = 89.0%, visual flip 2/73 = 2.7%; single heads L26H5 46.6% and L31H3 41.1% rank 1st and 2nd (L26H0 a close 3rd). All six criteria pass. Open: @SohamRShinde's approval of D40/D67/D68. The patching and knockout runs disagree on one object each (cached vs full-forward top-1 in fp16, see RESULTS.md); no verdict depends on it. **Next: T-circuit discovery** with the same patching pipeline.

## Blockers
1. **Venue: page limit and deadline fixed, workshop name still open.** As of 2026-10-31 the paper is **6 pages, due 20 Nov 2026** (D01a, decided by @SohamRShinde), so plan section 9's 4-page fallback is moot (D07). Still unconfirmed: which AAAI-27 workshop, its template, anonymity rules and archival status. AIR-FM is not an option (it was the AAAI-2026 edition); GSR-VLM remains plan v2.1's candidate. With 20 days left, the open question is what the two extra pages buy -- the V condition, H3, a second model -- not whether the core fits.
2. ~~Lietzow repo not yet inspected~~ **RESOLVED 2026-10-19.** Cloned `github.com/nlietzow/vision-default-prior-override` (MIT, Copyright 2026 Niclas Lietzow, commit `145b453`). Answers below. Their head lists are **not** in the repo (`data/` is git-ignored, and `knockout_targets.py` reads an unpublished `data/classify_attn_heads.json`), so Tables 5-7 were transcribed by hand into `configs/heads_lietzow_*.json` on 2026-10-19 (five models) and are validated by `heads.load_head_set` (D66).

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
| Image grid | `min_pixels = max_pixels = 1280*28*28` (~1280 image tokens), area pinned, shape follows the aspect ratio | `vlm/qwen.py` |
| 4-bit config | nf4, double quantisation, compute dtype bf16 where supported else fp16 | `vlm/qwen.py` |
| Dataset access | Needs an `HF_TOKEN` | README "Dataset" |
| Licence | MIT -- their code may be reused with attribution | `LICENSE` |

## Open questions (from audit)
- [x] ~~Exact Visual prompt~~ **resolved 2026-10-19** (see table above; note the answer-format suffix, absent from both the paper's §3.1 and its Table 1 caption).
- [x] ~~Visual-Counterfact column names~~ **resolved 2026-10-18:** `object`, `correct_answer` (stringified list), `incorrect_answer`, `original_image`, `counterfact_image`.
- [ ] **Visual-Counterfact size: 2 objects unaccounted for** (was 4, then 6 recovered). It was explanation (c): undocumented filtering. First Kaggle run (2026-10-19) kept 463. Hypothesis (a) **confirmed**: their `_sanitize_object_name` capitalizes before the plural test, and a word counts as plural only when `singular_noun(x)` differs from `x`, so deer, sheep, moose, goldfish, mackerel and "anemone fish" are kept -- 6 objects, giving **469**, exactly their reported valid count. inflect version was not the issue (Kaggle already had 7.5.0). Implemented in `lietzow_object_name`. Our cascade now gives **469 kept** (493 − 14 plural − 8 parenthetical − 2 colour overlap). The `list[str]` hypothesis is **dead**: every row's `correct_answer` is a list (0 bare strings, 0 unparseable). Remaining gap: **2 objects**. Both remaining hypotheses are now **dead**: every row's `correct_answer` is a list, and no row has a path-only image. The 2 colour-overlap drops are exactly radish and spider, matching their README. Last and most likely explanation, checked by cell 2c at no download cost: **the dataset grew.** The Hub shows the initial upload on 2025-05-21 and three more on 2025-07-15/16; if a pre-July revision has `color = 469`, their 469/467 describes that release and ours is the current one. **Then D20 needs a decision:** pin that revision for the replication gate, or use current and disclose. If no revision matches, stop and record the deviation.
- [ ] Dataset licence: not shown on the viewer page. Check the dataset card before publishing.
- [x] ~~Exact checkpoints~~ **resolved 2026-10-19** (see table above).
- [x] ~~Layer index base~~ **resolved 2026-10-19:** 0-based layer and head indices, matching D30/D60.
- [ ] V contrast: their `ContrastSpec`s are in `src/vdpo/types/contrast.py` (`PRIOR_CIRCUIT_CONTRAST`, visual variant) -- read it to settle which prompt is held fixed. Not yet read.
- [ ] Lietzow "61% of V-classified heads also classified under P": pooled over models? promoting+suppressing? It is a containment fraction, not Jaccard.
- [ ] Is Qwen2.5-VL-3B susceptible to typographic notes at all? No uploaded paper tests Qwen2.5-VL or PaliGemma on typographic attacks (SCAM evaluates neither).
- [ ] Last-token risk for image-varying contrasts (V, T): Nooralahzadeh et al. report last-token residual patching leaves 90–98% of outputs unchanged on image-variant contrasts. Does last-token head patching give usable restoration for T?
- [ ] Object-set confound: P "correctly conflicting" objects (model succeeds at override) and T fooled objects may be different subsets. Analyze on shared objects too?
- [ ] Novelty vs Hua et al. 2025 (image-vs-caption router/promotion heads in Qwen2.5-VL) and Nikankin et al. 2025 (random-normalized IoU circuit overlap). Not-uploaded related work (Dyslexify, Ortu, Golovanevsky 2025a/b) still unread.

## P split-half ceiling (2026-10-31)
- **Done**, from the existing P patching results, CPU only (`scripts/analyze_split_half.py`, D69 pre-registered first). Promoting-set Jaccard between halves: median **0.900** [0.900, 1.000]; chance p95 **0.059**. All 9 P heads are selected in 100% of 2,000 half-runs; only L31H1 wobbles (48%). The P circuit is highly reproducible, so J_PT has a wide interpretable range.
- **D70 (how J_PT is read) approved 2026-10-31**, before any T-circuit result existed. Jaccard primary; reason for the same-day revision recorded in the decision.

## T-circuit discovery (2026-10-31)
- **Notebook built, not run.** `notebooks/t_patching.ipynb` (owner @SohamRShinde), design pre-registered as **D71**: incongruent vs neutral note on `on_object_small`, Visual prompt, selection from all 469 objects on the cached patching path, P-shared objects swept first then seeded random, Kush's patching pipeline unchanged. Logic in `src/circuit_overlap/t_patching.py`.
- **Not verified on a GPU.** Every code cell is checked statically (`tests/test_t_patching_notebook.py`: parses, no name used before a cell defines it, outputs cleared, D71 roles, both loops checkpointed), and the CPU pieces are unit-tested, but the cells have not executed against the real model. Section 2's `[VERIFY]` cells are the first real test: look at the sample pair, and the cached-path checks must pass before the sweep.
- **The analysis is ready and pre-registered** (D72, committed before the T results): `scripts/analyze_t_circuit.py`. Validated on known-answer controls from the P data (T = P -> "self-agreement"; scrambled -> "chance"; 3/5/7 of 9 heads moved -> "between" with an unbiased bootstrap). The same controls showed the all-head rank correlation is insensitive to circuit identity (0.974 with 7 of 9 heads moved), so it is reported as landscape similarity only.
- **Runtime ~8-9 h** (81 ms per patched step from Kush's run x 1,152 steps per object x ~280 objects, plus ~75 min of prompt runs). Under Kaggle's 12 h limit, but close enough that the resume path matters.

## Next steps (proposed, not yet agreed)
1. Email organizers; re-scope to confirmed page limit.
2. Clone Lietzow repo; resolve checkpoint / prompt / indexing questions; write `configs/heads_lietzow_*.json` from Tables 5–6.
3. **T behavioral screen first** (inference only, cheap): does a note flip Qwen2.5-VL-3B on ≥ ~50 objects?
4. CPU smoke tests, then replication gate (P on Qwen-3B).
5. Apply plan edits from chat-1 audit (§2 table, novelty wording, 61% definition, last-token risk, object-matched analysis).
