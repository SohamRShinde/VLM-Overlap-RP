# STATE.md — One Circuit or Two? (VLM circuit overlap)

**Last updated:** 2026-10-18 (chat 2: repo, scaffolding, T-screen code, dataset schema verified)
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
2. Lietzow repo not yet inspected: exact checkpoints, prompt string, V-contrast prompt, index base.

## Open questions (from audit)
- [ ] Exact Visual prompt: Lietzow §3.1 has "What color is this {object} here?"; their Table 1 caption has "What color is this strawberry?" (no "here"). Check repo.
- [x] ~~Visual-Counterfact column names~~ **resolved 2026-10-18:** `object`, `correct_answer` (stringified list), `incorrect_answer`, `original_image`, `counterfact_image`.
- [ ] **Visual-Counterfact size, still open.** The Hub's `color` split has **493 rows**, matching Nooralahzadeh; Lietzow say 469 objects, 467 after dropping radish and spider. Three candidate explanations, in order of likelihood, all cheap to check (see `notebooks/t_screen.ipynb` cell 2b): (a) 493 *rows* but fewer unique `object` values -- count `len(set(ds["object"]))`; (b) Lietzow used an earlier Hub revision -- list the dataset's commit history and load the revision dated before their submission; (c) Lietzow applied a filter they do not describe -- check their repo's loading code. Resolve before the replication gate: a head-set mismatch would otherwise be unattributable between dataset and pipeline.
- [ ] Dataset licence: not shown on the viewer page. Check the dataset card before publishing.
- [ ] Exact checkpoints: Lietzow writes "Qwen-VL-2.5 (3B, 7B)" citing the Qwen2-VL paper, and "PaliGemma (3B, 10B) with a Gemma 2 backbone" citing PaliGemma 2. Instruct/mix/pt and resolution not stated.
- [ ] Layer index base: paper uses L{l}H{h}; H0 appears (heads 0-based); layer base not stated.
- [ ] V contrast (Lietzow App. F): which prompt is held fixed? Not stated.
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
