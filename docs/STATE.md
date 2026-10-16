# STATE.md — One Circuit or Two? (VLM circuit overlap)

**Last updated:** 2026-10-16 (chat 1: audit + document setup)
**Phase:** Pre-experiment. No code run, no results. Plan of record: `VLM_Circuit_Overlap_Research_Plan_v2.md` (15 Oct 2026), pending the edits listed below.

## Status
- Audit of plan_v2 against uploaded PDFs: **done** (chat 1). Lietzow head lists (plan App. A) match Lietzow Tables 5–7 exactly for Qwen-VL 3B, PaliGemma 3B, Qwen-VL 7B. All App. B numbers match.
- Venue requirements: **BLOCKED.** Uploaded `AIR-FM.md` is the **AAAI 2026** edition (deadline 23 Oct 2025, 4-page limit). No AAAI-27 CFP in project files. Plan's "6 pages, non-archival" is unsupported.
- Repo, smoke tests, replication gate, T screen: not started.

## Blockers
1. AAAI-27 AIR-FM CFP: deadline, page limit (4 vs 6), archival status, anonymity, template. Email air-fm@googlegroups.com (contact listed in the 2026 CFP).
2. Lietzow repo not yet inspected: exact checkpoints, prompt string, V-contrast prompt, index base.

## Open questions (from audit)
- [ ] Exact Visual prompt: Lietzow §3.1 has "What color is this {object} here?"; their Table 1 caption has "What color is this strawberry?" (no "here"). Check repo.
- [ ] Visual-Counterfact size: Lietzow says 469 (467 after exclusions); Nooralahzadeh et al. say Color split = 493. Which release/version?
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
