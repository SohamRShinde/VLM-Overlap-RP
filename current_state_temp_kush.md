# Current state: handover from Kush to Soham (31 Oct 2026)

Temporary handover note written by kushviradiya (with Claude Code) so that
@SohamRShinde can continue from here. The permanent record is still
`docs/RESULTS.md`, `docs/STATE.md` and `docs/DECISIONS.md`. This file collects
what happened in my sessions in one place. Delete it once you have read it.

**In one line:** the replication gate (D40) **passed on all six criteria**.
Our pipeline re-finds exactly Lietzow's Qwen2.5-VL-3B heads, and switching them
off has the effect the paper reports. The P side is ready, so the next step is
T-circuit discovery.

---

## 1. Where the project stands

| Part | Status | Where |
|---|---|---|
| T condition, behavioural screen (D41) | **Done (yours).** 286 usable objects of 436 eligible (65.6%), gate passed | RESULTS.md rows of 2026-10-20 |
| Replication gate part 1: group knockout, argmax flips | **Done** 25 Oct, re-run 27 Oct with identical numbers | `results/replication_flip_rate_qwen25vl3b.json` |
| Replication gate part 2: head re-discovery by patching | **Done** 27 Oct, recorded 31 Oct | `results/replication_patching_qwen25vl3b.json` |
| Replication gate part 3: pairwise knockout, group and single heads | **Done** 27 Oct, recorded 31 Oct | `results/replication_knockout_qwen25vl3b.json` |
| D40 verdict | **PASS** (all of a-f) | `scripts/analyze_replication_gate.py` |
| D40 / D67 / D68 approval | **Waiting for you** (section 6) | `docs/DECISIONS.md` |
| T-circuit discovery | **Not started** | section 8 |

Commits from this round: `ed22968` and `3c294c8` (notebooks, D40/D67/D68
proposals), `90286fa` (cached-step fix), `59315fb` (patching result),
`dff3e32` (knockout result, D40 pass).

---

## 2. Setup used for every run

- Qwen/Qwen2.5-VL-3B-Instruct, 4-bit nf4 with double quantisation, **fp16
  compute** on a Kaggle T4 (the T4 has no native bf16; see the gotcha in
  CLAUDE.md). 1333 image tokens per image, with the area pinned as in Lietzow.
- Visual-Counterfact colour split: 493 Hub rows, of which 469 survive
  Lietzow's loader filters (22 dropped for the name, 2 for colour overlap).
- Prompts are Lietzow's exact strings, both ending in "Respond with the color name only."
  - Visual: "What color is this {object} here?"
  - Prior: "What color is {a/an object} usually?"
- **Correctly conflicting subset:** objects where the Visual prompt's top-1
  answer is the counterfactual colour and the Prior prompt's top-1 answer is a
  real colour. **73 objects**, the same count Lietzow report.
- **Head knockout:** zero the `o_proj` input slice
  `[head*head_dim:(head+1)*head_dim]` at the last token. This is diffed against
  their `head_knockout` at commit 145b453: same slice, same position.
- **Patching:** last-token head patching in both directions. P2V puts a
  Prior-run head activation into the Visual run; V2P does the reverse. We use
  their restoration score, then PCA on the per-head mean (P2V, V2P) and a
  +/-2 sigma cut on PC1, which gives promoting and suppressing heads.
- **Speed-up:** `patching.LastTokenRunner` caches the prompt minus its last
  token and re-runs only the last token for each patch. That is 81 ms per patched step on a
  T4, so the full 576-head x 2-direction sweep takes about 1.9 h plus about
  47 min of prompt runs.

---

## 3. What was run

| Run | Notebook | Commit | What it does | Output files |
|---|---|---|---|---|
| Flip rate, sections 1-4 | `notebooks/replication_flip_rate.ipynb` | `90286fa` (first run `a922e93`) | Finds the 73 conflicting objects; group knockout of their 9 promoting heads; flip = any change of the top-1 token | `results/replication_flip_rate_qwen25vl3b.json`, `results/replication_flip_rate_baseline_top1.jsonl`, `results/replication_flip_rate_ablated_top1.jsonl` |
| Flip rate, section 5 (pairwise knockout) | same notebook | `90286fa` | Lietzow's pairwise flip (D67) for the promoting group, the suppressing group, and each of the 15 heads alone | `results/replication_knockout_qwen25vl3b.json` |
| Patching | `notebooks/replication_patching.ipynb` | `90286fa` | Patches all 576 heads in both directions on the conflicting objects, then classifies the heads | `results/replication_patching_qwen25vl3b.json` |
| Local tests | `.venv/bin/python -m pytest` | `dff3e32` | CPU unit tests, including a tiny random Qwen2.5-VL that checks the cached path against a full forward pass | 173 passed, 3 skipped (the tiny-model tests need torch) |

---

## 4. Results

### 4.1 Patching re-discovers their heads exactly

| | Ours | Lietzow Table 5 |
|---|---|---|
| Promoting | L26H0, L26H5, L26H6, L27H1, L27H4, L28H3, L31H3, L31H7, L34H14 | the same 9 |
| Suppressing | L26H3, L27H2, L28H4, L30H3, L31H8, L34H10 | the same 6 |

- Their 9 promoting heads found: **9/9**, containment 1.00.
- **Jaccard 1.000**, against 0.008 for random sets; permutation p = 0.0001,
  the floor at 10,000 permutations.
- Their 9 heads hold PC1 ranks **1-9** of 576 (median 5).
- Strongest by PC1: L31H3 (0.399), L26H5 (0.307), L26H0 (0.271).

### 4.2 Knockout reproduces their effect sizes

Flip definition D67 (pairwise, as in their code). n = 73.

| Knockout | Prior flip | Visual flip | Lietzow |
|---|---|---|---|
| All 9 promoting heads | **65/73 = 89.0%** | **2/73 = 2.7%** | 84.9% / 1.4% |
| All 6 suppressing heads | 19/73 = 26.0% | 0% | not compared |
| L26H5 alone | **34/73 = 46.6%** | 1.4% | 41.1% |
| L31H3 alone | **30/73 = 41.1%** | 0% | 37.0% |
| L26H0 alone | 29/73 = 39.7% | 0% | |
| L26H6 alone | 27.4% | 0% | |
| L28H3 / L31H7 / L34H14 alone | 8.2% / 5.5% / 4.1% | 0% | |
| L27H1 / L27H4 alone | 0% / 0% | 0% | |
| Suppressing heads alone | L31H8 16.4%, L30H3 13.7%, L27H2 4.1%, L26H3 2.7%, L34H10 1.4%, L28H4 0% | 0% | |

With the older argmax definition (any change of the top-1 token), the same
knockout gives 66/73 = 90.4% prior and 5/73 = 6.8% visual. The 27 Oct re-run
reproduced these exactly. Most of the gap in visual flips is answers moving
to a third colour, which the pairwise definition does not count.

### 4.3 D40 verdict

| Criterion (thresholds proposed 27 Oct, before parts 2 and 3 ran) | Result | |
|---|---|---|
| (a) group prior flip >= 66.7% | 89.0% | pass |
| (b) group visual flip <= 10% | 2.7% | pass |
| (c) >= 5 of their 9 promoting heads found | 9 | pass |
| (d) Jaccard beats chance, p < 0.001 | p = 0.0001 | pass |
| (e) median PC1 rank of their 9 <= 29 | 5 | pass |
| (f) L26H5 and L31H3 in the top 3 single heads | 1st and 2nd | pass |

**Verdict: PASS.** Reproduce it with:

```bash
.venv/bin/python scripts/analyze_replication_gate.py results/replication_knockout_qwen25vl3b.json results/replication_patching_qwen25vl3b.json
```

---

## 5. Caveats: read before building on this

1. **The two runs differ by one object each.** 72 of the 73 objects are shared.
   Patching includes computer keyboard (i 137) and leaves out i 356. The
   patching notebook picks the top-1 answer on the cached last-token path,
   while the flip-rate notebook uses a full forward pass. In fp16 the two paths
   differ by about 0.125 logits, which changes a near-tied top-1. On the 72
   shared objects every criterion still passes.
   **Fix for T:** choose the subset with the same code path that is then
   patched.
2. **The no-knockout baseline has one flip** (i 356, the same near-tie) and
   one exact tie (i 336). Dropping them gives 88.7-88.9% prior and 2.8%
   visual, and the top-3 single heads do not change.
3. **The 2 sigma cut sits right on a head.** The 10th head, L31H1, scores 0.0634
   against a cut of 0.0636. On the 72 shared objects it crosses the cut (10
   promoting heads, Jaccard 0.90). "Their 9 are our top 9" is the more
   robust statement than "exactly 9/9". This will matter for J_PT: report the
   all-head rank correlation (plan 3.5), not only set overlap.
4. **(f) is close at third place.** L31H3 (30 flips) is one object ahead of
   L26H0 (29).
5. **Two promoting heads do nothing alone.** L27H1 and L27H4 flip no answers
   when knocked out individually, so their effect is group-level only.
6. **The suppressing group flips 26% of prior answers when zeroed.** This is
   not a contradiction, because the class is defined by restoration direction,
   but it should be understood before suppressing heads enter an overlap
   claim.
7. **The P-side and T-side compute dtypes differ.** Your T screen ran in
   emulated bf16; all replication runs ran in fp16.
8. **(a) and (b) were not blind.** Part 1's argmax numbers existed before D40
   got thresholds, as D40 itself discloses. (c)-(f) were blind.

---

## 6. Decisions waiting for you

All three are PROPOSED in `docs/DECISIONS.md` and were written by me on 27 Oct.

- **D40:** the six numeric thresholds above. Their results are now known, so
  approve them as they were written, or record a disagreement. Do not retune
  them.
- **D67:** the flip is Lietzow's pairwise definition (from their
  `knockout_attn_heads.py`, commit 145b453), not "top-1 changed".
- **D68:** the P head set used for H1/H2 is **our re-discovered set**, so P and
  T come out of the identical pipeline. Here our set equals theirs, so this
  changes nothing in practice for Qwen-3B.

---

## 7. Bugs found and fixed along the way

These are useful context if something similar breaks on your side. CLAUDE.md
has the full list.

- **bitsandbytes ImportError on Kaggle.** Installing it into a kernel that has
  already imported transformers does nothing; restart the kernel.
- **A 16-hour run that was lost.** The cause was emulated bf16 on the T4 plus
  no checkpointing. The notebooks now time one step first and checkpoint one
  JSON line per object (D44).
- **Crash in the cached last-token step** (`A inner dim (2805760) does not match
  weight (2048)`). Qwen's multimodal RoPE needs the prefill `position_ids`
  carried over to the one-token step. This was fixed in `90286fa`. The CPU
  tests had missed it because they did not pass `mm_token_type_ids`; they now
  do.
- **A stale Kaggle checkout.** The first section-5 run cloned `3c294c8`, from
  before the fix, and crashed. Cell 1 prints the commit; check it on every run.

---

## 8. Suggested next steps

1. **Approve or amend D40, D67 and D68** (section 6).
2. **T-circuit discovery** (plan 3.4), with the same pipeline:
   - Source and target are the neutral-note and incongruent-note images of the
     same object, both with the Visual prompt.
   - "Correctly conflicting" for T means the model is right on the neutral note
     and fooled by the incongruent note. Start from the 286 usable objects in
     `results/t_usable_objects.json`, and re-select them on the cached path
     (caveat 1).
   - Reuse `patching.LastTokenRunner`, `patch_example` and `classify_heads`
     unchanged; only the pair of inputs differs. The 286 objects take about
     4x as long as the 73 P objects (roughly 7-8 h on a T4), so checkpoint, or
     split the run across sessions.
   - This needs a new notebook (for example `notebooks/t_patching.ipynb`) with
     its own Kaggle output prefix. Decide who owns it and add it to the table
     in CLAUDE.md.
3. **Before comparing P and T:** compute a split-half ceiling for P (73 objects
   split in two) so a low J_PT can be told apart from noise (plan 3.5, risk 1).
4. **Small leftovers:**
   - Re-run the 5 numerics-failure objects from the T screen in fp32.
   - Look at the 8 objects the promoting group does not flip: black swan,
     bluejay, broccoli, elk, honeydew, lime, sea, skunk.

---

## 9. Files

- Results: `results/replication_patching_qwen25vl3b.json`,
  `results/replication_knockout_qwen25vl3b.json`,
  `results/replication_flip_rate_qwen25vl3b.json`,
  `results/replication_flip_rate_baseline_top1.jsonl`,
  `results/replication_flip_rate_ablated_top1.jsonl`
- Analysis: `scripts/analyze_replication_gate.py` (D40),
  `scripts/analyze_replication.py` (argmax part 1)
- Code: `src/circuit_overlap/patching.py`, `src/circuit_overlap/replication.py`
- Notebooks (mine): `notebooks/replication_flip_rate.ipynb`,
  `notebooks/replication_patching.ipynb`
- Write-ups: `docs/RESULTS.md` (rows and notes dated 2026-10-27/31),
  `docs/STATE.md`, and the status block in `CLAUDE.md`
