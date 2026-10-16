# Does In-Image Text Hijack the Same Heads as Prompted Priors? A Circuit-Overlap Test in Vision-Language Models

**Version:** v2.1 (16 Oct 2026). Supersedes v1 and v2; see §0 for what changed.
**Target:** AAAI-27 workshop **GSR-VLM** (Geometric and Spatial Reasoning in Vision-Language Models). Relevant listed topic: "Interpretability, robustness, and generalization." Workshops run 22-23 Feb 2027, Montréal.
**Deadline:** AAAI's general workshop deadline is **20 Nov 2026** (AoE), notifications by 2 Dec 2026. **[VERIFY]** GSR-VLM's own deadline, page limit, template, anonymity rules, and archival status on its call for papers.
**Presentation:** AAAI does not permit virtual presentations. **[VERIFY]** that one author can present in person before submitting.
**Team:** undergraduates (**[VERIFY]** team size: the Project description says 2, earlier drafts said 4), free Kaggle/Colab GPUs, no backbone training.

Items marked **[VERIFY]** must be checked against the source before they appear in the paper.

---

## 0. Corrections to the original plan

1. **Direction of Lietzow et al. was misdescribed.** The original plan treated their heads as making the model prefer memory over correct vision (a failure). In the paper, the conflict condition is a *Prior prompt on a counterfactual image* (blue strawberry, "what color is a strawberry usually?"). The prompt-requested answer is the memorized one, and the model usually fails by following the image. The "prior-override" heads perform the *correct, requested* behavior. Visual grounding is the default; prior grounding needs a sparse set of heads (2.5-4.8% of heads, second half of the network).
2. **The mechanism they found is image-vs-text routing.** In Qwen-VL and LLaVA-NeXT, classified heads shift attention away from image tokens toward text tokens under prior grounding (mean image-attention delta +0.19 to +0.24). PaliGemma routes differently (attended representations change, attention does not). Typographic attacks put the competing signal *inside the image*, so there is no obvious reason these heads would handle them. The prior should lean toward **low overlap**.
3. **Model mismatch.** Lietzow et al. used Qwen2.5-VL (3B, 7B), LLaVA-NeXT 7B (Mistral), PaliGemma (3B, 10B). **LLaVA-1.5 is not in their set**, so their head indices do not transfer to it.
4. **Confounded comparison.** The original design compared Lietzow's color task with SCAM's object-ID task, which differ in question type, answer vocabulary, dataset, and head-selection method. v2 uses one task and one selection method for all conditions (§3).
5. **Novelty claim softened.** Prior work on circuits in VLMs exists (§2). The defensible claim is narrower: in-image text vs. prompt-elicited priors, in an answer-matched design.
6. **H1c (text-manifold alignment) dropped.** Post-it patches aligning with a text subspace is close to true by definition, and the plan never said how token-level alignment would predict head-level routing.
7. **H1d′ (pre-generation probe) demoted to optional.** Late-layer last-token states already contain the answer, so a probe there reads the output rather than predicting it (§8).

**Added in v2.1:**

8. **Venue changed from AIR-FM to GSR-VLM.** The uploaded AIR-FM call was for AAAI-26 (4 pages, excluding references and appendix) and AIR-FM is not on the AAAI-27 list. GSR-VLM is the only VLM-specific AAAI-27 workshop and lists interpretability and robustness as a topic. Framing: an interpretability study of how VLMs arbitrate between visual evidence and competing signals, with a small spatial analysis (note position/size, §3.7). Do not present it as a spatial-reasoning paper.
9. **Audit corrections (chat-1 audit):** Lietzow's 61% is a *containment* fraction, not a Jaccard, so it cannot be compared directly to our $J$ (§3.5). P and T select different object subsets, so overlap is also reported on shared objects (§3.5). The T-condition behavioral screen runs first, as the cheapest kill-test (§4). Dataset size and prompt wording have small discrepancies across sources (§3.1, §3.2).

---

## 1. Question and Hypotheses

**Question.** When a VLM's answer is determined by something other than the object's visible color, is the same set of attention heads responsible regardless of where the competing signal comes from?

Three conflict sources, all on the same color-identification task, same objects, same color-word vocabulary:

| Code | Conflict source | Contrast (single factor varied) | Reference |
|---|---|---|---|
| **P** | Prompted prior: counterfactual image + "what color is a {obj} usually?" vs. visual prompt | Prompt varies, image fixed | Lietzow et al. main analysis |
| **V** | Visual evidence: original vs. counterfactual image, prompt fixed | Image varies, prompt fixed | Lietzow et al. Appendix F |
| **T** | Typographic: original-color image with a note naming a wrong color | Note *word* varies, pixels otherwise identical | **New** |

**Hypotheses (pre-register thresholds in DECISIONS.md before seeing T-results):**

- **H0 (replication gate).** Running Lietzow's classification on Qwen2.5-VL-3B reproduces their P circuit (substantial overlap with their reported head list, and group ablation flips a majority of prior-grounded predictions; they report 84.9% for Qwen-VL 3B). Failure means a pipeline bug, not a finding.
- **H1 (headline).** The T circuit's overlap with the P circuit is quantified by Jaccard $J_{PT}$ against a chance floor and a split-half ceiling. Two competing predictions:
  - *H1-specific:* P heads implement image-to-text routing, so $J_{PT}$ is near chance, and $J_{VT} > J_{PT}$ because V and T both vary image content.
  - *H1-general:* P heads implement a generic "answer from a non-object-color source" mechanism, so $J_{PT}$ is well above chance, approaching the P split-half ceiling.
- **H2 (causal transfer).** Ablating $\mathcal{H}_P$ changes behavior on T (and vice versa) by more than a size- and layer-matched random head set.
- **H3 (mechanism, descriptive).** Under T, do the T-circuit heads attend to the note's tokens more than to object tokens (using the known overlay mask), and do their outputs write the note's color word? Reported whatever $J$ turns out to be. Includes a small spatial analysis: does attack success, and attention to note tokens, depend on note position and size relative to the object (§3.7)?
- **H4 (external validity, secondary).** The same head sets evaluated on SCAM/NoSCAM (object identification) show consistent directions.

A low-overlap result and a high-overlap result are both publishable. State this in the paper, and avoid claiming a null from small samples (§6).

---

## 2. Related Work Status

Verified = full text read during planning. Everything else: re-verify before citing numbers.

| Source | Status | Role |
|---|---|---|
| Lietzow et al. 2026, arXiv 2606.28273 (Vision-Default, Prior-Override) | **Read in full.** Repo: github.com/nlietzow/vision-default-prior-override.git **[VERIFY repo contents and license]** | Defines P and V contrasts, classification method, head lists |
| Golovanevsky et al. 2025a, arXiv 2505.17127 (Visual-Counterfact) | Cited in Lietzow; not read | Dataset (469 objects; 467 usable) **[VERIFY access]** |
| Hua et al. 2025, arXiv 2507.01790; Ortu et al. 2025, arXiv 2507.13868; Nikankin et al. 2025, arXiv 2506.09047; Golovanevsky et al. 2025b, arXiv 2406.16320 | Cited in Lietzow's reference list; not read | Prior circuit/routing work in VLMs; read before claiming novelty |
| Nooralahzadeh et al. 2026 (Arbitration Failure, Not Perceptual Blindness), arXiv 2604.09364 per the original plan | Lietzow describes it as studying models that answer from prior knowledge despite a visual prompt, *not* typographic conflict. Audit: uses a Visual-Counterfact Color split of 493 items. **ID unverified** | Related work only |
| SCAM / NoSCAM | Uploaded. Audit: LVLM evaluation uses a 2-option multiple-choice prompt | Secondary evaluation set **[VERIFY numbers and license]** |
| Dyslexify (2508.20570), 2607.02494 | Unverified | Encoder-side typographic circuits for related work |
| Saini et al. 2605.08245, Liu et al. (IPI probing) 2608.02657, Sahoo et al. 2606.02907 | Unverified | Only needed if H1d′ is revived |

**Re-search for scoops in week 1:** "typographic attack attention heads decoder," "circuit overlap vision-language," "Jaccard overlap override heads," plus OpenReview workshop tracks and the Transformer Circuits Thread.

**Note on Lietzow's own appendix.** Under the V contrast, 61% of classified heads were also classified under the P contrast. This is a **containment** fraction $|A\cap B|/|A|$, not a Jaccard. Engage with it in the paper, but compare like with like: report containment alongside $J$ when citing it (§3.5).

---

## 3. Experimental Design

### 3.1 Data
- **Visual-Counterfact** (467 usable objects after excluding radish and spider, as in Lietzow). Nooralahzadeh report a Color split of 493, so **[VERIFY]** which dataset version Lietzow used and pin it. Each object has an original color $c_o$ and counterfactual color $c_{cf}$, with original and recolored images.
- **Note overlay generator (new code).** For each *original-color* image, paste a post-it-style note containing a color word, with controlled position, size, font, and background. Record the exact pixel mask (known by construction) and map it to patch tokens for the model's grid.
- **Note variants per object:**
  - *incongruent:* word = $c_{cf}$ (the attack)
  - *neutral:* a non-color word of matched length (primary baseline)
  - *congruent:* word = $c_o$ (secondary baseline)
  - *no note* (secondary baseline)
- Use 2-3 layouts per object; **bootstrap over objects**, not over images, since layouts of one object are correlated.
- Restrict to color words that tokenize to a single first token in each model **[VERIFY per tokenizer]**.
- Keep SCAM/NoSCAM for H4.

### 3.2 Prompts
- Reuse Lietzow's exact templates: Visual "What color is this {object} here?" and Prior "What color is {a(n) object} usually?". Use the Visual prompt for T. **[VERIFY]** the exact Visual string against their repo: the audit found the word "here" differs between their §3.1 and Table 1 caption.
- **Token set matched, roles not.** The logit-difference token pair is $(c_o, c_{cf})$ in all conditions, so the vocabulary is shared. But the roles differ: in P the memory answer is $c_o$ and the image says $c_{cf}$; in T the image says $c_o$ and the note says $c_{cf}$. State this limitation in the paper. It means "writing-head" overlap may be partly shaped by token role, so analyze routing and writing heads separately.

### 3.3 Models
- **Primary: Qwen2.5-VL-3B** (fits a 16 GB T4 in fp16). Lietzow report 73 of 467 "correctly conflicting" examples for Qwen-VL 3B, so sample size is tight (§6).
- **Second: PaliGemma-3B**, the Gemma-2-backbone variant **[VERIFY exact checkpoint in their repo]**.
- **Stretch:** Qwen2.5-VL-7B (needs care on a T4; check memory).
- Use fp16 (not 4-bit) for all mechanistic work. Check for NaN/overflow on Qwen in fp16 against a reference run on a few examples.
- Check early that typographic notes actually shift the answer. Qwen may be robust. See risks.

### 3.4 Circuit discovery (identical across P, V, T)
Follow Lietzow exactly so differences cannot be blamed on method:
- Activation patching at the **last token position**, at head output $z_h^{(\ell)}$ (input to $W_O$), both directions.
- Restoration score $R_d(c) = \dfrac{\Delta^{\text{patched}}_d(c) - \Delta^{\text{target}}_d}{\Delta^{\text{source}}_d - \Delta^{\text{target}}_d}$, with $\Delta = \text{logit}(t_{c_o}) - \text{logit}(t_{c_{cf}})$.
- Per head, take the two-direction restoration vector; PCA across heads; classify heads beyond $\pm 2\sigma$ on PC1 as promoting or suppressing.
- Restrict to correctly conflicting examples (T: model correct on the neutral-note image and fooled by the incongruent note).
- Group ablation: zero-ablate head outputs at the last token; report flip rates.
- Use their repo / NNsight where possible so indexing conventions match **[VERIFY layer and head indexing: 0- or 1-based]**.

**Speed-up (validate against full runs on ~10 examples first).** Because patching is at the last token and attention is causal, earlier positions' states are unchanged. Cache the prompt's K/V once, patch the last token at layer $\ell$, and recompute only the last token through layers $\ge \ell$. Alternatively, rank heads with attribution patching and confirm the top ~30-50 with real patching.

### 3.5 Overlap statistics

Let $\mathcal{H}_P, \mathcal{H}_V, \mathcal{H}_T$ be the promoting head sets.

$$J_{AB} = \frac{|\mathcal{H}_A \cap \mathcal{H}_B|}{|\mathcal{H}_A \cup \mathcal{H}_B|}$$

Report all of:
- **Chance floor:** expected $J$ for random sets of the same sizes (Qwen2.5-VL-3B: 36 layers × 16 heads = 576 **[VERIFY config]**; two random 9-head sets give $J \approx 0.008$). Compute by permutation, with a p-value.
- **Split-half ceiling:** run discovery on two random halves of the same condition and compute $J$. This shows how reproducible a circuit is at this sample size. Without it, a low $J_{PT}$ cannot be told apart from noise.
- **Threshold-free:** Spearman correlation of per-head restoration scores across *all* heads between conditions.
- **Comparability metrics:** containment $|A\cap B|/|A|$ (to compare with Lietzow's 61%), and a random-normalized $J$ (observed over chance), as used by Nikankin et al. for circuit IoU.
- **Shared-object check:** P and T keep different object subsets (prior override succeeds vs. attack succeeds). Recompute head sets on objects present in both subsets and report $J$ there too; if it differs a lot from the full-set $J$, sample selection is driving the result.
- **Layer profile:** overlap per layer band, since heads cluster in the second half of the network.
- **Bootstrap CIs** over objects for every statistic.
- Cross-model comparison between Qwen and PaliGemma (their routing mechanisms differ per Lietzow).

### 3.6 Cross-ablation (H2)
$$T_{A\to B} = P(\hat y_B = y_{\text{grounded}} \mid \text{ablate } \mathcal{H}_A) - P(\hat y_B = y_{\text{grounded}} \mid \text{no ablation})$$
Compute $T_{P\to T}$ and $T_{T\to P}$, each against **random controls** matching the head count and layer distribution (many draws). Report collateral effects: clean accuracy on the Visual prompt (original image, no note) after ablation. Robustness: mean-ablation in addition to zero-ablation.

### 3.7 Mechanism (H3)
- For T-circuit heads, attention mass on note tokens vs object tokens under incongruent vs neutral notes (uses the known overlay mask).
- **Spatial analysis (for the GSR-VLM audience; small, secondary):** using the existing 2-3 layouts per object, report attack success and note-token attention as a function of note position (on vs. beside the object) and note size relative to the object. No new data needed.
- Lietzow-style logit-lens on head-output differences: does any head write the note's color token?
- Qwen2.5-VL: set `min_pixels = max_pixels` for a deterministic grid, and account for the 2×2 token merge when mapping masks to tokens.

### 3.8 External validity (H4)
Evaluate $\mathcal{H}_P$ and $\mathcal{H}_T$ ablations on SCAM/NoSCAM (object identification). Report as secondary. Do not mix SCAM into the headline $J$.

---

## 4. Replication Gate and Go/No-Go (≈3 days)

Do this before anything else; it replaces the original 24-hour PoC. **Run step 3 (T viability) first or in parallel:** it is the cheapest kill-test, and nothing yet shows Qwen2.5-VL-3B falls for notes.

1. **CPU smoke tests pass:** a tiny random-init model exercises every hook, shape, mask, and ablation (§7).
2. **Replication gate (Qwen2.5-VL-3B):** rerun Lietzow's pipeline. Suggested pass criteria (set final values in DECISIONS.md): the rerun's promoting set substantially overlaps their Table 5 list (Appendix A), group ablation flips a clear majority of prior-grounded predictions, and the top individual heads in their Table 7 (L26H5: 41.1%, L31H3: 37.0%) rank near the top.
3. **T-condition viability:** the incongruent note moves the answer on enough examples. Suggested minimum: ≥ ~50 usable fooled objects for the primary model.

| Outcome | Action |
|---|---|
| Gate passed | Proceed to §10 |
| Replication fails | Check indexing, tokenization, prompt template, fp16 overflow before any conclusion |
| Too few fooled examples | Increase note salience (size, contrast), try PaliGemma as primary, or use a continuous variant: patch on examples where the note shifts the logit difference significantly even without an argmax flip |

---

## 5. Compute (rough; measure in the smoke test and revise)

| Phase | Estimate |
|---|---|
| Replication gate (P, V) on Qwen-3B | ~2-5 GPU-h |
| T-condition data + behavioral screen | ~2-4 GPU-h |
| Full head sweep per condition (with the last-token speed-up or attribution patching) | ~5-15 GPU-h |
| Cross-ablation + random controls | ~3-6 GPU-h |
| H3 mechanism, H4 SCAM | ~3-6 GPU-h |
| PaliGemma-3B repeat | ~8-15 GPU-h |

**[VERIFY]** Kaggle's current weekly GPU quota and session limits. Save intermediate JSON frequently so a timed-out session loses little. Kaggle internet access (needed to `git clone`) requires account verification.

---

## 6. Risks

1. **Few "correctly conflicting" examples.** Lietzow had 73 for Qwen-3B. Circuit sets estimated on ~70 examples are noisy, which is why split-half ceilings and bootstrap CIs are mandatory. Do not interpret a low $J$ as "different circuits" unless it is clearly below the split-half ceiling. Use equivalence-style wording ("overlap not distinguishable from chance at this n") rather than absolute claims.
2. **Qwen may resist typographic attacks.** Mitigate as in §4.
3. **Role asymmetry** of the token pair (§3.2). Separate routing and writing analyses.
4. **Prompt differs between P and T** (T uses the Visual prompt). Acknowledge. A noted image with the Prior prompt is a possible extension.
5. **Post-it presence vs conflict.** The neutral-word note baseline removes the "a note exists" effect. Keep the no-note and congruent baselines as secondary checks.
6. **Head-index convention** or checkpoint version differences versus Lietzow. Caught by the replication gate.
7. **fp16 overflow** on Qwen. Check logits against a higher-precision reference.
8. **Decodable ≠ causally used.** Treat patching and ablation as the causal evidence, and any probe as supporting only.
9. **Scoop risk.** The area moves fast. Re-search weekly.
10. **Citation errors.** The v1 plan misdescribed a key paper. Never cite numbers from memory. Quote from the PDF.

---

## 7. Engineering Notes

- **Develop with a git repo** (`src/`, `configs/`, `tests/`, `results/`, `notebooks/`, `docs/`). Notebooks are thin wrappers around `src/`.
- **CPU smoke tests with a tiny random-init config** of the target architecture: one forward pass, hook placement, mask-to-token mapping, ablation zeroes exactly the intended heads, patching returns the source activation.
- **Unit tests** for $\Delta$, $R$, flip rate, $J$, the permutation null, and split-half code on dummy tensors.
- **Head lists are data:** `configs/heads_lietzow_qwen3b.json`, etc., loaded from files, never typed from memory.
- **Every result file records:** model, checkpoint, $n$, seed, prompt, note layout, commit hash.
- Large artifacts (weights, activations) stay out of git. Keep small JSON results and configs.

---

## 8. Optional: Pre-Generation Probe (only if time remains)

If revived, fix the issues in the original plan:
- Restrict probes to layers **before** the critical window (Lietzow: windows begin at 52-76% of depth) and compare against a logit-lens baseline at the same layer.
- Control the format confound (note area, contrast, word frequency) with nested cross-validation and evaluation within area bins.
- Include a shuffled-label negative control.
Frame it as methodological transfer, not a first discovery.

---

## 9. Paper Outline (length per GSR-VLM call; **[VERIFY]** page limit)

1. **Intro:** three sources of "distrust vision"; question and why it matters for VLM robustness; contributions. Frame as interpretability and robustness of VLMs (GSR-VLM topic), not as spatial reasoning.
2. **Related work:** perception-knowledge conflict circuits; typographic attacks (encoder-side); circuit overlap work.
3. **Method:** the three contrasts; note generator; discovery pipeline; overlap and cross-ablation statistics.
4. **Results:** replication; $J$ with chance floor and ceiling; all-head correlation; cross-ablation with random controls; mechanism (attention to note tokens); SCAM check.
5. **Discussion and limitations:** token-role asymmetry, small $n$, model scale, last-token-only interventions.

**Figures:** (1) design schematic; (2) head-score scatter per condition with promoting heads marked; (3) overlap bars with chance/ceiling bands; (4) cross-ablation vs random controls; (5) attention-on-note-tokens by note position and size.

**If the limit is 4 pages:** headline = replication + P/V/T overlap + cross-ablation on Qwen2.5-VL-3B; H3 shrinks to one figure; everything else goes to the appendix or is cut.

**Working titles:**
- "Does In-Image Text Hijack the Same Heads as Prompted Priors?"
- "One Circuit or Two? Overlap Between Prior-Override and Typographic Override in VLMs"

---

## 10. Timeline (to 20 Nov 2026; adjust if GSR-VLM's own deadline differs)

- **16-19 Oct:** confirm GSR-VLM rules and in-person attendance; repo and smoke tests; note generator; **T behavioral screen**; start Lietzow pipeline on Qwen-3B.
- **20-26 Oct:** finish replication gate (§4); save P and V head lists; pre-register thresholds in DECISIONS.md; scoop re-search.
- **27 Oct-2 Nov:** T-circuit discovery; $J$, containment, correlations, split-half ceilings, shared-object check, bootstrap.
- **3-9 Nov:** cross-ablation with random controls; H3 including note-position analysis. Start writing intro and method in parallel.
- **10-16 Nov:** figures and full draft; red-team review of every claim and number.
- **17-20 Nov:** polish, anonymize, check format; submit with a buffer day.
- **Stretch (only if ahead):** PaliGemma-3B repeat, SCAM check, Qwen-7B, H1d′.

---

## Appendix A. Lietzow et al. Head Lists (copied from their Tables 5-7; **[VERIFY against the PDF before use]**)

Format: L{layer}H{head}. Confirm whether indices are 0-based and how they map to HF module paths.

**Qwen-VL 3B** (9 promoting): L26H0, L26H5, L26H6, L27H1, L27H4, L28H3, L31H3, L31H7, L34H14
Suppressing (6): L26H3, L27H2, L28H4, L30H3, L31H8, L34H10
Single-head prior-flip rates: L26H5 41.1%, L31H3 37.0%.

**PaliGemma 3B** (8 promoting): L15H7, L16H4, L17H7, L20H3, L22H1, L24H3, L25H2, L25H4
Suppressing (2): L17H4, L25H5
Single-head: L15H7 58.7% (alone ≈ 60% of group effect), L20H3 36.4%.

**Qwen-VL 7B** (21 promoting): L17H21, L18H8, L18H9, L18H24, L19H22, L19H23, L19H24, L20H1, L20H3, L20H5, L21H5, L21H19, L22H1, L22H13, L23H6, L23H11, L24H21, L24H27, L26H24, L26H25, L26H26
Suppressing (8): L18H7, L20H2, L20H21, L21H14, L22H25, L25H25, L26H22, L27H3
Redundancy: effect distributed (85.5% of group-flipped examples cannot be flipped by any single head).

## Appendix B. Reference Numbers from Lietzow et al.

- Group ablation of promoting attention heads, Prior / Visual flip rates: Qwen-VL 3B 84.9 / 1.4; Qwen-VL 7B 68.4 / 2.4; LLaVA-NeXT 75.0 / 7.5; PaliGemma 3B 95.9 / 0.8; PaliGemma 10B 88.7 / 1.7.
- Classified heads: 2.5-4.8% of all heads; critical window begins at 52-76% of depth, spanning 7-16 layers.
- Correctly conflicting examples (of 467): Qwen-VL 3B 73; Qwen-VL 7B 212; LLaVA-NeXT 80; PaliGemma 3B 121; PaliGemma 10B 177.
- Conflict accuracy (Prior prompt, counterfactual image): Qwen-VL 3B 17.7%.
- V-contrast (Appendix F): 1.6-5.8% of heads classified; 61% of V-classified heads also classified under the P contrast.
- Tools: NNsight for interventions.

## Appendix C. Living Documents to Keep Alongside This Plan

Master copies live in the repo's `docs/` folder. After any change, replace the copy in the Project's knowledge (delete the old file, upload the new one).


- `STATE.md`: current status, open questions, next steps, blockers.
- `DECISIONS.md`: dated decisions and pre-registered thresholds.
- `RESULTS.md`: every number with model, $n$, seed, prompt, commit.
