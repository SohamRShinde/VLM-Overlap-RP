# DECISIONS.md

Status key: **MADE** = adopted in plan_v2 and not challenged by the audit. **TENTATIVE** = in plan_v2 but has open parameters or depends on unverified facts. **PROPOSED** = raised in the chat-1 audit; not yet adopted. **BLOCKED** = contradicted by uploaded files or needs external info.
**REOPENED** = was recorded as made, but has been withdrawn and is open again.
Source date for plan_v2 decisions: 2026-10-15. Audit annotations: 2026-10-16. Updated for plan_v2.1: 2026-10-16. Chat-2 updates (venue reopened, dataset schema, implementation decisions): 2026-10-18.

## Venue and scope
| ID | Decision | Status | Notes |
|---|---|---|---|
| D01 | ~~Target AAAI-27 AIR-FM~~ ~~→ Target AAAI-27 GSR-VLM workshop~~ → **No AAAI-27 venue chosen** | **REOPENED (2026-10-18)** | AIR-FM was the **AAAI-2026** edition and is not on the AAAI-27 list. GSR-VLM was proposed in plan_v2.1 §0.8 as the only VLM-specific AAAI-27 workshop, but it was never actually decided. Treat plan_v2.1's GSR-VLM references as a candidate, not the plan of record. Everything downstream of the venue (D01a, D02, D04, D07) is provisional until this is settled. |
| D01a | Page limit, template, anonymity, archival status | **BLOCKED** (on D01) | Unknowable until a workshop is chosen. |
| D02 | Timeline to AAAI general deadline 20 Nov 2026 (AoE); notifications 2 Dec | TENTATIVE | The general deadline is the planning date; the chosen workshop may differ. |
| D04 | Framing: interpretability/robustness study of VLM arbitration between visual evidence and competing signals; small spatial analysis (note position/size); not a spatial-reasoning paper | TENTATIVE (was MADE; downgraded 2026-10-18) | The spatial add-on (§3.7) existed to fit GSR-VLM's topic list. It is still a reasonable secondary analysis, but its prominence should follow the venue decision, not precede it. |
| D05 | Confirm one author can present in person (AAAI does not permit virtual presentations) | **BLOCKED** | Resolve before submitting. |
| D06 | Team size | **BLOCKED** | Project description says 2; plan v1/v2 said 4. Affects scope. |
| D07 | If page limit is 4: headline = replication + P/V/T overlap + cross-ablation on Qwen2.5-VL-3B only | TENTATIVE | plan_v2.1 §9 |
| D03 | Low- and high-overlap outcomes both publishable; equivalence-style wording for nulls | MADE | plan §1, §6.1 |

## Models
| ID | Decision | Status | Notes |
|---|---|---|---|
| D10 | Primary model: Qwen2.5-VL-3B, fp16 (no 4-bit) for all mechanistic work | MADE (checkpoint TENTATIVE) | Exact HF checkpoint not given in Lietzow; check repo. fp16 overflow check vs higher-precision reference. |
| D11 | Second model: PaliGemma-3B, Gemma-2-backbone variant | TENTATIVE | Lietzow cites PaliGemma 2; variant/resolution unverified. |
| D12 | Stretch: Qwen2.5-VL-7B | TENTATIVE | T4 memory unverified. |
| D13 | No LLaVA-1.5 (not in Lietzow's set) | MADE | Confirmed: Lietzow models are Qwen-VL 3B/7B, LLaVA-NeXT 7B, PaliGemma 3B/10B. |

## Task and data
| ID | Decision | Status | Notes |
|---|---|---|---|
| D20 | Single task: color identification on Visual-Counterfact, radish and spider excluded | MADE (count **still open**) | Source pinned 2026-10-18: `mgolov/Visual-Counterfact`, `split="color"`, **493 rows**, verified in the Hub viewer. Columns: `object`, `correct_answer`, `incorrect_answer`, `original_image`, `counterfact_image` (recorded in `configs/data.json`). 493 vs Lietzow's 469/467 is unresolved — rows vs unique objects, an earlier revision, or undocumented filtering. See STATE.md; checked by notebook cell 2b. Licence not yet verified. |
| D29 | **Reference colour for Δ.** `correct_answer` is a *list* of acceptable colours, but Δ = logit(c_o) − logit(c_cf) needs one c_o. Rule: use the colour the model itself gives on the **neutral-note** image when that answer is in the accepted set; otherwise fall back to the first listed colour. Record the colour used in every result row (`reference_color`). | MADE (2026-10-18) | Implemented in `t_screen.choose_reference_color`. Rationale: the model's own neutral-note belief is what the attack has to overturn, so the Δ shift measures the attack rather than an arbitrary choice between two correct colours. The alternative (always the first listed colour) can make Δ_neutral already negative on an object the model answers correctly, which would mislabel a working attack as a no-op. Both are recoverable after the fact: the run stores every colour's first-token logit. |
| D29a | **Multi-colour objects: headline restricted to single-colour objects?** Proposal: headline analysis uses objects whose `correct_answer` has exactly one colour; multi-colour objects are reported as a robustness check. | **PROPOSED — decide before seeing T results (D38)** | Argument for: the token-role analysis (plan §3.2) assumes one c_o, and with two accepted colours the "unattacked" logit mass is split across tokens, shrinking Δ for reasons unrelated to the attack. Argument against: it discards data when n is already tight (Lietzow had 73 correctly-conflicting examples), and single-colour objects may be the prototypical ones, biasing the sample. The screen reports `n_objects_multicolor` so the cost is known before committing. |
| D21 | Three conditions: P (prompt varies, CF image), V (image varies, prompt fixed; Lietzow App. F), T (note word varies on original-color image, Visual prompt) | MADE | V prompt choice unverified (not stated in Lietzow App. F). |
| D22 | Prompts: Lietzow templates; Visual prompt used for T | MADE (string TENTATIVE) | "here?" discrepancy between Lietzow §3.1 and Table 1 caption. |
| D23 | Logit-difference token pair (c_o, c_cf) in all conditions; token roles differ (acknowledged) | MADE | Analyze routing vs writing heads separately. |
| D24 | Note variants: incongruent (c_cf), neutral non-color word (primary baseline), congruent (c_o), no note | MADE | |
| D25 | 2–3 layouts per object; bootstrap over objects, not images | MADE | |
| D26 | Only color words with single first token per tokenizer | MADE | Per-tokenizer check pending. |
| D27 | Correctly-conflicting restriction; for T = correct on neutral note AND fooled by incongruent note | MADE | "Correct" is **membership** in `correct_answer`'s colour list, not equality (D29). Implemented in `screen.classify`. |
| D28 | Also report overlap on objects shared between P and T example sets | MADE (2026-10-16) | Audit: P and T select different object subsets. plan_v2.1 §3.5 |

## Circuit discovery and statistics
| ID | Decision | Status | Notes |
|---|---|---|---|
| D30 | Discovery identical to Lietzow: last-token patching of head output z (W_O input), both directions, restoration score R_d, PCA on per-head mean restoration vectors, ±2σ on PC1 | MADE | Confirmed against Lietzow §3.2. |
| D31 | Zero-ablation of head outputs at last token; group flip rates; mean-ablation as robustness | MADE | |
| D32 | Speed-up: last-token-only recompute, or attribution patching + real patching on top 30–50 | TENTATIVE | Validate vs full runs on ~10 examples. |
| D33 | Overlap: Jaccard on promoting sets; permutation chance floor + p-value; split-half ceiling; Spearman over all heads; layer-band profile; bootstrap CIs; random-normalized J (as in Nikankin et al.) | MADE (normalized J added 2026-10-16) | plan_v2.1 §3.5 |
| D34 | Do not compare our J directly to Lietzow's 61% (containment fraction); report containment |A∩B|/|A| alongside J | MADE (2026-10-16) | plan_v2.1 §2, §3.5 |
| D35 | H2 cross-ablation T_{P→T}, T_{T→P} vs size- and layer-matched random head sets; report collateral clean accuracy | MADE | |
| D36 | H3 descriptive: attention mass on note vs object tokens (known mask); logit lens on head-output differences | MADE | Qwen: min_pixels = max_pixels, 2×2 merge (not verifiable from uploaded files). |
| D39 | H3 spatial add-on: attack success and note-token attention vs note position and size, from existing layouts | MADE (2026-10-16) | No new data; secondary. plan_v2.1 §3.7 |
| D37 | H4: SCAM/NoSCAM as secondary only, never pooled into headline J | MADE | SCAM LVLM protocol is a 2-option multiple-choice prompt. |
| D38 | Pre-register H1 thresholds and gate criteria before seeing T results | TENTATIVE | Values not yet set. |

## Gates
| ID | Decision | Status | Notes |
|---|---|---|---|
| D40 | Replication gate on Qwen-3B: substantial overlap with Lietzow Table 5; group ablation flips clear majority (they report 84.9%); L26H5 / L31H3 rank near top | TENTATIVE | Numeric pass thresholds not set. |
| D41 | T viability: ≥ ~50 usable fooled objects for primary model | TENTATIVE | |
| D42 | Run T behavioral screen before / in parallel with replication | MADE (2026-10-16) | Cheapest kill-test; no uploaded evidence that Qwen2.5-VL-3B is susceptible. plan_v2.1 §4 |

## Implementation (chat 2, 2026-10-18)
| ID | Decision | Status | Notes |
|---|---|---|---|
| D60 | Head sets are `(layer, head)` tuples, 0-based, loaded from `configs/heads_*.json`; never typed from memory | MADE | Matches D30. Index base against Lietzow still unverified (blocker 2). |
| D61 | Dataset access goes through a **manifest** (`object_id`, `object_name`, `image`, `original_colors`, `counterfactual_color`) written by the notebook; tested code never touches the dataset's own schema | MADE | `object` is not guaranteed unique across 493 rows and is also the prompt noun, so key and noun are separate fields. |
| D62 | Pin the image grid with `min_pixels = max_pixels = 448*448` (256 image tokens) for the T screen | MADE | Required for mask→token mapping (plan §3.7). Lower grids render the note's text a few pixels tall; an illegible note looks exactly like a robust model, i.e. a false negative on the kill-test. |
| D63 | Note layouts: centre = "on object", corner = "beside object"; 3 layouts in `configs/notes.json` | TENTATIVE | Approximation pending per-object masks. Affects the H3 spatial analysis, not the screen. |
| D64 | Every result file is written through `provenance.result_envelope` (model, checkpoint, n, seed, prompt, layouts, commit hash) | MADE | plan §7. Commit records `-dirty` or `unknown` rather than a fake hash. |

## Dropped / demoted
| ID | Decision | Status |
|---|---|---|
| D50 | H1c (text-manifold alignment) dropped | MADE |
| D51 | H1d′ (pre-generation probe) optional only | MADE |
