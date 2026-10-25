# RESULTS.md

Every reported number must have a row here. Reference numbers from other papers are NOT results; they live in plan_v2 Appendix B with a page citation.

| date | experiment | model | n | seed | prompt | commit | result | file path |
|---|---|---|---|---|---|---|---|---|
| 2026-10-19 | T screen, **pilot smoke run** (not a result) | Qwen2.5-VL-3B-Instruct, 4-bit nf4, emulated bf16 compute (T4) | 5 objects x 3 layouts | n/a (greedy) | Visual: "What color is this {object} here? Respond with the color name only." | f144b4d | **0 of 15 renders fooled.** Attack shift Delta_neutral - Delta_attacked = **+7.1 to +9.7 logits**, so the note is read and moves the logits strongly but does not cross zero. Margins (Delta after attack) not yet recorded - the smoke cell now prints them. | not saved (console only) |

| 2026-10-20 | **T-condition behavioural screen, full run** (D41 gate) | Qwen2.5-VL-3B-Instruct, 4-bit nf4 double-quant, emulated-bf16 compute on a T4 | 469 objects x 3 layouts x 3 variants = 1407 records | n/a (greedy) | Visual: "What color is this {object} here? Respond with the color name only." | run under `f144b4d`, saved under `c34c2b1` | **GATE PASSED: 286 usable (fooled) objects** of 436 eligible = **65.6%**; bar was 50. Clean accuracy with a neutral note 93.0% (436/469). Attack shift (Delta_neutral - Delta_attacked): mean **+8.50**, median +8.56, **min +2.12** -- every record moved toward the note. Congruent shift: mean **-2.10**, max +1.38 -- congruent notes move Delta the *other* way. **100% of attack shifts exceed the largest congruent shift.** Margin after attack: median -0.38, negative (flipped) in 56.2%. Numerics failures 6 records / 5 objects. Unparsed 14 neutral, 3 incongruent. | [results/t_screen_qwen25vl3b.json](../results/t_screen_qwen25vl3b.json), [results/t_usable_objects.json](../results/t_usable_objects.json), checkpoint [results/t_screen_checkpoint.jsonl](../results/t_screen_checkpoint.jsonl) |
| 2026-10-20 | Position vs size of the note (plan 3.7), same run | as above | 1401 clean records | n/a | as above | `c34c2b1` | Fool rate by layout: **on_object_small 58.3%** (area 3.2%, occlusion 4.7%), **on_object_medium 56.7%** (area 6.4%, occlusion 9.3%), **beside_object 44.2%** (area 5.5%, occlusion 2.3%). Paired within object: fooled on-object only **99**, beside only **20**, both 167. Paired on size: small only 29, medium only 24, both 213. | same files; reproduce with `scripts/analyze_t_screen.py` |
| 2026-10-25 | **Replication gate, part 1: group-ablation flip rate** (P condition, D40) | Qwen2.5-VL-3B-Instruct, 4-bit nf4 double-quant, **fp16 compute** on a T4 (see the file's `correction` field) | 73 correctly-conflicting objects of 469 | n/a (greedy, top-1 of first answer token) | Visual and Prior prompts, both with "Respond with the color name only." | `a922e93` | Ablating Lietzow's 9 promoting heads (zeroed o_proj input slice, last token): **prior flip 66/73 = 90.4%** [81.5, 95.3] vs Lietzow 84.9%, Fisher p = 0.45; **visual flip 5/73 = 6.8%** [3.0, 15.1] vs Lietzow 1.4%, Fisher p = 0.21. Correctly conflicting 73 (Lietzow 73); top-1 differs between prompts 105. All 71 flips are colour changes, none a case variant of the same colour. | [results/replication_flip_rate_qwen25vl3b.json](../results/replication_flip_rate_qwen25vl3b.json); reproduce with `scripts/analyze_replication.py` |

### Notes on the replication flip-rate row (2026-10-25)
- **Reproduce with:** `.venv/bin/python scripts/analyze_replication.py results/replication_flip_rate_qwen25vl3b.json`. Intervals are Wilson 95%; Lietzow's counts are reconstructed from their percentages and n = 73, so they are approximate.
- **Prior flip matches.** 90.4% is a clear majority and statistically indistinguishable from 84.9%.
- **Visual flip is higher than theirs but not distinguishable at n = 73** (5 vs ~1 objects, p = 0.21). Suspects if it persists: fp16 compute here vs their likely bf16; 469 vs 467 objects; `head_knockout` is a re-implementation, not their file. The 5 objects: blueberry, elk, great white shark, honeydew, skunk. The "0 of 71 are case variants" check was run in the notebook (console only), not stored in the file.
- **Compute dtype differs between conditions.** The T screen above ran in emulated bf16; this run in fp16. Keep it in mind when comparing P and T.
- **Only part 1 of D40.** It uses Lietzow's published heads. Re-discovering the heads by patching and the single-head checks (L26H5, L31H3) are still to do, and D40 has no numeric threshold yet.
- **Per-object answers were not committed.** `baseline_top1.jsonl` and `ablated_top1.jsonl` stayed in the Kaggle output; add them to `results/` if the flips need re-analysis.

### Notes on the full-run rows
- **Reproduce every number above with:** `.venv/bin/python scripts/analyze_t_screen.py results/t_screen_qwen25vl3b.json`. Nothing here is typed by hand.
- **D38 is settled by the congruent baseline.** A congruent note (naming the object's own colour) shifts Delta by -2.10 on average, i.e. *away* from the counterfactual, while the attack shifts it +8.50. The distributions do not overlap at all: 100% of attack shifts exceed the largest congruent shift (+1.38). So the attack's effect is the note's **content**, not the mere presence of a note. The pre-registered continuous criterion can therefore be stated as "attack shift above the maximum congruent shift", which every record satisfies.
- **Position beats size, paired.** The on-vs-beside asymmetry (99 vs 20) is within the same objects with the same words, so it is not object difficulty. The size comparison is near-symmetric (29 vs 24) even though medium notes cover roughly double the area and double the occlusion of small ones. Conclusion: the attack works by sitting **on** the object, not by covering it. The occlusion confound raised during design is empirically absent in this range.
- **Half-precision overflow is real (plan risk 7).** 5 objects (`ant_0039`, `bench_0067`, `boat_0080`, `crab_0144`, `otter_0306`) returned non-finite colour logits; `bench_0067` also generated `'!!!!'`, Qwen's overflow signature. 6 of 1407 records (0.4%). They are excluded from every rate by `screen.has_numerics_failure` and reported separately. **To do:** re-run those 5 in fp32 to confirm precision, not data, is the cause.
- **Reproducibility note.** An earlier run of the same screen with 2 variants and emulated-bf16 compute gave the same 286 usable objects and the same 65.6% rate, but slightly different per-layout counts (183/236/238 vs 187/237/242) and unparsed counts. So the headline is stable across the precision and batching changes while individual borderline records are not.
- **Not yet done:** these are behavioural rates, not circuit results. No head has been patched or ablated.

### Correction (2026-10-20)### Correction (2026-10-20): the compute dtype of both rows above
The two rows above originally said "fp16 compute (T4, no bf16)". That was wrong.
`torch.cuda.is_bf16_supported()` returns **True on a T4** because it counts
emulation, so the loader selected bfloat16 and the runs used **emulated bf16** --
which is also the likely reason for 8.4 s/generation. `t_screen.native_bf16`
now requires Ampere or newer, so a T4 computes in fp16. Any re-run will differ
from these rows in compute dtype, and the result envelope records it.

### Notes on the full-run row
- **Per-record data was not saved**, so this row is all that survives. The usable-object ids are needed as the example set for T-circuit discovery, so the run must be repeated. `run_screen` now checkpoints per object (commit below), and the congruent variant is included, so the repeat also yields the D38 baseline.
- **Position beats size.** `on_object_small` and `on_object_medium` fool equally (57.2% vs 56.5%) despite different occlusion, while moving the note off the object costs ~14 points. So the attack is not winning by hiding the object -- the occlusion confound of D63a is empirically absent at these sizes.
- The 65.6% fool rate sits far above Lietzow's prior-override conflict accuracy for the same model (17.7%). In-image text is a much stronger competing signal than a prompted prior here. Do not over-read: different conditions, and this is a behavioural rate, not a circuit claim.

### Notes on the pilot row
- **Five objects is a pilot, not evidence.** It cannot support any claim about whether Qwen2.5-VL-3B resists typographic notes; it exists to show the pipeline runs and to size the salience problem.
- The objects involved (`american chameleon`, `arctic fox`, ...) are the first 5 of the manifest, so they are **not** a random sample.
- Deviations from Lietzow to carry into any write-up: compute dtype (see the correction above); our object set is 469 vs their 467 (see STATE.md).

