# RESULTS.md

Every reported number must have a row here. Reference numbers from other papers are NOT results; they live in plan_v2 Appendix B with a page citation.

| date | experiment | model | n | seed | prompt | commit | result | file path |
|---|---|---|---|---|---|---|---|---|
| 2026-10-19 | T screen, **pilot smoke run** (not a result) | Qwen2.5-VL-3B-Instruct, 4-bit nf4, fp16 compute (T4, no bf16) | 5 objects x 3 layouts | n/a (greedy) | Visual: "What color is this {object} here? Respond with the color name only." | f144b4d | **0 of 15 renders fooled.** Attack shift Delta_neutral - Delta_attacked = **+7.1 to +9.7 logits**, so the note is read and moves the logits strongly but does not cross zero. Margins (Delta after attack) not yet recorded - the smoke cell now prints them. | not saved (console only) |

| 2026-10-19/20 | **T-condition behavioural screen, full run** (D41 gate) | Qwen2.5-VL-3B-Instruct, 4-bit nf4, fp16 compute (T4) | 469 objects x 3 layouts = 1407 records | n/a (greedy) | Visual: "What color is this {object} here? Respond with the color name only." | f144b4d | **GATE PASSED. 286 usable (fooled) objects**, bar was 50. Fool rate among eligible 65.6% (286/436). Clean accuracy with a neutral note 93.0% (436/469). Mean attack shift +8.48 logits; every object passed the 1.0-logit continuous criterion, showing that placeholder is useless. Unparsed answers: 17 neutral, 2 incongruent. Per layout fool rate: on_object_small 57.2%, on_object_medium 56.5%, beside_object 43.0%. Runtime 23604 s for 2814 generations (8.4 s/generation). | **LOST** - the JSON-saving cell never ran (session closed). Summary above transcribed from notebook output; per-record data, margins, occlusion and the 286 usable object ids must be regenerated. |

### Notes on the full-run row
- **Per-record data was not saved**, so this row is all that survives. The usable-object ids are needed as the example set for T-circuit discovery, so the run must be repeated. `run_screen` now checkpoints per object (commit below), and the congruent variant is included, so the repeat also yields the D38 baseline.
- **Position beats size.** `on_object_small` and `on_object_medium` fool equally (57.2% vs 56.5%) despite different occlusion, while moving the note off the object costs ~14 points. So the attack is not winning by hiding the object -- the occlusion confound of D63a is empirically absent at these sizes.
- The 65.6% fool rate sits far above Lietzow's prior-override conflict accuracy for the same model (17.7%). In-image text is a much stronger competing signal than a prompted prior here. Do not over-read: different conditions, and this is a behavioural rate, not a circuit claim.

### Notes on the pilot row
- **Five objects is a pilot, not evidence.** It cannot support any claim about whether Qwen2.5-VL-3B resists typographic notes; it exists to show the pipeline runs and to size the salience problem.
- The objects involved (`american chameleon`, `arctic fox`, ...) are the first 5 of the manifest, so they are **not** a random sample.
- Deviations from Lietzow to carry into any write-up: 4-bit compute dtype is fp16 here (T4 has no bf16) where theirs was probably bf16; our object set is 469 vs their 467 (2 unexplained, see STATE.md).

