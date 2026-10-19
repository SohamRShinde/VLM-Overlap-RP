# RESULTS.md

Every reported number must have a row here. Reference numbers from other papers are NOT results; they live in plan_v2 Appendix B with a page citation.

| date | experiment | model | n | seed | prompt | commit | result | file path |
|---|---|---|---|---|---|---|---|---|
| 2026-10-19 | T screen, **pilot smoke run** (not a result) | Qwen2.5-VL-3B-Instruct, 4-bit nf4, fp16 compute (T4, no bf16) | 5 objects x 3 layouts | n/a (greedy) | Visual: "What color is this {object} here? Respond with the color name only." | f144b4d | **0 of 15 renders fooled.** Attack shift Delta_neutral - Delta_attacked = **+7.1 to +9.7 logits**, so the note is read and moves the logits strongly but does not cross zero. Margins (Delta after attack) not yet recorded - the smoke cell now prints them. | not saved (console only) |

### Notes on the pilot row
- **Five objects is a pilot, not evidence.** It cannot support any claim about whether Qwen2.5-VL-3B resists typographic notes; it exists to show the pipeline runs and to size the salience problem.
- The objects involved (`american chameleon`, `arctic fox`, ...) are the first 5 of the manifest, so they are **not** a random sample.
- Deviations from Lietzow to carry into any write-up: 4-bit compute dtype is fp16 here (T4 has no bf16) where theirs was probably bf16; our object set is 469 vs their 467 (2 unexplained, see STATE.md).

