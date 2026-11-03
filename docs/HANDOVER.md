# Handover, 3 Nov 2026 (Soham -> Kush, then both in parallel)

Written for kushviradiya and his Claude Code session. Read this, then
`CLAUDE.md` (rules, gotchas), then `docs/RESULTS.md` (every number) and
`docs/DECISIONS.md` D70-D74 (how each result was read). Plan of record:
`docs/plan_v2.md`. Paper: 6 pages, due **20 Nov 2026**; workshop still
unconfirmed (D01, D01a).

## 1. Where things stand

Everything below is run, analysed and committed. All on Qwen2.5-VL-3B, 4-bit nf4.

| Step | Result (details in RESULTS.md) | Decision | Reproduce |
|---|---|---|---|
| T behavioural screen | 286 usable of 436 eligible; notes fool the model, by note *content* | D41, D38 | `scripts/analyze_t_screen.py` |
| Replication gate (P) | Re-discovered exactly Lietzow's 9 promoting / 6 suppressing heads; knockout flips 89.0% prior, 2.7% visual | D40, D67 | `scripts/analyze_replication_gate.py` |
| P split-half ceiling | J between halves median 0.900 (chance 0.059) | D69 | `scripts/analyze_split_half.py` |
| T circuit, J_PT (headline) | 12 T heads; **J_PT 0.312 [0.278, 0.375]**, "above chance, below self-agreement"; 5 shared heads | D70-D72 | `scripts/analyze_t_circuit.py` |
| H2 cross-ablation | Both primary tests null. Shared heads carry P (83.6%) not T (0/244); T runs through its 7 T-only heads (70.5%) | D73 | `scripts/analyze_cross_ablation.py` |
| H3 mechanism | **T-only heads read the note and write its colour**; with no note they attend to the recoloured object and write the image's colour. Shared heads do neither | D74 | `scripts/analyze_h3.py` |

**The story so far, in one paragraph.** Patching selects partly the same
heads for prompted-prior override (P) and typographic override (T), but
the overlap is in selection, not cause: the 5 shared heads are what P
needs and do nothing for T. T runs through 7 later heads (layers 29-33)
that copy colour from wherever it is shown in the image -- the note if
there is one, the object otherwise -- into the answer.

## 2. What is left, split so it can run in parallel

Two tracks with **no shared files** except the append-only docs. Suggested
owners in brackets; swap if you prefer, and update the CLAUDE.md notebook
table if you do.

### Track A -- V condition, J_VT [Kush: it reuses his patching pipeline]

The plan's third contrast and the only headline prediction not yet tested:
H1-specific predicts **J_VT > J_PT**, because V and T both vary the image
(plan 1, 3.4, 3.5). It is also what Lietzow's "61% of V heads are also P
heads" compares against (a *containment*, not a Jaccard; report both).

1. **Settle the contrast first.** V = original vs counterfactual image,
   prompt fixed. Which prompt is held fixed is an open question (STATE.md):
   read `src/vdpo/types/contrast.py` in Lietzow's repo (commit `145b453`,
   `PRIOR_CIRCUIT_CONTRAST` and its visual variant). Also check which
   objects they keep for V (both images answered correctly?).
2. **Pre-register D75 before any code or run** (pull first; IDs collide).
   Reuse D70/D72's reading of J (chance floor, P ceiling, bootstrap) for
   J_PV and J_VT, and fix: object selection, n cap, the token pair and
   source/target roles, and whether H2/H3-style tests on V are in scope
   (suggest not, for time).
3. **Notebook** `notebooks/v_patching.ipynb`, built from
   `replication_patching.ipynb` / `t_patching.ipynb`; output prefix
   `/kaggle/working/v_patching/`; results `results/v_patching_*_qwen25vl3b.*`.
   Add its row to the CLAUDE.md ownership table and prefix list in the same
   commit. Checkpoint per object (D44). **Time one object first and multiply
   out** before the full run; V may keep far more objects than P's 73, so a
   cap may be needed.
4. Analysis script in `scripts/`, CPU-tested like the others, committed
   before the run. Append the rows to RESULTS.md.

Target: D75 pushed by **5 Nov**, results by **~10 Nov** so they make the draft.
If V can't land by then, the paper goes out without it and says so (P and T
alone already answer the title question).

### Track B -- paper and figures [Soham: needs no GPU]

1. **Figures from committed results**, as scripts in `scripts/figures/`
   writing to `figures/` (plan 9): design schematic; head-score scatter per
   condition with promoting heads marked; J_PT with chance and ceiling
   bands (leave a slot for J_VT); cross-ablation vs random controls;
   H3 attention/direct effect vs random, plus the 3-layout spatial panel.
   matplotlib is not in `.venv`: ask before installing it.
2. **Draft**: intro, method, results, limitations. Where the source lives
   (Overleaf or `paper/` in the repo) is still to agree.
3. **Related work and novelty**, before claiming anything (STATE.md open
   questions): Hua et al. 2025 (image-vs-caption routing heads in
   Qwen2.5-VL), Nikankin et al. 2025 (circuit IoU), Dyslexify, Ortu,
   Golovanevsky 2025a/b. Quote numbers from the PDFs, never from memory.
4. **Venue** (STATE.md blocker 1) and the dataset licence (open question).

### Joint, ~16-20 Nov
Red-team every claim and number against RESULTS.md (plan 10), polish, submit.

### Out of scope unless both agree (recommend: future work)
PaliGemma-3B repeat (8-15 GPU-h, whole pipeline to port), SCAM/NoSCAM (H4),
Qwen-7B, the pre-generation probe (plan 8). Smaller open questions from H2/H3,
each worth one sentence in the limitations if not run: why patching selects
the shared heads under T when they neither matter nor attend there; what
backs up the T-only heads' visual-colour role (removing them barely hurts
clean accuracy); mean ablation of the decomposition sets.

## 3. Caveats the paper must carry (from RESULTS.md)

- **Token roles differ** between P and T (plan 3.2): same colour pair, swapped roles.
- **J_PT is overlap of selection**; H2 shows the causal overlap is near nil.
- **H3's "object" tokens are the recoloured pixels**, not the whole object
  (the old english sheepdog's white fur is not in its set). Call it "the
  recoloured region".
- **H3's W test is partly built in** (the heads were selected for pushing
  toward the note's colour); the attention test A is the informative one.
- **p = 0.0099 is the floor** with 100 random draws; quote the excess and its CI.
- Last-token interventions and last-token attention only; one model; 4-bit.
- Reference numbers from Lietzow are not our results (RESULTS.md header).

## 4. Things that will bite (beyond CLAUDE.md's gotchas)

- **Only `h3_attention.ipynb` loads with eager attention**
  (`ScreenConfig(attn_implementation="eager")`); everything else is sdpa,
  and eager moved 23 near-tie objects. Do not mix the two in one comparison.
- `cross_ablation.build_plan` is the one source of the head sets
  (H_P, H_T, shared, P-only, T-only) and of the 100 layer-matched random
  draws; reuse it rather than rebuilding sets.
- `overlap.py` / `ceiling.py` hold the J, chance and ceiling code that D72
  fixed; J_VT should go through the same functions.
- Results come back by hand from Kaggle (no token there); commit, then push
  straight away -- an unpushed commit is invisible to the other person and
  to Kaggle.
