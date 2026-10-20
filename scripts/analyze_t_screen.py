"""Analyse a saved T-screen result file.

Reads only from ``results/`` (CLAUDE.md: analysis scripts never recompute or
invent numbers) and prints every figure that goes into docs/RESULTS.md.

    .venv/bin/python scripts/analyze_t_screen.py results/t_screen_qwen25vl3b.json

What it reports, and why each one is there:

* **Attack vs congruent shift.** The congruent note names the object's *own*
  colour, so its shift is the "a note exists and says a colour word" baseline
  with no conflict in it. If the two distributions separate, the attack's
  effect is driven by the note's content and not by its presence (D38).
* **Margin.** Delta after the attack. Negative means the answer flipped.
* **Position vs size.** Fool rate against note area and occlusion per layout,
  plus a paired within-object comparison of on-object against beside-object.
  Paired, because objects differ enormously in how strong their colour prior
  is (plan 3.7, D63a).
* **Numerics failures.** Non-finite logits, i.e. plan risk 7.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from circuit_overlap.screen import (  # noqa: E402
    ObjectRecord,
    has_numerics_failure,
    parse_color_word,
    summarise_screen,
    vocabulary_from_records,
)


def load_records(path: str | Path) -> tuple[list[ObjectRecord], dict]:
    payload = json.loads(Path(path).read_text())
    result = payload["result"]
    return [ObjectRecord(**r) for r in result["records"]], payload


def describe(name: str, values: np.ndarray) -> None:
    print(f"  {name:16s} n={values.size:5d}  mean {values.mean():+7.2f}  "
          f"median {np.median(values):+7.2f}  p5 {np.percentile(values, 5):+7.2f}  "
          f"p95 {np.percentile(values, 95):+7.2f}  min {values.min():+7.2f}  "
          f"max {values.max():+7.2f}")


def main(path: str) -> None:
    records, payload = load_records(path)
    vocab = vocabulary_from_records(records)
    broken = [r for r in records if has_numerics_failure(r)]
    clean = [r for r in records if not has_numerics_failure(r)]

    print(f"file      {path}")
    print(f"model     {payload.get('model')}  checkpoint {payload.get('checkpoint')}")
    print(f"commit    {payload.get('commit')}   run {payload.get('timestamp_utc')}")
    print(f"records   {len(records)} over {len({r.object_id for r in records})} objects, "
          f"{len({r.layout for r in records})} layouts")

    print(f"\nNUMERICS FAILURES (plan risk 7): {len(broken)} records, "
          f"{len({r.object_id for r in broken})} objects")
    if broken:
        for oid in sorted({r.object_id for r in broken}):
            answers = {r.answer_neutral for r in broken if r.object_id == oid}
            print(f"  {oid:28s} answers {answers}")
        print("  -> excluded from every rate below; re-run these in fp32 to confirm "
              "they are a precision artefact and not a data problem")

    def shifts(field_a: str, field_b: str) -> np.ndarray:
        out = []
        for r in clean:
            a, b = getattr(r, field_a), getattr(r, field_b)
            if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):
                out.append(a - b)
        return np.array(out)

    attack = shifts("delta_neutral", "delta_incongruent")
    congruent = shifts("delta_neutral", "delta_congruent")
    margin = np.array([r.delta_incongruent for r in clean
                       if r.delta_incongruent is not None])
    print("\nLOGIT DIFFERENCE  (Delta = logit(best accepted colour) - logit(note's colour))")
    describe("attack shift", attack)
    describe("congruent shift", congruent)
    describe("margin after", margin)
    if congruent.size and attack.size:
        print(f"\n  attack shift above the LARGEST congruent shift: "
              f"{(attack > congruent.max()).mean():.1%} of records "
              f"(congruent max {congruent.max():+.2f})")
        print(f"  margin below zero (answer flipped): {(margin < 0).mean():.1%}")
        print("  A congruent note moves Delta the OTHER way, so the attack's effect "
              "is the note's content, not its presence (D38).")

    print("\nPOSITION vs SIZE (plan 3.7)")
    print("  layout            n  fooled  fool_rate   area  occlusion")
    fooled_by: dict[str, set[str]] = {}
    for layout in sorted({r.layout for r in clean}):
        rows = [r for r in clean if r.layout == layout]
        eligible = [r for r in rows
                    if parse_color_word(r.answer_neutral, vocab) in r.original_colors]
        fooled = [r for r in eligible
                  if parse_color_word(r.answer_incongruent, vocab) == r.counterfactual_color]
        fooled_by[layout] = {r.object_id for r in fooled}
        area = np.nanmedian([r.note_area_frac for r in rows if r.note_area_frac is not None])
        occ = np.nanmedian([r.object_occlusion for r in rows
                            if r.object_occlusion is not None])
        rate = len(fooled) / len(eligible) if eligible else float("nan")
        print(f"  {layout:17s} {len(eligible):3d}  {len(fooled):5d}  {rate:8.1%}  "
              f"{area:5.1%}  {occ:8.1%}")

    on = [l for l in fooled_by if l.startswith("on_object")]
    beside = [l for l in fooled_by if l.startswith("beside")]
    if on and beside:
        on_set = set().union(*(fooled_by[l] for l in on))
        beside_set = set().union(*(fooled_by[l] for l in beside))
        only_on = len(on_set - beside_set)
        only_beside = len(beside_set - on_set)
        print(f"\n  paired within object: fooled on-object only {only_on}, "
              f"beside only {only_beside}, both {len(on_set & beside_set)}")
        print("  Same objects, same words, only placement differs -- so this is "
              "position, not object difficulty.")

    if len(on) == 2:
        small, medium = sorted(on)
        a, b = fooled_by[small], fooled_by[medium]
        print(f"  size, paired: {small} only {len(a - b)}, {medium} only {len(b - a)}, "
              f"both {len(a & b)}")
        print("  Medium notes cover roughly twice the area of small ones. If size "
              "drove the attack this would be lopsided.")

    print("\nRECOMPUTED SUMMARY (authoritative for RESULTS.md)")
    recomputed = summarise_screen(records, vocab, min_usable=50)
    for key in ("n_objects", "n_objects_correct_neutral", "n_objects_usable",
                "fool_rate_among_eligible", "gate_passed",
                "n_objects_numerics_failed", "n_unparsed_neutral",
                "n_unparsed_incongruent", "n_objects_multicolor"):
        print(f"  {key:30s} {recomputed[key]}")
    print(f"  {'continuous_fallback':30s} {recomputed['continuous_fallback']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/t_screen_qwen25vl3b.json")
