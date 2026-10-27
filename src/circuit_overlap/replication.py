"""Shared pieces of the replication gate (plan section 4 step 2, D40).

Used by ``notebooks/replication_flip_rate.ipynb`` (knockout) and
``notebooks/replication_patching.ipynb`` (head re-discovery), so both select
the same objects and count flips the same way.

* ``lietzow_rows``: Lietzow's three row filters on Visual-Counterfact
  (plural / parenthetical names, unparseable colours, attack colour also
  correct) -- the same rules as ``t_screen.ipynb`` cell 5.
* ``is_correctly_conflicting``: their ``load_correct_example_ids``: the
  Visual prompt's top-1 is the counterfactual colour **and** the Prior
  prompt's top-1 is one of the real colours (token-id membership).
* ``knockout_flip_counts``: their pairwise flip (D67).
* ``run_cached``: append-only JSONL so a restarted kernel resumes (D44).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .data import lietzow_object_name, parse_color_list

__all__ = [
    "lietzow_rows",
    "color_token_sets",
    "is_correctly_conflicting",
    "knockout_flip_counts",
    "run_cached",
]


def lietzow_rows(dataset: Iterable[Mapping], inflect_engine) -> tuple[list[dict], Counter]:
    """Rows that survive Lietzow's loader, without their images.

    Each row is ``{"i", "name", "correct", "attack"}``; ``i`` indexes the
    dataset so images can be read on demand.
    """
    rows, dropped = [], Counter()
    for i, row in enumerate(dataset):
        try:
            name = lietzow_object_name(row["object"], inflect_engine=inflect_engine)
        except ValueError:
            dropped["name"] += 1
            continue
        try:
            correct = parse_color_list(row["correct_answer"], require_list=True)
            attack = parse_color_list(row["incorrect_answer"])[0]
        except (ValueError, TypeError):
            dropped["colours"] += 1
            continue
        if attack in correct:                   # radish and spider, per their README
            dropped["overlap"] += 1
            continue
        rows.append({"i": i, "name": name, "correct": sorted(correct), "attack": attack})
    return rows, dropped


def color_token_sets(row: Mapping, color_ids: Mapping[str, Sequence[int]]) -> tuple[set, set]:
    """(token ids of the attack colour, token ids of the real colours)."""
    attack = set(color_ids[row["attack"]])
    real = {t for c in row["correct"] for t in color_ids[c]}
    return attack, real


def is_correctly_conflicting(row: Mapping, top1_visual: int, top1_prior: int,
                             color_ids: Mapping[str, Sequence[int]],
                             ambiguous: Iterable[str] = ()) -> bool:
    """Visual top-1 is the shown colour and Prior top-1 is a real colour.

    Objects using a colour in ``ambiguous`` (one that shares a first token
    with another colour, D26) are excluded: a first-token test cannot tell
    their answers apart.
    """
    if set(ambiguous) & ({row["attack"]} | set(row["correct"])):
        return False
    attack, real = color_token_sets(row, color_ids)
    return top1_visual in attack and top1_prior in real


def knockout_flip_counts(records: Sequence[Mapping], config: str) -> dict:
    """Lietzow's pairwise flips for one knockout config (D67).

    Each record holds ``record[f"{config}/prior"]`` and
    ``record[f"{config}/visual"]`` as ``[logit of the prior run's argmax token,
    logit of the visual run's argmax token]`` under that knockout. A run is
    visual-grounded when the second exceeds the first (ties go to prior, as in
    their ``predicted`` field).
    """
    prior_flips = visual_flips = 0
    for r in records:
        lp, lv = r[f"{config}/prior"]
        if lv > lp:
            prior_flips += 1
        lp, lv = r[f"{config}/visual"]
        if not lv > lp:
            visual_flips += 1
    n = len(records)
    return {
        "n": n,
        "prior_flips": prior_flips,
        "visual_flips": visual_flips,
        "prior_flip_rate": prior_flips / n if n else float("nan"),
        "visual_flip_rate": visual_flips / n if n else float("nan"),
    }


def run_cached(path: str | Path, todo: Sequence[Mapping], fn: Callable[[Mapping], dict],
               key: str = "i", progress_every: int = 25) -> list[dict]:
    """Apply ``fn`` to each row, appending one JSON line per row to ``path``.

    Rows already in the file are skipped, so a restarted kernel resumes where
    it stopped. Returns the records in ``todo`` order. Delete the file when
    the prompt, head set or anything else that changes the answers changes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    done: dict = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done[d[key]] = d
    remaining = [r for r in todo if r[key] not in done]
    print(f"{path.name}: {len(todo) - len(remaining)} cached, {len(remaining)} to run", flush=True)
    t0 = time.time()
    with path.open("a") as f:
        for n_new, r in enumerate(remaining, 1):
            d = {key: r[key], **fn(r)}
            f.write(json.dumps(d) + "\n")
            f.flush()
            done[r[key]] = d
            if progress_every and n_new % progress_every == 0:
                rate = (time.time() - t0) / n_new
                print(f"  {len(todo) - len(remaining) + n_new}/{len(todo)}  {rate:.1f} s/item  "
                      f"~{rate * (len(remaining) - n_new) / 60:.0f} min left", flush=True)
    return [done[r[key]] for r in todo]
