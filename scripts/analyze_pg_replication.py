"""PaliGemma-3B replication gate and T viability, read by D76. Written before any run.

    .venv/bin/python scripts/analyze_pg_replication.py \\
        results/pg_selection_paligemma2_3b.json \\
        results/pg_replication_knockout_paligemma2_3b.json \\
        results/pg_replication_patching_paligemma2_3b.json

CPU only. The gate goes through the same code as Qwen's D40
(``analyze_replication_gate.main``) with D76's thresholds; the selection
counts are recomputed from the per-object records and the stored colour
tokens, never read from a summary. Missing files are reported as not run.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_replication_gate import main as gate_main  # noqa: E402
from circuit_overlap.heads import load_head_set, parse_head_label  # noqa: E402
from circuit_overlap.replication import is_correctly_conflicting  # noqa: E402
from circuit_overlap.t_patching import is_t_conflicting  # noqa: E402

# D76 (DECISIONS.md), pre-registered 2026-11-05. Change only through DECISIONS.md.
# (f)'s two heads are the ones Lietzow report single-head flips for, read from
# the config rather than typed (CLAUDE.md: heads are data).
_HEADS = load_head_set(ROOT / "configs" / "heads_lietzow_paligemma2_3b.json")
D76 = {"label": "D76", "min_prior_flip": 2 / 3, "max_visual_flip": 0.10, "min_found": 5,
       "max_perm_p": 0.001, "max_median_rank": 10, "top_k_single": 3,
       "single_heads": tuple(parse_head_label(k) for k in _HEADS.single_head_prior_flip_pct),
       "n_perm": 10_000}
MIN_T_CONFLICTING = 50   # D76 (6), D41's bar
MIN_PATCHED = 60         # D76a (2): fewer patched objects -> (c)-(e) not run


def selection_counts(path: Path) -> dict:
    """P and T conflicting counts, recomputed from the records (D76 (4), (6))."""
    payload = json.loads(path.read_text())
    res = payload["result"]
    color_ids = {k: v for k, v in res["color_ids"].items()}
    ambiguous = set(res["ambiguous_colours"])
    status = Counter(r["status"] for r in res["records"])
    ok = [r for r in res["records"] if r["status"] == "ok"]
    p = [r for r in ok if is_correctly_conflicting(r, r["top1_visual"], r["top1_prior"], color_ids, ambiguous)]
    t = [r for r in ok if is_t_conflicting(r, r["top1_neutral"], r["top1_incongruent"], color_ids, ambiguous)]
    stored_p = sum(bool(r.get("p_conflicting")) for r in ok)
    stored_t = sum(bool(r.get("t_conflicting")) for r in ok)
    print(f"selection  {path}  (commit {payload['commit']})")
    print(f"  {len(res['records'])} objects, status {dict(status)}; ambiguous colours (D26): {sorted(ambiguous)}")
    print(f"  P correctly conflicting: {len(p)}  (Lietzow report 121 for PaliGemma 3B; reported, not judged)")
    print(f"  T conflicting (on_object_small): {len(t)}  -> T viability (>= {MIN_T_CONFLICTING}): "
          f"{'pass' if len(t) >= MIN_T_CONFLICTING else 'FAIL'}")
    print(f"  P and T both: {len({r['i'] for r in p} & {r['i'] for r in t})}")
    if (stored_p, stored_t) != (len(p), len(t)):
        print(f"  WARNING: notebook flags give P {stored_p}, T {stored_t}; recomputed values are used")
    verify = res.get("verify", {})
    if verify:
        print("  checks:", json.dumps(verify, default=str)[:600])
    return {"n_p": len(p), "n_t": len(t), "t_viable": len(t) >= MIN_T_CONFLICTING,
            "p_ids": [r["i"] for r in p], "t_ids": [r["i"] for r in t]}


def main(selection: str | None, knockout: str | None, patching: str | None) -> dict:
    out: dict = {}
    if selection and Path(selection).exists():
        out["selection"] = selection_counts(Path(selection))
        print()
    elif selection:
        print(f"missing: {selection}\n")
    if patching and Path(patching).exists():
        n = len(json.loads(Path(patching).read_text())["result"]["records"])
        print(f"patching: {n} objects patched (D76a minimum {MIN_PATCHED})")
        if n < MIN_PATCHED:
            print(f"  fewer than {MIN_PATCHED}: criteria (c)-(e) are not run\n")
            patching = None
    out["gate"] = gate_main(knockout, patching, gate=D76)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
