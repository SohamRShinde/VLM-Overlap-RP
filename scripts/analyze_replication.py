"""Analyse a saved replication flip-rate result file.

Reads only from ``results/`` (CLAUDE.md: analysis scripts never recompute or
invent numbers) and prints every figure that goes into docs/RESULTS.md.

    .venv/bin/python scripts/analyze_replication.py results/replication_flip_rate_qwen25vl3b.json

What it reports, and why each one is there:

* **Prior and visual flip rates** with Wilson 95% intervals. n is ~73, so one
  object moves a rate by ~1.4 points and the interval is wide.
* **Comparison with Lietzow.** Their rates are also on ~73 examples, so the
  right comparison is two-sample (Fisher's exact test on the flip counts),
  not "does our interval contain their point estimate". Their counts are
  reconstructed from the published percentages and n, so they are
  approximate.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_exact_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for k1/n1 vs k2/n2."""
    total, flips = n1 + n2, k1 + k2
    denom = math.comb(total, flips)

    def prob(k: int) -> float:
        return math.comb(n1, k) * math.comb(n2, flips - k) / denom

    observed = prob(k1)
    lo, hi = max(0, flips - n2), min(n1, flips)
    return min(1.0, sum(p for k in range(lo, hi + 1)
                        if (p := prob(k)) <= observed * (1 + 1e-9)))


def main(path: str) -> None:
    payload = json.loads(Path(path).read_text())
    result = payload["result"]
    reported = result["reported"]
    n = result["n_correctly_conflicting"]
    n_ref = reported["n"]

    print(f"file      {path}")
    print(f"commit    {payload['commit']}   model {payload['model']}   "
          f"{payload.get('quantisation', '?')}, compute {payload.get('compute_dtype', '?')}")
    if "correction" in payload:
        print(f"CORRECTION  {payload['correction']}")
    print(f"objects   {result['n_objects']} after filters (dropped {result['dropped']}); "
          f"Lietzow {reported.get('n_objects', '?')}")
    print(f"subset    top-1 differs {result['n_top1_differs']}, "
          f"correctly conflicting {n} (Lietzow {n_ref})")
    print(f"ablated   {len(result['heads_ablated'])} promoting heads")
    print()
    for kind in ("prior", "visual"):
        rate = result[f"{kind}_flip"]
        k = round(rate * n)
        lo, hi = wilson_interval(k, n)
        ref_rate = reported[f"{kind}_flip"]
        k_ref = round(ref_rate * n_ref)
        ref_lo, ref_hi = wilson_interval(k_ref, n_ref)
        p = fisher_exact_two_sided(k, n, k_ref, n_ref)
        print(f"{kind:6s} flip  {k}/{n} = {rate:.1%}  95% CI [{lo:.1%}, {hi:.1%}]   "
              f"Lietzow ~{k_ref}/{n_ref} = {ref_rate:.1%}  [{ref_lo:.1%}, {ref_hi:.1%}]   "
              f"Fisher p = {p:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: analyze_replication.py results/<file>.json")
    main(sys.argv[1])
