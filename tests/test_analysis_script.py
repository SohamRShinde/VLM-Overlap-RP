"""The analysis script is how numbers reach RESULTS.md, so it must not rot."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from circuit_overlap.screen import ObjectRecord

ROOT = Path(__file__).resolve().parents[1]


def make_result_file(path: Path) -> None:
    records = []
    for i in range(8):
        for layout in ("on_object_small", "on_object_medium", "beside_object"):
            fooled = layout.startswith("on_object") and i % 2 == 0
            records.append(ObjectRecord(
                object_id=f"obj{i}", object_name=f"obj{i}",
                original_colors=("red",), counterfactual_color="blue",
                answer_neutral="Red.",
                answer_incongruent="Blue." if fooled else "Red.",
                answer_congruent="Red.",
                delta_neutral=8.0, delta_incongruent=-1.0 if fooled else 2.0,
                delta_congruent=9.5, layout=layout, reference_color="red",
                color_logits_neutral={"red": 6.0, "blue": -2.0},
                color_logits_incongruent={"red": 1.0, "blue": 2.0},
                color_logits_congruent={"red": 7.0, "blue": -2.5},
                note_area_frac=0.05, object_occlusion=0.09,
            ))
    # one half-precision casualty (plan risk 7)
    records.append(ObjectRecord(
        object_id="bench", original_colors=("brown",), counterfactual_color="pink",
        answer_neutral="!!!!", answer_incongruent="!!!!", layout="beside_object",
        delta_neutral=float("nan"), delta_incongruent=float("nan"),
    ))
    path.write_text(json.dumps({
        "model": "test", "checkpoint": "test", "commit": "abc1234",
        "timestamp_utc": "2026-10-20T00:00:00+00:00",
        "result": {"records": [asdict(r) for r in records], "summary": {}},
    }))


def test_script_runs_and_reports_the_key_quantities(tmp_path):
    path = tmp_path / "res.json"
    make_result_file(path)
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analyze_t_screen.py"), str(path)],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout
    assert "NUMERICS FAILURES" in out and "bench" in out
    assert "attack shift" in out and "congruent shift" in out
    assert "paired within object" in out
    import re

    assert re.search(r"n_objects_numerics_failed\s+1\b", out)
    # The fabricated data has on-object fooling and beside-object not, so the
    # paired comparison must come out one-sided.
    assert "on-object only 4" in out
