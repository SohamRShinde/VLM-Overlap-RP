"""Real dataset images for Figure 1 (fig_design), written to figures/inputs/.

    .venv/bin/python scripts/fig1_inputs.py

Fetches one Visual-Counterfact row (split="color", the strawberry, row 422:
original red, counterfactual blue) from the Hugging Face dataset viewer and
writes the four inputs the schematic shows:

    original.png        the real-colour image                (V source)
    counterfact.png     the recoloured image                 (P source and target, V target)
    note_attack.png     original + note with the counterfactual colour   (T source)
    note_neutral.png    original + the matched neutral-word note         (T target)

The notes are drawn by the experiment's own code (configs/notes.json layout
"on_object_small", the layout the paper reports; circuit_overlap.notes.render_note, neutral word from
matched_neutral_word), so they are the T stimuli Qwen saw on this object, up to
the font available on this machine. Qwen2.5-VL-3B answered "Red" with the
neutral note and "Blue" with the attack note at this layout
(results/t_screen_checkpoint.jsonl, strawberry_0422).

Only needs re-running if the example object changes; make_figures.py reads the
PNGs and does not touch the network.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuit_overlap.notes import matched_neutral_word, object_bbox_from_pair, render_note  # noqa: E402
from circuit_overlap.t_screen import load_notes_config  # noqa: E402

DATASET, SPLIT, ROW = "mgolov/Visual-Counterfact", "color", 422
LAYOUT = "on_object_small"
SIDE = 360   # px; the tile is ~0.6 in wide in the paper, so this is >500 dpi


def fetch_row() -> dict:
    url = (f"https://datasets-server.huggingface.co/rows?dataset={DATASET}&config=default"
           f"&split={SPLIT}&offset={ROW}&length=1")
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["rows"][0]["row"]


def fetch_image(src: str) -> Image.Image:
    with urllib.request.urlopen(src, timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def main(out: Path = ROOT / "figures" / "inputs") -> None:
    row = fetch_row()
    if row["object"] != "strawberry":
        raise SystemExit(f"row {ROW} is {row['object']!r}, not the strawberry; the dataset changed")
    original = fetch_image(row["original_image"]["src"])
    counterfact = fetch_image(row["counterfact_image"]["src"])
    cf_color = row["incorrect_answer"].strip().lower()

    cfg = load_notes_config(ROOT / "configs" / "notes.json")
    spec = next(s for s in cfg["specs"] if s.name == LAYOUT)
    neutral = matched_neutral_word(cf_color, cfg["neutral_words"])
    bbox = object_bbox_from_pair(original, counterfact)

    images = {
        "original": original,
        "counterfact": counterfact,
        "note_attack": render_note(original, cf_color, spec, bbox=bbox).image,
        "note_neutral": render_note(original, neutral, spec, bbox=bbox).image,
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, img in images.items():
        img.resize((SIDE, SIDE), Image.LANCZOS).save(out / f"{name}.png", optimize=True)
    meta = {"dataset": DATASET, "split": SPLIT, "row": ROW, "object": row["object"],
            "original_colors": row["correct_answer"], "counterfactual_color": cf_color,
            "layout": LAYOUT, "neutral_word": neutral}
    (out / "inputs.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
