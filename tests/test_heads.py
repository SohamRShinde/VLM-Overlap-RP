"""Head-set configs are data the whole study depends on: validate them."""

import json
from pathlib import Path

import pytest

from circuit_overlap.heads import (
    format_head_label,
    load_head_set,
    parse_head_label,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("heads_lietzow_*.json"))


def test_configs_exist():
    assert CONFIGS, "no configs/heads_lietzow_*.json found"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_config_loads_and_validates(path):
    hs = load_head_set(path)
    assert hs.promoting and hs.checkpoint
    assert not set(hs.promoting) & set(hs.suppressing)
    assert hs.source, "a head list must record where it was transcribed from"


def test_qwen3b_matches_the_papers_reported_counts():
    hs = load_head_set(ROOT / "configs" / "heads_lietzow_qwen3b.json")
    assert len(hs.promoting) == 9 and len(hs.suppressing) == 6
    assert (26, 5) in hs.promoting and (31, 3) in hs.promoting
    # Lietzow: classified heads sit in the second half of the network.
    assert hs.layer_range()[0] >= 26


def test_single_head_entries_are_promoting_heads():
    for path in CONFIGS:
        hs = load_head_set(path)
        for label in hs.single_head_prior_flip_pct:
            assert parse_head_label(label) in hs.promoting, (path.name, label)


def test_label_round_trip():
    assert parse_head_label("L26H5") == (26, 5)
    assert format_head_label((26, 5)) == "L26H5"
    assert parse_head_label(format_head_label((34, 14))) == (34, 14)
    for bad in ("26H5", "LxH5", "L26", ""):
        with pytest.raises(ValueError):
            parse_head_label(bad)


def test_bounds_check_catches_a_one_based_index_mistake(tmp_path):
    p = tmp_path / "heads_x.json"
    p.write_text(json.dumps({"promoting": [[36, 0]], "suppressing": [],
                             "counts_reported": {"promoting": 1, "suppressing": 0}}))
    load_head_set(p)                                    # no bounds given: fine
    with pytest.raises(ValueError, match="1-based"):
        load_head_set(p, n_layers=36)                   # layer 36 of 36 -> invalid


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"promoting": [[1, 1], [1, 1]], "suppressing": []}, "duplicate"),
        ({"promoting": [[1, 1]], "suppressing": [[1, 1]]}, "both promoting"),
        ({"promoting": [[-1, 0]], "suppressing": []}, "negative"),
        ({"promoting": [[1]], "suppressing": []}, "not \\[layer, head\\]"),
        ({"promoting": [[1, 1]], "suppressing": [],
          "counts_reported": {"promoting": 9}}, "transcription error"),
        ({"promoting": [[1, 1]], "suppressing": [],
          "single_head_prior_flip_pct": {"L2H2": 10.0}}, "not in the promoting set"),
    ],
)
def test_malformed_configs_are_rejected(tmp_path, payload, match):
    p = tmp_path / "heads_bad.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        load_head_set(p)
