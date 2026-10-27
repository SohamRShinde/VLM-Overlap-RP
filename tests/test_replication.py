"""Row filters, the correctly-conflicting test, pairwise flips, and resume."""

import json

from circuit_overlap.replication import (
    is_correctly_conflicting,
    knockout_flip_counts,
    lietzow_rows,
    run_cached,
)


def test_lietzow_rows_applies_the_three_filters():
    data = [
        {"object": "apple", "correct_answer": "['red', 'green']", "incorrect_answer": "['blue']"},
        {"object": "apple (fruit)", "correct_answer": "['red']", "incorrect_answer": "['blue']"},
        {"object": "cow", "correct_answer": "black", "incorrect_answer": "['blue']"},   # not a list
        {"object": "radish", "correct_answer": "['red', 'white']", "incorrect_answer": "['white']"},
    ]
    rows, dropped = lietzow_rows(data, inflect_engine=None)
    assert [r["name"] for r in rows] == ["apple"] and rows[0]["i"] == 0
    assert rows[0]["correct"] == ["green", "red"] and rows[0]["attack"] == "blue"
    assert dropped == {"name": 1, "colours": 1, "overlap": 1}


def test_is_correctly_conflicting():
    ids = {"red": [1, 2], "green": [3], "blue": [4, 5], "teal": [6]}
    row = {"correct": ["red", "green"], "attack": "blue"}
    assert is_correctly_conflicting(row, top1_visual=5, top1_prior=3, color_ids=ids)
    assert not is_correctly_conflicting(row, top1_visual=5, top1_prior=5, color_ids=ids)
    assert not is_correctly_conflicting(row, top1_visual=1, top1_prior=1, color_ids=ids)
    assert not is_correctly_conflicting(row, 5, 3, ids, ambiguous=["green"])


def test_knockout_flip_counts_is_pairwise():
    recs = [
        # prior run stays prior, visual run stays visual: no flips
        {"grp/prior": [5.0, 1.0], "grp/visual": [1.0, 5.0]},
        # prior run now prefers the visual token: prior flip
        {"grp/prior": [1.0, 2.0], "grp/visual": [1.0, 5.0]},
        # visual run ties: counts as no longer visual-grounded (their `>`)
        {"grp/prior": [5.0, 1.0], "grp/visual": [3.0, 3.0]},
    ]
    c = knockout_flip_counts(recs, "grp")
    assert (c["n"], c["prior_flips"], c["visual_flips"]) == (3, 1, 1)
    assert abs(c["prior_flip_rate"] - 1 / 3) < 1e-12


def test_run_cached_resumes(tmp_path):
    path = tmp_path / "out" / "x.jsonl"
    rows = [{"i": k} for k in range(4)]
    calls = []

    def fn(r):
        calls.append(r["i"])
        if r["i"] == 2 and calls.count(2) == 1:
            raise KeyboardInterrupt
        return {"v": r["i"] * 10}

    try:
        run_cached(path, rows, fn)
    except KeyboardInterrupt:
        pass
    out = run_cached(path, rows, fn)
    assert calls == [0, 1, 2, 2, 3]
    assert [o["v"] for o in out] == [0, 10, 20, 30]
    assert len(path.read_text().splitlines()) == 4
    assert json.loads(path.read_text().splitlines()[0]) == {"i": 0, "v": 0}
