"""CPU pieces of the layer replay (D78 (3)); the GPU checks live in the notebook."""

from circuit_overlap.replay import replay_groups, start_layer


def test_start_layer_is_the_earliest_ablated_layer():
    assert start_layer([(20, 3), (15, 7), (25, 2)]) == 15
    assert start_layer([]) is None


def test_replay_groups_batch_by_start_layer_and_keep_order():
    configs = {"none": [], "a": [(20, 1)], "b": [(15, 0), (22, 1)], "c": [(20, 5), (24, 0)],
               "d": [(15, 2)], "e": [(20, 0)]}
    groups = replay_groups(["none", "a", "b", "c", "d", "e"], configs, batch_size=2, floor=15)
    assert groups == [(15, ["none", "b"]), (15, ["d"]), (20, ["a", "c"]), (20, ["e"])]


def test_replay_groups_never_start_after_a_rows_ablated_layer():
    configs = {f"x{k}": [(15 + k % 11, 0), (25, 1)] for k in range(40)}
    for start, chunk in replay_groups(list(configs), configs, batch_size=8, floor=15):
        assert all(start_layer(configs[n]) == start for n in chunk)
    names = [n for _, chunk in replay_groups(list(configs), configs, 8, 15) for n in chunk]
    assert sorted(names) == sorted(configs)             # every config exactly once


def test_replay_groups_on_the_real_paligemma_plan():
    """Every configuration of D78's plan lands in exactly one batch, at its own start layer."""
    import json
    from pathlib import Path
    from circuit_overlap.cross_ablation import build_plan

    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / "results" / "pg_replication_patching_paligemma2_3b.json").read_text())["result"]
    t = json.loads((root / "results" / "pg_t_patching_paligemma2_3b.json").read_text())["result"]
    plan = build_plan(p, t, n_heads_per_layer=8)
    configs = {k: [tuple(h) for h in v] for k, v in plan["configs"].items()}
    first = min(s for s in (start_layer(h) for h in configs.values()) if s is not None)
    for cond in ("P", "T"):
        names = [n for n, _ in plan["zero"][cond]]
        groups = replay_groups(names, configs, 8, first)
        flat = [n for _, chunk in groups for n in chunk]
        assert sorted(flat) == sorted(names)
        for start, chunk in groups:
            for n in chunk:
                assert (start_layer(configs[n]) or first) == start
