"""Split-half ceiling (plan 3.5, D69). CPU only, synthetic patching records."""

import json
from pathlib import Path

import numpy as np
import pytest

from circuit_overlap.ceiling import rankdata, spearman, spearman_brown, split_half_ceiling
from circuit_overlap.patching import head_restoration_scores

ROOT = Path(__file__).resolve().parents[1]


def make_record(p2v, v2p):
    """A patching record whose restoration scores come out exactly as given.

    Baselines: Delta_prior = 1, Delta_visual = 0, so the denominator is 1 and
    P2V = p2v_diff, V2P = 1 - v2p_diff.
    """
    p2v, v2p = np.asarray(p2v, float), np.asarray(v2p, float)
    return {
        "prior_token_logit_prior_run": 1.0, "visual_token_logit_prior_run": 0.0,
        "prior_token_logit_visual_run": 0.0, "visual_token_logit_visual_run": 0.0,
        "p2v_logit_prior": p2v.tolist(), "p2v_logit_visual": np.zeros_like(p2v).tolist(),
        "v2p_logit_prior": (1.0 - v2p).tolist(), "v2p_logit_visual": np.zeros_like(v2p).tolist(),
    }


CIRCUIT = [(4, 1), (4, 6), (5, 2), (6, 7)]


def planted(n=40, noise=0.02, seed=0, shape=(8, 8), background=None):
    """Records where CIRCUIT restores strongly on top of a per-head background.

    background: fixed per-head true effects shared by every record (graded
    small effects, as real heads have); None means the background is pure
    noise, i.e. every non-circuit head has no consistent effect at all.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros(shape) if background is None else np.asarray(background, float)
    out = []
    for _ in range(n):
        p2v = base + rng.normal(0, noise, shape)
        v2p = base + rng.normal(0, noise, shape)
        for l, h in CIRCUIT:
            p2v[l, h] += 0.6
            v2p[l, h] += 0.6
        out.append(make_record(p2v, v2p))
    return out


def test_synthetic_record_reproduces_the_intended_scores():
    p2v = np.full((2, 3), 0.25)
    v2p = np.full((2, 3), 0.75)
    got_p2v, got_v2p = head_restoration_scores(make_record(p2v, v2p))
    np.testing.assert_allclose(got_p2v, p2v)
    np.testing.assert_allclose(got_v2p, v2p)


# --- rank statistics -------------------------------------------------------


def test_rankdata_averages_ties():
    np.testing.assert_allclose(rankdata([10, 20, 20, 30]), [1, 2.5, 2.5, 4])


def test_spearman_known_values():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # monotone but nonlinear: rank correlation is still perfect
    assert spearman([1, 2, 3, 4], [1, 8, 27, 64]) == pytest.approx(1.0)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_spearman_brown():
    assert spearman_brown(0.5) == pytest.approx(2 / 3)
    assert spearman_brown(1.0) == pytest.approx(1.0)
    assert spearman_brown(0.0) == pytest.approx(0.0)
    assert np.isnan(spearman_brown(float("nan")))


# --- the ceiling -----------------------------------------------------------


def test_a_clear_circuit_agrees_with_itself():
    out = split_half_ceiling(planted(), n_splits=50, seed=0, n_perm=500)
    assert sorted(out["full_sample"]["promoting"]) == sorted(f"L{l}H{h}" for l, h in CIRCUIT)
    assert out["jaccard_promoting"]["median"] == pytest.approx(1.0)
    assert out["share_of_splits_with_identical_promoting_sets"] > 0.9
    for l, h in CIRCUIT:
        assert out["selection_frequency_promoting"][f"L{l}H{h}"] == pytest.approx(1.0)


def test_all_head_rank_correlation_is_swamped_by_heads_that_do_nothing():
    """A documented property, not a bug: with a perfect circuit but a pure
    noise background, the all-head Spearman is low, because the many inactive
    heads reshuffle randomly between halves. With a graded, consistent
    background (each head a small fixed effect) it is high. Which regime real
    heads are in decides whether the rank correlation is a usable statistic
    (D70)."""
    noise_bg = split_half_ceiling(planted(), n_splits=30, seed=0, n_perm=200)
    graded = np.linspace(-0.1, 0.1, 64).reshape(8, 8)
    graded_bg = split_half_ceiling(planted(background=graded), n_splits=30, seed=0,
                                   n_perm=200)
    assert noise_bg["jaccard_promoting"]["median"] == pytest.approx(1.0)
    assert noise_bg["pc1_spearman"]["median"] < 0.3
    assert graded_bg["pc1_spearman"]["median"] > 0.9


def test_pure_noise_does_not():
    """No circuit: the halves disagree, and the ceiling sits near chance.
    This is the case where a low J_PT would be meaningless."""
    rng = np.random.default_rng(1)
    records = [make_record(rng.normal(0, 1, (8, 8)), rng.normal(0, 1, (8, 8)))
               for _ in range(40)]
    out = split_half_ceiling(records, n_splits=50, seed=0, n_perm=500)
    assert out["jaccard_promoting"]["median"] < 0.5
    assert abs(out["pc1_spearman"]["median"]) < 0.3


def test_ceiling_falls_as_the_signal_weakens():
    strong = split_half_ceiling(planted(noise=0.02), n_splits=40, seed=0, n_perm=200)
    weak = split_half_ceiling(planted(noise=2.5), n_splits=40, seed=0, n_perm=200)
    assert weak["jaccard_promoting"]["median"] < strong["jaccard_promoting"]["median"]


def test_seeded_and_reproducible():
    recs = planted(n=20)
    a = split_half_ceiling(recs, n_splits=10, seed=3, n_perm=100)
    b = split_half_ceiling(recs, n_splits=10, seed=3, n_perm=100)
    c = split_half_ceiling(planted(n=20, noise=0.3), n_splits=10, seed=4, n_perm=100)
    assert a["per_split"] == b["per_split"]
    assert a["per_split"] != c["per_split"]


def test_odd_count_and_degenerate_records():
    recs = planted(n=21)
    flat = make_record(np.zeros((8, 8)), np.zeros((8, 8)))
    flat["prior_token_logit_prior_run"] = 0.0     # denominator 0: no contrast
    out = split_half_ceiling(recs + [flat], n_splits=5, seed=0, n_perm=100)
    assert out["n_objects"] == 21
    assert out["n_dropped_degenerate"] == 1
    assert out["half_sizes"] == [10, 11]


def test_too_few_records_raise():
    with pytest.raises(ValueError, match="at least 4"):
        split_half_ceiling(planted(n=3), n_splits=5)


def test_output_is_json_serialisable():
    json.dumps(split_half_ceiling(planted(n=10), n_splits=3, n_perm=50))


def test_runs_on_the_committed_p_patching_results():
    """The real input path: guards the field names against drift."""
    path = ROOT / "results" / "replication_patching_qwen25vl3b.json"
    if not path.exists():
        pytest.skip("P patching results not present")
    records = json.loads(path.read_text())["result"]["records"]
    out = split_half_ceiling(records, n_splits=3, seed=0, n_perm=100)
    assert out["n_objects"] == 73
    assert out["n_heads_total"] == 576
    assert len(out["full_sample"]["promoting"]) == 9
