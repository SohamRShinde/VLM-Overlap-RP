"""CPU unit tests for the metrics in docs/plan_v2.md section 3.

Dummy tensors only: no model, no weights, no network.
"""

import math

import numpy as np
import pytest

from circuit_overlap.metrics import (
    containment,
    flip_rate,
    jaccard,
    jaccard_permutation_test,
    logit_difference,
    permutation_null_jaccard,
    restoration_score,
)

# --------------------------------------------------------------------------
# Delta: logit difference (plan 3.4)
# --------------------------------------------------------------------------


def test_logit_difference_hand_computed():
    logits = np.array([0.5, 2.0, -1.0, 3.25])
    # idx 1 ("red") minus idx 2 ("blue") = 2.0 - (-1.0) = 3.0
    assert logit_difference(logits, 1, 2) == pytest.approx(3.0)


def test_logit_difference_is_batched():
    logits = np.array([[0.0, 1.0, 2.0], [5.0, 1.0, -1.0]])
    out = logit_difference(logits, 0, 2)
    assert out.shape == (2,)
    np.testing.assert_allclose(out, [-2.0, 6.0])


def test_logit_difference_sign_flips_when_token_pair_is_swapped():
    logits = np.array([[0.0, 1.0, 2.0], [5.0, 1.0, -1.0]])
    np.testing.assert_allclose(
        logit_difference(logits, 0, 2), -logit_difference(logits, 2, 0)
    )


def test_logit_difference_rejects_out_of_range_token_id():
    with pytest.raises(IndexError):
        logit_difference(np.zeros(4), 0, 9)


# --------------------------------------------------------------------------
# R: restoration score (plan 3.4)
# --------------------------------------------------------------------------


def test_restoration_score_is_zero_at_target_and_one_at_source():
    assert restoration_score(-2.0, -2.0, 4.0) == pytest.approx(0.0)
    assert restoration_score(4.0, -2.0, 4.0) == pytest.approx(1.0)


def test_restoration_score_is_half_at_the_midpoint():
    assert restoration_score(1.0, -2.0, 4.0) == pytest.approx(0.5)


def test_restoration_score_handles_the_reverse_patching_direction():
    # Source below target: the plan patches in both directions (3.4).
    assert restoration_score(1.0, 4.0, -2.0) == pytest.approx(0.5)
    assert restoration_score(-2.0, 4.0, -2.0) == pytest.approx(1.0)


def test_restoration_score_is_not_clipped_outside_unit_interval():
    assert restoration_score(7.0, -2.0, 4.0) == pytest.approx(1.5)
    assert restoration_score(-5.0, -2.0, 4.0) == pytest.approx(-0.5)


def test_restoration_score_returns_nan_for_a_vanishing_contrast():
    out = restoration_score(0.3, 1.0, 1.0 + 1e-12)
    assert np.isnan(out)


def test_restoration_score_broadcasts_and_isolates_degenerate_examples():
    patched = np.array([1.0, 0.5, 2.0])
    target = np.array([-2.0, 0.5, 0.0])
    source = np.array([4.0, 0.5, 4.0])  # middle example has no contrast
    out = restoration_score(patched, target, source)
    assert out.shape == (3,)
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], [0.5, 0.5])


# --------------------------------------------------------------------------
# Flip rate (plan 3.4)
# --------------------------------------------------------------------------


def test_flip_rate_all_and_none():
    before = np.array(["red", "red", "blue"])
    assert flip_rate(before, np.array(["blue", "blue", "red"])) == pytest.approx(1.0)
    assert flip_rate(before, before) == pytest.approx(0.0)


def test_flip_rate_mixed_is_an_exact_fraction():
    before = np.array([1, 2, 3, 4])
    after = np.array([1, 9, 3, 9])
    assert flip_rate(before, after) == pytest.approx(0.5)


def test_flip_rate_mask_restricts_the_denominator():
    before = np.array([1, 2, 3, 4])
    after = np.array([9, 9, 3, 4])
    # Only the first two examples are "correctly conflicting"; both flip.
    mask = np.array([True, True, False, False])
    assert flip_rate(before, after, mask=mask) == pytest.approx(1.0)
    assert flip_rate(before, after) == pytest.approx(0.5)


def test_flip_rate_of_an_empty_selection_is_nan_not_zero():
    before = np.array([1, 2])
    mask = np.array([False, False])
    assert math.isnan(flip_rate(before, before, mask=mask))


def test_flip_rate_on_signs_of_the_logit_difference():
    delta_before = np.array([2.0, 1.0, -3.0])
    delta_after = np.array([-1.0, 0.5, -4.0])
    out = flip_rate(np.sign(delta_before), np.sign(delta_after))
    assert out == pytest.approx(1 / 3)


def test_flip_rate_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        flip_rate(np.array([1, 2, 3]), np.array([1, 2]))


# --------------------------------------------------------------------------
# J: Jaccard, and containment (plan 3.5)
# --------------------------------------------------------------------------

HEADS_A = [(26, 0), (26, 5), (27, 1)]
HEADS_B = [(26, 5), (27, 1), (31, 3), (34, 14)]


def test_jaccard_disjoint_and_identical():
    assert jaccard([(1, 0)], [(2, 0)]) == pytest.approx(0.0)
    assert jaccard(HEADS_A, list(HEADS_A)) == pytest.approx(1.0)


def test_jaccard_partial_overlap_hand_checked():
    # intersection {(26,5), (27,1)} = 2; union = 5
    assert jaccard(HEADS_A, HEADS_B) == pytest.approx(2 / 5)


def test_jaccard_is_symmetric_and_ignores_order_and_duplicates():
    shuffled = [(27, 1), (26, 5), (26, 0), (26, 0)]
    assert jaccard(HEADS_A, HEADS_B) == pytest.approx(jaccard(HEADS_B, HEADS_A))
    assert jaccard(shuffled, HEADS_B) == pytest.approx(jaccard(HEADS_A, HEADS_B))


def test_jaccard_accepts_sets_and_other_iterables():
    assert jaccard(set(HEADS_A), tuple(HEADS_B)) == pytest.approx(2 / 5)


def test_jaccard_of_two_empty_sets_is_zero_by_convention():
    # Documented choice: no heads found in either condition is not evidence of
    # a shared circuit.
    assert jaccard([], []) == 0.0
    assert jaccard(HEADS_A, []) == 0.0


def test_jaccard_rejects_unhashable_head_ids():
    with pytest.raises(TypeError):
        jaccard([[26, 0]], HEADS_B)


def test_containment_is_asymmetric_and_matches_the_lietzow_style_fraction():
    # 2 of A's 3 heads are in B; 2 of B's 4 heads are in A.
    assert containment(HEADS_A, HEADS_B) == pytest.approx(2 / 3)
    assert containment(HEADS_B, HEADS_A) == pytest.approx(0.5)


def test_containment_of_an_empty_first_set_is_nan():
    assert math.isnan(containment([], HEADS_B))


# --------------------------------------------------------------------------
# Permutation null for J (plan 3.5)
# --------------------------------------------------------------------------


def test_permutation_null_is_reproducible_for_a_fixed_seed():
    kwargs = dict(size_a=9, size_b=9, n_heads=576, n_perm=200, seed=7)
    np.testing.assert_array_equal(
        permutation_null_jaccard(**kwargs), permutation_null_jaccard(**kwargs)
    )


def test_permutation_null_differs_between_seeds():
    a = permutation_null_jaccard(9, 9, 576, n_perm=200, seed=1)
    b = permutation_null_jaccard(9, 9, 576, n_perm=200, seed=2)
    assert not np.array_equal(a, b)


def test_permutation_null_values_are_valid_jaccards():
    null = permutation_null_jaccard(9, 12, 576, n_perm=500, seed=0)
    assert null.shape == (500,)
    assert np.all((null >= 0.0) & (null <= 1.0))


def test_chance_floor_for_two_nine_head_sets_in_576_heads():
    # plan 3.5 quotes J ~ 0.008 for two random 9-head sets in 576 heads.
    # Analytic expectation: E[|A&B|] = 9*9/576 = 0.1406, so J ~ 0.1406/(18-0.1406).
    null = permutation_null_jaccard(9, 9, 576, n_perm=20_000, seed=0)
    assert null.mean() == pytest.approx(0.008, abs=0.004)


def test_permutation_null_is_degenerate_at_the_extremes():
    # Empty set: no overlap is possible.
    assert np.all(permutation_null_jaccard(0, 9, 576, n_perm=50, seed=0) == 0.0)
    # Every head selected on both sides: the two sets must coincide.
    assert np.all(permutation_null_jaccard(4, 4, 4, n_perm=50, seed=0) == 1.0)


def test_permutation_null_rejects_impossible_sizes():
    with pytest.raises(ValueError):
        permutation_null_jaccard(600, 9, 576, n_perm=10, seed=0)
    with pytest.raises(ValueError):
        permutation_null_jaccard(9, 9, 576, n_perm=0, seed=0)


def test_permutation_test_reports_observed_null_and_bookkeeping():
    out = jaccard_permutation_test(HEADS_A, HEADS_B, n_heads=576, n_perm=1000, seed=0)
    assert out["jaccard"] == pytest.approx(2 / 5)
    assert out["n_a"] == 3 and out["n_b"] == 4 and out["n_intersection"] == 2
    assert out["n_heads"] == 576 and out["n_perm"] == 1000
    assert 0.0 <= out["p_value"] <= 1.0
    assert out["null_mean"] < out["jaccard"]
    assert out["normalised_jaccard"] > 1.0


def test_permutation_test_p_value_is_small_for_identical_sets():
    out = jaccard_permutation_test(HEADS_A, HEADS_A, n_heads=576, n_perm=1000, seed=0)
    assert out["jaccard"] == pytest.approx(1.0)
    # Never exactly 0: the (k+1)/(n+1) convention bounds it below.
    assert out["p_value"] == pytest.approx(1 / 1001)


def test_permutation_test_p_value_is_large_for_a_chance_level_overlap():
    # Two 9-head sets sharing nothing: chance already matches or beats this.
    set_a = [(0, h) for h in range(9)]
    set_b = [(1, h) for h in range(9)]
    out = jaccard_permutation_test(set_a, set_b, n_heads=576, n_perm=1000, seed=0)
    assert out["jaccard"] == 0.0
    assert out["p_value"] == pytest.approx(1.0)
