"""Patching scores, head classification, and the cached last-token path.

The numpy tests always run. The model tests build a tiny random-init
Qwen2.5-VL and are skipped where torch/transformers are not installed (the
repo's .venv deliberately has neither); run them with an interpreter that has
both, e.g. ``python -m pytest tests/test_patching.py``.
"""

import math

import numpy as np
import pytest

from circuit_overlap.patching import (
    classify_heads,
    head_restoration_scores,
    mean_restoration_grids,
    rank_heads,
)


def record(prior_run=(5.0, 1.0), visual_run=(1.0, 4.0), p2v=None, v2p=None, shape=(2, 3)):
    """prior_run / visual_run = (logit of prior token, logit of visual token)."""
    full = lambda v: [[v] * shape[1] for _ in range(shape[0])]  # noqa: E731
    return {
        "prior_token_logit_prior_run": prior_run[0], "visual_token_logit_prior_run": prior_run[1],
        "prior_token_logit_visual_run": visual_run[0], "visual_token_logit_visual_run": visual_run[1],
        "p2v_logit_prior": p2v[0] if p2v else full(visual_run[0]),
        "p2v_logit_visual": p2v[1] if p2v else full(visual_run[1]),
        "v2p_logit_prior": v2p[0] if v2p else full(prior_run[0]),
        "v2p_logit_visual": v2p[1] if v2p else full(prior_run[1]),
    }


def test_restoration_scores_hand_checked():
    # Delta_prior_run = 5 - 1 = 4, Delta_visual_run = 1 - 4 = -3, gap 7.
    r = record(p2v=([[1.0, 4.5, 2.5]] * 2, [[4.0, 1.0, 2.5]] * 2),
               v2p=([[5.0, 1.0, 1.5]] * 2, [[1.0, 4.0, 1.5]] * 2))
    p2v, v2p = head_restoration_scores(r)
    # P2V: Delta_patched -3 -> 0, +3.5 -> 1 (fully prior), 0 -> 3/7
    assert np.allclose(p2v[0], [0.0, 6.5 / 7, 3 / 7])
    # V2P: Delta_patched 4 -> 0, -3 -> 1 (fully visual), 0 -> 4/7
    assert np.allclose(v2p[0], [0.0, 1.0, 4 / 7])


def test_unpatched_runs_score_zero_and_degenerate_examples_are_dropped():
    p2v, v2p = head_restoration_scores(record())
    assert np.allclose(p2v, 0) and np.allclose(v2p, 0)
    assert head_restoration_scores(record(prior_run=(2.0, 1.0), visual_run=(3.0, 2.0))) is None
    good = record(p2v=([[4.0] * 3] * 2, [[1.0] * 3] * 2))     # Delta_patched = 3 -> 6/7
    p2v_mean, _, n = mean_restoration_grids(
        [good, record(prior_run=(2.0, 1.0), visual_run=(3.0, 2.0))])
    assert n == 1 and np.allclose(p2v_mean, 6 / 7)
    with pytest.raises(ValueError):
        mean_restoration_grids([])


def test_classify_heads_finds_planted_heads_and_orients_pc1():
    rng = np.random.default_rng(0)
    p2v = rng.normal(0, 0.01, (36, 16))
    v2p = rng.normal(0, 0.01, (36, 16))
    for layer, head in [(26, 5), (31, 3), (27, 1)]:
        p2v[layer, head] = v2p[layer, head] = 0.4
    for layer, head in [(30, 3), (34, 10)]:
        p2v[layer, head] = v2p[layer, head] = -0.3
    c = classify_heads(p2v, v2p)
    assert sorted(c["promoting"]) == [(26, 5), (27, 1), (31, 3)]
    assert sorted(c["suppressing"]) == [(30, 3), (34, 10)]
    assert sum(c["pc1_direction"]) > 0
    ranks = rank_heads(c["pc1_scores"])
    assert {ranks[h] for h in [(26, 5), (31, 3), (27, 1)]} == {1, 2, 3}
    assert ranks[(34, 10)] in (575, 576)


def test_classify_heads_matches_an_independent_svd():
    rng = np.random.default_rng(1)
    p2v, v2p = rng.normal(size=(6, 4)), rng.normal(size=(6, 4)) * 0.5
    c = classify_heads(p2v, v2p, sigma=1.0)
    points = np.column_stack([p2v.ravel(), v2p.ravel()])
    centred = points - points.mean(0)
    direction = np.linalg.svd(centred, full_matrices=False)[2][0]
    direction = direction if direction.sum() >= 0 else -direction
    assert np.allclose(c["pc1_scores"].ravel(), centred @ direction)
    s = centred @ direction
    expected = {(i // 4, i % 4) for i in np.flatnonzero(s > s.mean() + s.std())}
    assert set(c["promoting"]) == expected


def test_classify_heads_rejects_bad_grids():
    with pytest.raises(ValueError):
        classify_heads(np.zeros((2, 3)), np.zeros((3, 2)))
    bad = np.zeros((2, 3)); bad[0, 0] = math.nan
    with pytest.raises(ValueError):
        classify_heads(bad, np.zeros((2, 3)))


# --- the cached last-token path on a tiny real Qwen2.5-VL ----------------

IMAGE_TOKEN, VIDEO_TOKEN, VISION_START, VISION_END = 151, 152, 153, 154


def tiny_qwen():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=160, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
            rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        ),
        vision_config=dict(
            depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
            patch_size=14, spatial_merge_size=2, temporal_patch_size=2, window_size=56,
            fullatt_block_indexes=[1],
        ),
        image_token_id=IMAGE_TOKEN, video_token_id=VIDEO_TOKEN,
        vision_start_token_id=VISION_START, vision_end_token_id=VISION_END,
    )
    model = transformers.Qwen2_5_VLForConditionalGeneration(config).eval()
    return torch, model


def tiny_inputs(torch, text_tail):
    grid = torch.tensor([[1, 4, 4]])                       # 16 patches -> 4 tokens
    ids = [5, 6, VISION_START] + [IMAGE_TOKEN] * 4 + [VISION_END] + list(text_tail)
    ids = torch.tensor([ids])
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        # The real processor sends this (1 = image token). Without it the model
        # skips multimodal RoPE and a broken cached path can still pass.
        "mm_token_type_ids": (ids == IMAGE_TOKEN).long(),
        "pixel_values": torch.randn(16, 3 * 2 * 14 * 14, generator=torch.Generator().manual_seed(1)),
        "image_grid_thw": grid,
    }


def test_cached_last_token_path_matches_full_forward():
    torch, model = tiny_qwen()
    from circuit_overlap.patching import LastTokenRunner

    run = LastTokenRunner(model, tiny_inputs(torch, [20, 21, 22, 23]))
    # The multimodal-RoPE path must be the one exercised: it is the one that
    # broke on Kaggle (positions spanning the whole prompt for a 1-token step).
    assert run.position_ids is not None and run.position_ids.shape[-1] == 1
    assert run.check_against_full_forward(atol=1e-4) < 1e-4
    # Repeated calls must not grow or corrupt the cached prefix.
    assert torch.allclose(run.logits(), run.logits())


def test_patched_head_matches_full_forward_with_the_same_patch():
    torch, model = tiny_qwen()
    from circuit_overlap.patching import (
        LastTokenRunner, capture_last_token_o_proj_inputs, patch_head_last_token,
    )

    source_inputs = tiny_inputs(torch, [30, 31, 32, 33])
    target_inputs = tiny_inputs(torch, [20, 21, 22, 23])
    source, target = LastTokenRunner(model, source_inputs), LastTokenRunner(model, target_inputs)
    with capture_last_token_o_proj_inputs(model) as acts:
        source.logits()
    with torch.no_grad():
        full_source = None
        with capture_last_token_o_proj_inputs(model) as acts_full:
            model(**source_inputs, use_cache=False)
    for layer in acts:                                     # capture is path-independent
        assert torch.allclose(acts[layer], acts_full[layer], atol=1e-5)

    head_dim, layer, head = 16, 2, 1
    value = acts[layer][head * head_dim:(head + 1) * head_dim]
    with patch_head_last_token(model, layer, head, head_dim, value):
        fast = target.logits()
        with torch.no_grad():
            full = model(**target_inputs, use_cache=False).logits[0, -1].float()
    assert torch.allclose(fast, full, atol=1e-4)
    assert not torch.allclose(fast, target.logits(), atol=1e-6)   # the patch did something
    del full_source


def test_patch_example_record_scores_like_lietzow():
    torch, model = tiny_qwen()
    from circuit_overlap.patching import LastTokenRunner, patch_example

    source = LastTokenRunner(model, tiny_inputs(torch, [30, 31, 32, 33]))
    target = LastTokenRunner(model, tiny_inputs(torch, [20, 21, 22, 23]))
    rec = patch_example(model, source, target, head_dim=16)
    if rec is None:
        pytest.skip("random model gave the same argmax for both prompts")
    assert np.array(rec["p2v_logit_prior"]).shape == (4, 4)
    p2v, v2p = head_restoration_scores(rec)
    assert np.isfinite(p2v).all() and np.isfinite(v2p).all()
    # Patching every head of the last layer at once would restore fully; one
    # head alone restores only part, so scores stay well inside a sane range.
    assert np.abs(p2v).max() < 50 and np.abs(v2p).max() < 50
