# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm import SamplingParams
from vllm.dynamic_sampling import (DYNAMIC_SAMPLING_EPS,
                                   GREEDY_TEMPERATURE,
                                   compute_dynamic_temperature,
                                   get_dynamic_sampling_config)


def test_confidence_gated_temperature_switches_to_greedy():
    logits = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    config = get_dynamic_sampling_config({
        "dynamic_sampling_policy": "confidence_gated",
        "dynamic_sampling_kwargs": {
            "T_high": 0.7,
            "maxprob_threshold": 0.9,
        },
    })
    assert config is not None

    temps = compute_dynamic_temperature(logits, config)

    assert temps[0].item() == pytest.approx(GREEDY_TEMPERATURE)
    assert temps[1].item() == pytest.approx(0.7)


def test_entropy_continuous_respects_bounds_and_entropy_ordering():
    logits = torch.tensor([
        [8.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    config = get_dynamic_sampling_config({
        "dynamic_sampling_policy": "entropy_continuous",
        "dynamic_sampling_kwargs": {
            "T_min": 0.2,
            "T_max": 0.9,
        },
    })
    assert config is not None

    temps = compute_dynamic_temperature(logits, config)

    assert torch.all(temps >= 0.2)
    assert torch.all(temps <= 0.9)
    assert temps[0].item() < temps[1].item()


def test_edt_uses_raw_entropy_formula():
    logits = torch.tensor([
        [8.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    config = get_dynamic_sampling_config({
        "dynamic_sampling_policy": "edt",
        "dynamic_sampling_kwargs": {
            "T0": 0.7,
            "theta": 0.5,
            "N": 0.8,
        },
    })
    assert config is not None

    temps = compute_dynamic_temperature(logits, config)

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    safe_entropy = torch.clamp_min(entropy, DYNAMIC_SAMPLING_EPS)
    expected = 0.7 * torch.exp(math.log(0.8) * (0.5 / safe_entropy))
    expected = torch.clamp(expected, min=0.0, max=0.7)
    expected = torch.where(
        expected < DYNAMIC_SAMPLING_EPS,
        torch.full_like(expected, GREEDY_TEMPERATURE),
        expected,
    )

    assert temps.tolist() == pytest.approx(expected.tolist())
    assert temps[0].item() < temps[1].item() < temps[2].item()


def test_entropy_shift_tracks_entropy_around_anchor_temperature():
    logits = torch.tensor([
        [8.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    config = get_dynamic_sampling_config({
        "dynamic_sampling_policy": "entropy_shift",
        "dynamic_sampling_kwargs": {
            "T_base": 0.8,
            "delta": 0.2,
            "H_mean": 0.5,
        },
    })
    assert config is not None

    temps = compute_dynamic_temperature(logits, config)

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    entropy_norm = torch.clamp(entropy / math.log(logits.shape[-1]), 0.0, 1.0)
    expected = torch.clamp(0.8 + 0.2 * (entropy_norm - 0.5), 0.7, 0.9)

    assert temps.tolist() == pytest.approx(expected.tolist())
    assert temps[0].item() < 0.8 < temps[2].item()


def test_entropy_adaptive_uses_low_temp_for_low_entropy():
    logits = torch.tensor([
        [8.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    config = get_dynamic_sampling_config({
        "dynamic_sampling_policy": "entropy_adaptive",
        "dynamic_sampling_kwargs": {
            "H_threshold": 0.5,
            "T_low": 0.2,
            "T_high": 0.8,
        },
    })
    assert config is not None

    temps = compute_dynamic_temperature(logits, config)

    assert temps.tolist() == pytest.approx([0.2, 0.8])


def test_sampling_params_rejects_unknown_dynamic_policy():
    with pytest.raises(ValueError, match="Unsupported dynamic sampling policy"):
        SamplingParams(extra_args={
            "dynamic_sampling_policy": "not_real",
        })


def test_sampling_params_rejects_invalid_dynamic_policy_kwargs():
    with pytest.raises(ValueError, match="Unsupported kwargs"):
        SamplingParams(extra_args={
            "dynamic_sampling_policy": "entropy_continuous",
            "dynamic_sampling_kwargs": {
                "bad_key": 1.0,
            },
        })


def test_entropy_adaptive_rejects_threshold_above_one():
    with pytest.raises(ValueError, match="H_threshold"):
        SamplingParams(extra_args={
            "dynamic_sampling_policy": "entropy_adaptive",
            "dynamic_sampling_kwargs": {
                "H_threshold": 1.5,
            },
        })


def test_entropy_shift_rejects_negative_derived_min_temperature():
    with pytest.raises(ValueError, match="T_base - delta \\* H_mean"):
        SamplingParams(extra_args={
            "dynamic_sampling_policy": "entropy_shift",
            "dynamic_sampling_kwargs": {
                "T_base": 0.1,
                "delta": 0.3,
                "H_mean": 0.5,
            },
        })


def test_edt_rejects_base_outside_open_unit_interval():
    with pytest.raises(ValueError, match="N must be in \\(0, 1\\)"):
        SamplingParams(extra_args={
            "dynamic_sampling_policy": "edt",
            "dynamic_sampling_kwargs": {
                "N": 1.0,
            },
        })


def test_edt_rejects_negative_theta():
    with pytest.raises(ValueError, match="theta"):
        SamplingParams(extra_args={
            "dynamic_sampling_policy": "edt",
            "dynamic_sampling_kwargs": {
                "theta": -0.1,
            },
        })
