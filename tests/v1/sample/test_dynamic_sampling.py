# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import SamplingParams
from vllm.dynamic_sampling import (GREEDY_TEMPERATURE,
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
