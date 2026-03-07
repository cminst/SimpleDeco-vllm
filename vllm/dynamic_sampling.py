# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

DYNAMIC_SAMPLING_POLICY_ARG = "dynamic_sampling_policy"
DYNAMIC_SAMPLING_KWARGS_ARG = "dynamic_sampling_kwargs"
DYNAMIC_SAMPLING_EPS = 1e-5
GREEDY_TEMPERATURE = -1.0


@dataclass(frozen=True)
class DynamicSamplingConfig:
    name: str
    kwargs_items: tuple[tuple[str, float], ...]

    @property
    def kwargs(self) -> dict[str, float]:
        return dict(self.kwargs_items)


_POLICY_DEFAULTS: dict[str, dict[str, float]] = {
    "confidence_gated": {
        "T_high": 1.0,
        "maxprob_threshold": 0.9,
    },
    "entropy_continuous": {
        "T_min": 0.3,
        "T_max": 1.0,
    },
    "entropy_adaptive": {
        "H_threshold": 0.15,
        "T_low": 0.3,
        "T_high": 1.0,
    },
}


def validate_dynamic_sampling_extra_args(
    extra_args: Optional[Mapping[str, Any]],
) -> None:
    if not extra_args:
        return

    if DYNAMIC_SAMPLING_POLICY_ARG not in extra_args:
        if DYNAMIC_SAMPLING_KWARGS_ARG in extra_args:
            raise ValueError(
                f"`{DYNAMIC_SAMPLING_KWARGS_ARG}` requires "
                f"`{DYNAMIC_SAMPLING_POLICY_ARG}` to also be set.")
        return

    _parse_dynamic_sampling_config(extra_args)


def get_dynamic_sampling_config(
    extra_args: Optional[Mapping[str, Any]],
) -> Optional[DynamicSamplingConfig]:
    if not extra_args or DYNAMIC_SAMPLING_POLICY_ARG not in extra_args:
        return None
    return _parse_dynamic_sampling_config(extra_args)


def compute_dynamic_temperature(
    logits: torch.Tensor,
    config: DynamicSamplingConfig,
) -> torch.Tensor:
    kwargs = config.kwargs
    logits = logits.to(torch.float32)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    if config.name == "confidence_gated":
        temps = torch.full(
            (logits.shape[0], ),
            kwargs["T_high"],
            device=logits.device,
            dtype=torch.float32,
        )
        greedy_mask = probs.amax(dim=-1) > kwargs["maxprob_threshold"]
        temps = torch.where(
            greedy_mask,
            torch.full_like(temps, GREEDY_TEMPERATURE),
            temps,
        )
        return _sanitize_temperatures(temps)

    safe_log_probs = torch.where(
        torch.isfinite(log_probs),
        log_probs,
        torch.zeros_like(log_probs),
    )
    entropy = -(probs * safe_log_probs).sum(dim=-1)

    if config.name == "entropy_continuous":
        entropy_max = math.log(logits.shape[-1])
        entropy_norm = torch.clamp(entropy / entropy_max, 0.0, 1.0)
        temps = kwargs["T_min"] + (kwargs["T_max"] - kwargs["T_min"]
                                    ) * entropy_norm
        return _sanitize_temperatures(temps)

    if config.name == "entropy_adaptive":
        entropy_max = math.log(logits.shape[-1])
        if entropy_max > 0.0:
            entropy_norm = entropy / entropy_max
        else:
            entropy_norm = torch.zeros_like(entropy)
        temps = torch.where(
            entropy_norm < kwargs["H_threshold"],
            torch.full_like(entropy, kwargs["T_low"]),
            torch.full_like(entropy, kwargs["T_high"]),
        )
        return _sanitize_temperatures(temps)

    raise ValueError(f"Unsupported dynamic sampling policy: {config.name}")


def _sanitize_temperatures(temps: torch.Tensor) -> torch.Tensor:
    return torch.where(
        temps < DYNAMIC_SAMPLING_EPS,
        torch.full_like(temps, GREEDY_TEMPERATURE),
        temps,
    )


def _parse_dynamic_sampling_config(
    extra_args: Mapping[str, Any],
) -> DynamicSamplingConfig:
    policy = extra_args[DYNAMIC_SAMPLING_POLICY_ARG]
    if not isinstance(policy, str):
        raise ValueError(
            f"`{DYNAMIC_SAMPLING_POLICY_ARG}` must be a string, got "
            f"{type(policy).__name__}.")
    if policy not in _POLICY_DEFAULTS:
        raise ValueError(
            f"Unsupported dynamic sampling policy: {policy}. "
            f"Supported policies: {sorted(_POLICY_DEFAULTS)}")

    raw_kwargs = extra_args.get(DYNAMIC_SAMPLING_KWARGS_ARG, {})
    if raw_kwargs is None:
        raw_kwargs = {}
    if not isinstance(raw_kwargs, Mapping):
        raise ValueError(
            f"`{DYNAMIC_SAMPLING_KWARGS_ARG}` must be a mapping, got "
            f"{type(raw_kwargs).__name__}.")

    supported_keys = _POLICY_DEFAULTS[policy]
    unknown_keys = sorted(set(raw_kwargs) - set(supported_keys))
    if unknown_keys:
        raise ValueError(
            f"Unsupported kwargs for dynamic sampling policy {policy}: "
            f"{unknown_keys}")

    kwargs: dict[str, float] = dict(supported_keys)
    for key, value in raw_kwargs.items():
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Dynamic sampling kwarg {key} must be numeric, got "
                f"{type(value).__name__}.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(
                f"Dynamic sampling kwarg {key} must be finite, got {value}.")
        kwargs[key] = value

    if policy == "confidence_gated":
        _validate_non_negative(kwargs["T_high"], "T_high")
        if not 0.0 <= kwargs["maxprob_threshold"] <= 1.0:
            raise ValueError("maxprob_threshold must be in [0, 1].")
    elif policy == "entropy_continuous":
        _validate_non_negative(kwargs["T_min"], "T_min")
        _validate_non_negative(kwargs["T_max"], "T_max")
        if kwargs["T_min"] > kwargs["T_max"]:
            raise ValueError("T_min must be less than or equal to T_max.")
    elif policy == "entropy_adaptive":
        _validate_non_negative(kwargs["H_threshold"], "H_threshold")
        if kwargs["H_threshold"] > 1.0:
            raise ValueError("H_threshold must be less than or equal to 1.")
        _validate_non_negative(kwargs["T_low"], "T_low")
        _validate_non_negative(kwargs["T_high"], "T_high")

    return DynamicSamplingConfig(
        name=policy,
        kwargs_items=tuple(sorted(kwargs.items())),
    )


def _validate_non_negative(value: float, key: str) -> None:
    if value < 0.0:
        raise ValueError(f"{key} must be non-negative.")
