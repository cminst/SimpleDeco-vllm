from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

from vllm.dynamic_sampling import get_dynamic_sampling_config

AUTODECO_HEADS_ARG = "autodeco_heads"
AUTODECO_TEMPERATURE_HEAD = "temperature"
AUTODECO_TOP_P_HEAD = "top_p"

_HEAD_ORDER = (
    AUTODECO_TEMPERATURE_HEAD,
    AUTODECO_TOP_P_HEAD,
)
_HEAD_ALIASES = {
    "temp": AUTODECO_TEMPERATURE_HEAD,
    "temperature": AUTODECO_TEMPERATURE_HEAD,
    "top_p": AUTODECO_TOP_P_HEAD,
    "top-p": AUTODECO_TOP_P_HEAD,
    "topp": AUTODECO_TOP_P_HEAD,
}
_ALL_HEAD_ALIASES = {"all", "both"}
_NO_HEAD_ALIASES = {"none", "base"}


@dataclass(frozen=True)
class AutoDecoHeadSelection:
    use_temperature_head: bool = True
    use_top_p_head: bool = True

    def to_list(self) -> list[str]:
        heads: list[str] = []
        if self.use_temperature_head:
            heads.append(AUTODECO_TEMPERATURE_HEAD)
        if self.use_top_p_head:
            heads.append(AUTODECO_TOP_P_HEAD)
        return heads

    def any_enabled(self) -> bool:
        return self.use_temperature_head or self.use_top_p_head


def normalize_autodeco_heads_value(value: Any) -> list[str]:
    tokens = _coerce_head_tokens(value)
    requested = set[str]()
    for token in tokens:
        if token in _ALL_HEAD_ALIASES:
            if len(tokens) > 1:
                raise ValueError(
                    "`autodeco_heads` cannot mix `all`/`both` with specific heads."
                )
            requested.update(_HEAD_ORDER)
            continue
        if token in _NO_HEAD_ALIASES:
            if len(tokens) > 1:
                raise ValueError(
                    "`autodeco_heads` cannot mix `none`/`base` with other heads."
                )
            requested.clear()
            break
        if token not in _HEAD_ALIASES:
            allowed = sorted(
                set(_HEAD_ALIASES)
                | _ALL_HEAD_ALIASES
                | _NO_HEAD_ALIASES
            )
            raise ValueError(
                f"Unsupported autodeco head {token!r}. Supported values: {allowed}"
            )
        requested.add(_HEAD_ALIASES[token])
    return [head for head in _HEAD_ORDER if head in requested]


def normalize_autodeco_extra_args(
    extra_args: MutableMapping[str, Any] | None,
) -> None:
    if not extra_args or AUTODECO_HEADS_ARG not in extra_args:
        return
    extra_args[AUTODECO_HEADS_ARG] = normalize_autodeco_heads_value(
        extra_args[AUTODECO_HEADS_ARG]
    )


def get_requested_autodeco_head_selection(
    extra_args: Mapping[str, Any] | None,
) -> AutoDecoHeadSelection:
    if not extra_args or AUTODECO_HEADS_ARG not in extra_args:
        return AutoDecoHeadSelection()
    normalized = normalize_autodeco_heads_value(extra_args[AUTODECO_HEADS_ARG])
    return AutoDecoHeadSelection(
        use_temperature_head=AUTODECO_TEMPERATURE_HEAD in normalized,
        use_top_p_head=AUTODECO_TOP_P_HEAD in normalized,
    )


def get_effective_autodeco_head_selection(
    extra_args: Mapping[str, Any] | None,
    *,
    enable_temperature_head: bool,
    enable_top_p_head: bool,
) -> AutoDecoHeadSelection:
    requested = get_requested_autodeco_head_selection(extra_args)
    return AutoDecoHeadSelection(
        use_temperature_head=(
            requested.use_temperature_head and enable_temperature_head
        ),
        use_top_p_head=requested.use_top_p_head and enable_top_p_head,
    )


def validate_autodeco_runtime_extra_args(
    extra_args: MutableMapping[str, Any] | None,
    *,
    is_autodeco_model: bool,
    enable_temperature_head: bool,
    enable_top_p_head: bool,
) -> None:
    if not extra_args:
        return
    normalize_autodeco_extra_args(extra_args)
    if not is_autodeco_model:
        return

    requested_selection = get_requested_autodeco_head_selection(extra_args)
    dynamic_sampling_config = get_dynamic_sampling_config(extra_args)
    selection = get_effective_autodeco_head_selection(
        extra_args,
        enable_temperature_head=enable_temperature_head,
        enable_top_p_head=enable_top_p_head,
    )
    if (
        AUTODECO_HEADS_ARG in extra_args
        and requested_selection.any_enabled()
        and selection != requested_selection
    ):
        missing_heads = [
            head for head in requested_selection.to_list()
            if head not in selection.to_list()
        ]
        raise ValueError(
            "AutoDeco request asked for checkpoint heads that are unavailable. "
            f"Requested {requested_selection.to_list()}, but the checkpoint only "
            "enables "
            f"{_available_head_names(enable_temperature_head, enable_top_p_head)}. "
            f"Missing requested heads: {missing_heads}."
        )
    if dynamic_sampling_config is None:
        return

    if selection.any_enabled():
        raise ValueError(
            "AutoDeco requests cannot combine `dynamic_sampling_policy` with "
            "enabled AutoDeco heads. Use `autodeco_heads=none` or a non-AutoDeco "
            "model."
        )


def _coerce_head_tokens(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_tokens = value.split(",")
    elif isinstance(value, Iterable):
        raw_tokens = list(value)
    else:
        raise ValueError(
            "`autodeco_heads` must be a comma-separated string or a sequence "
            f"of strings, got {type(value).__name__}."
        )

    tokens: list[str] = []
    for token in raw_tokens:
        if not isinstance(token, str):
            raise ValueError(
                "`autodeco_heads` entries must be strings, got "
                f"{type(token).__name__}."
            )
        normalized = token.strip().lower()
        if not normalized:
            continue
        tokens.append(normalized)
    return tokens


def _available_head_names(
    enable_temperature_head: bool,
    enable_top_p_head: bool,
) -> list[str]:
    available: list[str] = []
    if enable_temperature_head:
        available.append(AUTODECO_TEMPERATURE_HEAD)
    if enable_top_p_head:
        available.append(AUTODECO_TOP_P_HEAD)
    return available
