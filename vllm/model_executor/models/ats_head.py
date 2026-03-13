# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import nn
from transformers.activations import ACT2FN


def _feature_dim_for_key(feature_key: str, hidden_size: int) -> int:
    if feature_key in {"hidden_states", "output_token_feature"}:
        return hidden_size
    if feature_key in {"maxlogit", "logits_std"}:
        return 1
    raise ValueError(f"Unsupported ATS feature_key component: {feature_key}")


def get_feature(
    feature_key: str,
    *,
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> torch.Tensor:
    if feature_key == "hidden_states":
        return hidden_states
    if feature_key == "maxlogit":
        return logits.max(dim=-1, keepdim=True).values
    if feature_key == "output_token_feature":
        top_indices = logits.argmax(dim=-1)
        return lm_head_weight[top_indices]
    if feature_key == "logits_std":
        return torch.log(logits.std(dim=-1, keepdim=True).clamp_min(1e-8))
    raise ValueError(f"Unsupported ATS feature_key: {feature_key}")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float()
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device, torch.get_default_dtype())

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        num_key_value_heads: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.attention_dropout = attention_dropout
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        causal_mask = torch.full(
            (seq_len, seq_len),
            fill_value=torch.finfo(attn_weights.dtype).min,
            device=hidden_states.device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_weights = attn_weights + causal_mask.view(1, 1, seq_len, seq_len)
        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, None, None, :].to(attn_weights.dtype)) * torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights + mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.dropout(attn_weights, p=self.attention_dropout, train=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class ATSMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.self_attn = SelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
        )
        self.mlp = ATSMLP(hidden_size, intermediate_size, hidden_act=hidden_act)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class BaseATSHead(nn.Module):
    def __init__(
        self,
        *,
        feature_key: str,
        max_temperature: float,
        normalize_logits: bool,
    ):
        super().__init__()
        self.feature_keys = feature_key.split("+")
        self.max_temperature = max_temperature
        self.normalize_logits = normalize_logits

    def construct_features(
        self,
        *,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        features = [
            get_feature(
                key,
                hidden_states=hidden_states,
                logits=logits,
                lm_head_weight=lm_head_weight,
            )
            for key in self.feature_keys
        ]
        return torch.cat(features, dim=-1)

    def get_head_output(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_temperature_scale(
        self,
        *,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        lm_head_weight: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        features = self.construct_features(
            hidden_states=hidden_states,
            logits=logits,
            lm_head_weight=lm_head_weight,
        )
        head_output = self.get_head_output(
            features,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return torch.exp(head_output).clamp(max=self.max_temperature)

    def apply_scale(
        self,
        logits: torch.Tensor,
        temperature_scale: torch.Tensor,
    ) -> torch.Tensor:
        if self.normalize_logits:
            logits = logits / logits.std(dim=-1, keepdim=True).clamp_min(1e-8)
        return logits * temperature_scale


class LinearATSHead(BaseATSHead):
    def __init__(self, in_features: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.linear = nn.Linear(in_features, 1)

    def get_head_output(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        del attention_mask, position_ids
        return self.linear(features.to(self.linear.weight.dtype))


class MLPATSHead(BaseATSHead):
    def __init__(
        self,
        in_features: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.mlp = ATSMLP(in_features, intermediate_size, hidden_act=hidden_act)
        self.linear = nn.Linear(in_features, 1)

    def get_head_output(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        del attention_mask, position_ids
        features = features.to(self.linear.weight.dtype)
        return self.linear(self.mlp(features))


class TransformerATSHead(BaseATSHead):
    def __init__(
        self,
        in_features: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        num_key_value_heads: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.transformer = DecoderLayer(
            hidden_size=in_features,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            hidden_act=hidden_act,
            rms_norm_eps=rms_norm_eps,
        )
        self.linear = nn.Linear(in_features, 1)

    def get_head_output(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        features = features.to(self.linear.weight.dtype)
        hidden_states = self.transformer(
            hidden_states=features,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.linear(hidden_states)


def build_ats_head(config: Any) -> BaseATSHead:
    in_features = sum(
        _feature_dim_for_key(key, config.hidden_size)
        for key in config.feature_key.split("+")
    )
    common_kwargs = {
        "feature_key": config.feature_key,
        "max_temperature": config.max_temperature,
        "normalize_logits": config.normalize_logits,
    }
    if config.calibration_type == "linear":
        return LinearATSHead(in_features=in_features, **common_kwargs)
    if config.calibration_type == "mlp":
        return MLPATSHead(
            in_features=in_features,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            **common_kwargs,
        )
    if config.calibration_type == "transformer":
        return TransformerATSHead(
            in_features=in_features,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_bias=config.attention_bias,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported ATS calibration_type: {config.calibration_type}")
