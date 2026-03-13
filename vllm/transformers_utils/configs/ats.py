# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

from transformers import PretrainedConfig


class ATSConfig(PretrainedConfig):
    model_type = "ats"
    has_no_defaults_at_init = True

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        base_model_type: Optional[str] = None,
        calibration_type: str = "transformer",
        feature_key: str = "hidden_states",
        freeze_base_model: bool = True,
        normalize_logits: bool = False,
        max_temperature: float = 10.0,
        loss_type: str = "selective_smoothing",
        label_smoothing: float = 1.0,
        smooth_loss_weight: float = 0.5,
        label_smoothing_type: str = "uniform",
        smoothing_topk: int = 5,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 4096,
        attention_dropout: float = 0.0,
        num_attention_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        overwrite_logits: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        hidden_size = kwargs.get("hidden_size") or kwargs.get("n_embd")
        if hidden_size is None:
            raise ValueError("ATSConfig requires hidden_size from the saved checkpoint config.")
        num_attention_heads = num_attention_heads or kwargs.get("num_attention_heads") or kwargs.get("n_head")
        max_position_embeddings = kwargs.get("max_position_embeddings") or kwargs.get("n_positions") or kwargs.get("n_ctx") or max_position_embeddings
        self.base_model_name_or_path = base_model_name_or_path
        if base_model_name_or_path is not None:
            self._name_or_path = base_model_name_or_path
        self.base_model_type = base_model_type or kwargs.get("base_model_type") or kwargs.get("model_type")
        self.hidden_size = hidden_size
        self.calibration_type = calibration_type
        self.feature_key = feature_key
        self.freeze_base_model = freeze_base_model
        self.normalize_logits = normalize_logits
        self.max_temperature = max_temperature
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.smooth_loss_weight = smooth_loss_weight
        self.label_smoothing_type = label_smoothing_type
        self.smoothing_topk = smoothing_topk
        self.in_features = hidden_size
        self.intermediate_size = intermediate_size or kwargs.get("intermediate_size", hidden_size * 4)
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or kwargs.get("num_key_value_heads", self.num_attention_heads)
        self.hidden_act = hidden_act
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.overwrite_logits = overwrite_logits
        self.architectures = ["ATSModelForCausalLM"]
