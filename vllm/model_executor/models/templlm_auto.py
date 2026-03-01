# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 CMINST.
# Copyright 2026 The AutoDeco team.
# Copyright 2023 The vLLM team.
#
# This code is taken directly from AutoDeco-vllm's Github with slight
# modifications for config-related bug fixes.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AutoDeco vLLM implementation for training temperature and top-p heads."""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedModel,
    GenerationMixin,
    PretrainedConfig,
)
from transformers.utils import ModelOutput, logging
from transformers.cache_utils import Cache

from .autodeco_heads import TempHead, TopPHead

logger = logging.get_logger(__name__)

def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

@dataclass
class AutoDecoOutputWithPast(ModelOutput):
    """
    Output class for AutoDeco models with past key values.
    Compatible with both standard and MoE models.
    """
    loss: Optional[torch.FloatTensor] = None
    temp_loss: Optional[torch.FloatTensor] = None
    top_p_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None  # For MoE models
    lm_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    temp_logits: Optional[torch.FloatTensor] = None
    top_p_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor, ...]] = None  # For MoE models


class AutoDecoModelForCausalLMConfig(PretrainedConfig):

    model_type = "autodeco"  # Class attribute - REQUIRED for transformers registration!

    def __init__(
        self,
        enable_temperature_head: bool = True,
        enable_top_p_head: bool = True,
        use_enhanced_features: bool = True,
        base_model_name_or_path: Optional[str] = None,
        train_temp: bool = False,
        train_top_p: bool = False,
        **kwargs,  # All base model config parameters
    ):
        super().__init__(**kwargs)
        hidden_size = kwargs.get("hidden_size")

        self.enable_temperature_head = enable_temperature_head
        self.enable_top_p_head = enable_top_p_head
        self.top_p_hidden_size = hidden_size
        self.temperature_hidden_size = hidden_size
        self.use_enhanced_features = use_enhanced_features
        self.train_temp = train_temp
        self.train_top_p = train_top_p
        self.base_model_name_or_path = base_model_name_or_path
        if base_model_name_or_path is not None:
            self._name_or_path = base_model_name_or_path

        # Important! Need to preserve base_model_type from config.json
        self.base_model_type = kwargs.get("base_model_type")

class AutoDecoModelForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Unified AutoDeco model that wraps any AutoModelForCausalLM with
    temperature and top-p prediction heads.

    This eliminates the need for separate model files for each architecture.
    """

    config_class = AutoDecoModelForCausalLMConfig
    base_model_prefix = "AutoDeco"
    supports_gradient_checkpointing = True
    _no_split_modules = []  # Will be set based on base model

    def __init__(self, config: AutoDecoModelForCausalLMConfig, **kwargs):
        """
        Initialize AutoDeco model.

        Args:
            config: AutoDecoConfig instance with base model information
            **kwargs: Additional arguments (for backward compatibility)
        """
        if not isinstance(config, AutoDecoModelForCausalLMConfig):
            raise ValueError(
                f"config must be an AutoDecoModelForCausalLMConfig instance, got {type(config)}"
            )

        super().__init__(config)

        # Get base model path
        base_model_path = config.base_model_name_or_path
        if base_model_path is None:
            raise ValueError("config.base_model_name_or_path must be specified")

        # Load the base causal LM model
        logger.info(f"Loading base model from {base_model_path}")
        logger.info(f"Base model type: {config.base_model_type}")

        # For loading from pretrained AutoDeco checkpoint, base model might already be loaded
        if kwargs.get("load_base_model") is False:
            # This will be used when loading from a saved AutoDeco checkpoint
            logger.info("Skipping base model loading (will be loaded from checkpoint)")
            # Create a placeholder that will be populated by from_pretrained
            self.llm = None
        else:
            base_model_args = kwargs.get("base_model_args", ())
            self.llm = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                *base_model_args,
                **kwargs.get("base_model_kwargs", {}),
            )

        # Get hidden size from config
        hidden_size = config.hidden_size

        # Initialize AutoDeco heads
        self._init_heads(hidden_size, config.use_enhanced_features)

        # Training flags
        self.train_temp = getattr(config, "train_temp", False)
        self.train_top_p = getattr(config, "train_top_p", False)

        # Check if base model is MoE
        self._init_moe_config()

        # Copy _no_split_modules from base model if available
        if self.llm is not None and hasattr(self.llm, "_no_split_modules"):
            self._no_split_modules = self.llm._no_split_modules

        logger.info("AutoDeco model initialized:")
        logger.info(f"  - base_model_type={config.base_model_type}, base_model_name_or_path={config.base_model_name_or_path}")
        logger.info(f"  - train_temp={self.train_temp}, train_top_p={self.train_top_p}")

        # Log training mode
        if self.train_temp or self.train_top_p:
            heads = []
            if self.train_temp:
                heads.append("temp_head")
            if self.train_top_p:
                heads.append("top_p_head")
            logger.info(f"  - Training mode: AutoDeco heads ({', '.join(heads)})")
        else:
            logger.info("  - Training mode: Base LLM (standard language modeling)")

        logger.info(f"  - is_moe={self.is_moe}")
        logger.info(f"  - hidden_size={hidden_size}")

    def _init_heads(self, hidden_size: int, use_enhanced_features: bool) -> None:
        self.temp_head = TempHead(hidden_size)
        self.top_p_head = TopPHead(
            hidden_size,
            use_enhanced_features=use_enhanced_features,
        )

    def _init_moe_config(self) -> None:
        config = self.config
        self.is_moe = hasattr(config, "num_local_experts") or hasattr(config, "num_experts")
        if self.is_moe:
            self.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.01)
            self.num_experts = getattr(config, "num_local_experts", None) or getattr(config, "num_experts", None)
            self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)

    @staticmethod
    def _flatten_heads_state_dict(temp_head_state: dict, top_p_head_state: dict) -> dict:
        flat_state_dict = {}
        for key, value in temp_head_state.items():
            flat_state_dict[f"temp_head.{key}"] = value
        for key, value in top_p_head_state.items():
            flat_state_dict[f"top_p_head.{key}"] = value
        return flat_state_dict

    @staticmethod
    def _split_prefixed_heads_state_dict(flat_state_dict: dict) -> tuple[dict, dict]:
        temp_head_state = {}
        top_p_head_state = {}

        for key, value in flat_state_dict.items():
            if key.startswith("temp_head."):
                temp_head_state[key[len("temp_head."):]] = value
            elif key.startswith("top_p_head."):
                top_p_head_state[key[len("top_p_head."):]] = value

        return temp_head_state, top_p_head_state

    @classmethod
    def _to_autodeco_config(
        cls,
        config: PretrainedConfig,
        pretrained_model_name_or_path: str,
        train_temp: Optional[bool],
        train_top_p: Optional[bool],
        use_enhanced_features: Optional[bool],
    ) -> AutoDecoModelForCausalLMConfig:
        if isinstance(config, AutoDecoModelForCausalLMConfig):
            autodeco_config = config
        else:
            config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
            base_model_type = config_dict.pop("model_type", getattr(config, "model_type", None))
            autodeco_config = AutoDecoModelForCausalLMConfig(
                **config_dict,
                base_model_name_or_path=pretrained_model_name_or_path,
                base_model_type=base_model_type,
            )

        if autodeco_config.base_model_name_or_path is None:
            autodeco_config.base_model_name_or_path = pretrained_model_name_or_path
            autodeco_config._name_or_path = pretrained_model_name_or_path

        if train_temp is not None:
            autodeco_config.train_temp = train_temp
        if train_top_p is not None:
            autodeco_config.train_top_p = train_top_p
        if use_enhanced_features is not None:
            autodeco_config.use_enhanced_features = use_enhanced_features

        return autodeco_config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        config: Optional[AutoDecoModelForCausalLMConfig] = None,
        train_temp: Optional[bool] = None,
        train_top_p: Optional[bool] = None,
        use_enhanced_features: Optional[bool] = None,
        **kwargs,
    ):
        """
        Load AutoDeco model from either:
        1. A saved AutoDeco checkpoint (contains config.json with model_type="AutoDeco")
        2. A base model (will create new AutoDeco model on top)

        Args:
            pretrained_model_name_or_path: Path or name of the model
            config: Optional AutoDecoConfig (if None, will try to load from path)
            train_temp: Whether to train temperature head (for new models)
            train_top_p: Whether to train top-p head (for new models)
            use_enhanced_features: Whether to use enhanced features (for new models)
            **kwargs: Additional arguments passed to model loading
        """
        config_load_keys = (
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "token",
            "trust_remote_code",
            "use_auth_token",
        )
        config_kwargs = {key: kwargs[key] for key in config_load_keys if key in kwargs}

        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)

        is_autodeco_checkpoint = getattr(config, "model_type", None) == "autodeco"
        config = cls._to_autodeco_config(
            config=config,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            train_temp=train_temp,
            train_top_p=train_top_p,
            use_enhanced_features=use_enhanced_features,
        )

        if not is_autodeco_checkpoint:
            logger.info(
                "Creating a new AutoDeco model from base checkpoint: %s",
                pretrained_model_name_or_path,
            )
            return cls(config, base_model_args=model_args, base_model_kwargs=kwargs)

        checkpoint_path = Path(pretrained_model_name_or_path)
        logger.info("Loading AutoDeco model from heads-only checkpoint: %s", checkpoint_path)
        logger.info(
            "  - base_model_type=%s, base_model_name_or_path=%s",
            config.base_model_type,
            config.base_model_name_or_path,
        )
        logger.info("  - Loading base model from: %s", config.base_model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            *model_args,
            **kwargs,
        )

        # Instantiate lightweight shell first, then attach the loaded base model.
        model = cls(config, load_base_model=False)
        model.llm = base_model
        if hasattr(base_model, "_no_split_modules"):
            model._no_split_modules = base_model._no_split_modules

        safetensors_path = checkpoint_path / "autodeco_heads.safetensors"
        pytorch_path = checkpoint_path / "autodeco_heads.bin"
        loaded = False

        if safetensors_path.exists():
            logger.info("  - Loading heads from: %s", safetensors_path)
            try:
                from safetensors.torch import load_file

                flat_state_dict = load_file(safetensors_path)
                temp_head_state, top_p_head_state = cls._split_prefixed_heads_state_dict(flat_state_dict)
                model.temp_head.load_state_dict(temp_head_state)
                model.top_p_head.load_state_dict(top_p_head_state)
                logger.info("  ✓ Loaded heads from safetensors")
                loaded = True
            except ImportError:
                logger.warning("safetensors not available, trying pytorch format")

        if not loaded and pytorch_path.exists():
            logger.info("  - Loading heads from: %s", pytorch_path)
            autodeco_state_dict = torch.load(pytorch_path, map_location="cpu")
            if "temp_head" in autodeco_state_dict and "top_p_head" in autodeco_state_dict:
                temp_head_state = autodeco_state_dict["temp_head"]
                top_p_head_state = autodeco_state_dict["top_p_head"]
            else:
                temp_head_state, top_p_head_state = cls._split_prefixed_heads_state_dict(autodeco_state_dict)

            model.temp_head.load_state_dict(temp_head_state)
            model.top_p_head.load_state_dict(top_p_head_state)
            logger.info("  ✓ Loaded heads from pytorch checkpoint")
            loaded = True

        if not loaded:
            raise FileNotFoundError(
                f"Could not find AutoDeco heads weights at {checkpoint_path}. "
                f"Expected either {safetensors_path.name} or {pytorch_path.name}"
            )

        logger.info("✓ AutoDeco model loaded successfully from heads-only checkpoint")
        return model

    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs
    ):
        """
        Save AutoDeco model config and heads weights only (lightweight).

        This saves:
        1. config.json with AutoDecoConfig (includes base_model_name_or_path)
        2. ONLY temp_head and top_p_head weights (NOT the base model weights)

        The base model will be reloaded from base_model_name_or_path when loading.
        This significantly reduces checkpoint size (~5MB vs ~14GB).

        Use merge_autodeco.py script to create full checkpoint for vLLM deployment.
        """
        from transformers.modeling_utils import unwrap_model
        from huggingface_hub import split_torch_state_dict_into_shards

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # Unwrap model (handles PEFT wrapping)
        model_to_save = unwrap_model(self)

        # Save config
        if is_main_process:
            model_to_save.config.save_pretrained(save_directory)

        # Prepare state_dict (heads only)
        logger.info("Saving AutoDeco heads only (lightweight)")

        fallback_state_dict = self._flatten_heads_state_dict(
            model_to_save.temp_head.state_dict(),
            model_to_save.top_p_head.state_dict(),
        )

        if state_dict is None:
            # No state_dict provided, get it from model.
            state_dict = fallback_state_dict
        else:
            # state_dict provided (e.g., by Trainer), filter to keep only heads
            # The provided state_dict may have full model keys or already-prefixed keys
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if key.startswith("temp_head.") or key.startswith("top_p_head.")
            }

            # If no heads found with prefix, the state_dict might not have prefixes
            # In that case, extract from the model itself
            if not state_dict:
                logger.warning("No heads found in provided state_dict, extracting from model")
                state_dict = fallback_state_dict

        # Use custom name for heads-only checkpoint
        weights_name = "autodeco_heads.safetensors" if safe_serialization else "autodeco_heads.bin"
        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")

        # Shard if needed
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict,
            filename_pattern=filename_pattern,
            max_shard_size=max_shard_size
        )

        # Prepare index if sharded
        index = None
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }

        # Save shards
        for shard_file, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}

            if safe_serialization:
                from safetensors.torch import save_file
                save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
            else:
                save_function(shard, os.path.join(save_directory, shard_file))

        # Save index if sharded
        if index is not None:
            index_file = "autodeco_heads.safetensors.index.json" if safe_serialization else "autodeco_heads.bin.index.json"
            index_path = os.path.join(save_directory, index_file)
            with open(index_path, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(f"Model weights saved in {len(state_dict_split.filename_to_tensors)} shard(s)")
        else:
            logger.info(f"Model weights saved in {os.path.join(save_directory, weights_name)}")

        logger.info(f"✓ Lightweight checkpoint saved (base model: {self.config.base_model_name_or_path})")

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        if hasattr(self.llm, 'set_decoder'):
            self.llm.set_decoder(decoder)

    def get_decoder(self):
        if hasattr(self.llm, 'get_decoder'):
            return self.llm.get_decoder()
        return self.llm.model if hasattr(self.llm, 'model') else self.llm

    def _compute_temp_loss(self, unscaled_logits, temp_logits, labels):
        """Compute temperature loss"""
        unscaled_shift = unscaled_logits[:, :-1, :]
        temp_shift = temp_logits[:, :-1, :].clamp_min(1e-2)
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100

        # Base-model confidence of GT token
        with torch.no_grad():
            base_probs = torch.softmax(unscaled_shift, dim=-1)
            pred_ids = base_probs.argmax(dim=-1)
            correct_positions = (pred_ids == shift_labels.clamp_min(0))

            # Randomly drop 60% of positions where model argmax == GT
            rand_vals = torch.rand(shift_labels.shape, device=shift_labels.device)
            drop_mask = (rand_vals < 0.6) & valid_mask & correct_positions
            masked_valid_mask = valid_mask & (~drop_mask)

        scaled_shift = unscaled_shift / temp_shift
        scaled_valid = scaled_shift[masked_valid_mask]
        unscaled_valid = unscaled_shift[masked_valid_mask]
        labels_valid = shift_labels[masked_valid_mask]

        if labels_valid.numel() > 0:
            token_ce = F.cross_entropy(
                scaled_valid.view(-1, scaled_valid.size(-1)),
                labels_valid.view(-1),
                reduction='none',
                label_smoothing=0.0,
            )
            token_ce = token_ce * torch.softmax(unscaled_valid, dim=-1).gather(
                1, labels_valid.unsqueeze(-1)
            ).squeeze(-1).detach()
            return token_ce.mean()

        return torch.tensor(0.0, device=unscaled_logits.device)

    def _compute_top_p_loss(self, unscaled_logits, temp_logits, top_p_logits, labels, method='soft'):
        """
        Compute top-p loss

        Args:
            method: 'soft' for soft top-p with exponential decay, 'mse' for MSE loss
        """
        unscaled_shift = unscaled_logits[:, :-1, :]
        temp_shift = temp_logits[:, :-1, :]
        top_p_shift = top_p_logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        if method == 'soft':
            # Soft top-p loss with exponential decay
            steepness = 30.0
            scaled_logits = unscaled_shift / temp_shift.clamp_min(1e-8)
            probs = torch.softmax(scaled_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            overage = torch.relu(cumulative_probs - top_p_shift)
            decay_factor = torch.exp(-steepness * overage)

            mask = torch.zeros_like(probs).scatter_(-1, sorted_indices, decay_factor)
            masked_probs = probs * mask
            renormalized_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-9)
            log_probs = torch.log(renormalized_probs + 1e-9)

            valid_mask = shift_labels != -100
            if not valid_mask.any():
                return torch.tensor(0.0, device=unscaled_logits.device)

            log_probs = log_probs[valid_mask]
            labels_shift = shift_labels[valid_mask]
            unscaled_valid = unscaled_shift[valid_mask]

            token_ce = F.nll_loss(
                log_probs.view(-1, log_probs.size(-1)),
                labels_shift.view(-1),
                reduction='none'
            )
            token_ce = token_ce * torch.softmax(unscaled_valid, dim=-1).gather(
                1, labels_shift.unsqueeze(-1)
            ).squeeze(-1).detach()

            return token_ce.mean()

        elif method == 'mse':
            # MSE loss with asymmetric penalty
            with torch.no_grad():
                scaled_shift = unscaled_shift / temp_shift.clamp_min(1e-8)
                probs_sorted, idx_sorted = torch.softmax(scaled_shift, dim=-1).sort(dim=-1, descending=True)
                rank = (idx_sorted == shift_labels.unsqueeze(-1)).float().argmax(dim=-1)
                cumsum = probs_sorted.cumsum(dim=-1)
                top_p_labels = cumsum.gather(-1, rank.unsqueeze(-1)).squeeze(-1).detach()

                valid_mask = shift_labels != -100
                base_probs = torch.softmax(unscaled_shift, dim=-1)
                pred_ids = base_probs.argmax(dim=-1)
                correct_positions = (pred_ids == shift_labels.clamp_min(0))
                rand_vals = torch.rand(shift_labels.shape, device=shift_labels.device)
                drop_mask = (rand_vals < 0.6) & valid_mask & correct_positions
                masked_valid_mask = valid_mask & (~drop_mask)

            top_p_pred = top_p_shift.squeeze(-1)
            target = top_p_labels
            error = top_p_pred - target
            lambda_under = 2.0
            lambda_over = 1.0
            per_token_loss = torch.where(
                error < 0,
                lambda_under * error.pow(2),
                lambda_over * error.pow(2)
            )

            if not masked_valid_mask.any():
                return torch.tensor(0.0, device=unscaled_logits.device)

            return per_token_loss[masked_valid_mask].mean()

        else:
            raise ValueError(f"Unknown top-p loss method: {method}")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        top_p_loss_method: str = 'soft',  # 'soft' or 'mse'
        **kwargs,
    ) -> AutoDecoOutputWithPast:
        """
        Forward pass of AutoDeco model.

        Args:
            top_p_loss_method: Method for computing top-p loss ('soft' or 'mse')
        """
        # Prepare kwargs for base model
        base_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'inputs_embeds': inputs_embeds,
            'use_cache': use_cache,
            'output_attentions': output_attentions,
            'output_hidden_states': True,  # Always need hidden states
            'cache_position': cache_position,
        }

        # Add MoE-specific args if applicable
        if self.is_moe and output_router_logits is not None:
            base_kwargs['output_router_logits'] = output_router_logits

        # Forward through base model
        outputs = self.llm(**base_kwargs, **kwargs)

        # Get hidden states
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

        # Compute logits and predictions
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # Compute unscaled logits
        # If training base model (train_temp=False and train_top_p=False), need gradients
        # Otherwise, can use no_grad for efficiency
        if self.train_temp or self.train_top_p:
            # Training heads only - no gradient needed for base model logits
            with torch.no_grad():
                unscaled_logits = self.llm.lm_head(hidden_states[:, slice_indices, :])
        else:
            # Training base model - need gradients
            unscaled_logits = self.llm.lm_head(hidden_states[:, slice_indices, :])

        temp_logits = self.temp_head(hidden_states[:, slice_indices, :])
        top_p_logits = self.top_p_head(
            hidden_states[:, slice_indices, :],
            temp_logits.detach(),
            unscaled_logits=unscaled_logits,
        )

        # Compute losses
        loss, lm_loss, temp_loss, top_p_loss = None, None, None, None

        if labels is not None:
            if self.train_temp or self.train_top_p:
                # Mode 1: Training AutoDeco heads
                losses = []

                if self.train_temp:
                    temp_loss = self._compute_temp_loss(unscaled_logits, temp_logits, labels)
                    losses.append(temp_loss)

                if self.train_top_p:
                    top_p_loss = self._compute_top_p_loss(
                        unscaled_logits, temp_logits, top_p_logits, labels,
                        method=top_p_loss_method
                    )
                    losses.append(top_p_loss)

                if losses:
                    loss = sum(losses)

            else:
                # Mode 2: Training base LLM (when both train_temp and train_top_p are False)
                # Compute standard language modeling loss
                logger.debug("Computing standard LM loss (training base model)")
                lm_loss = self.llm.loss_function(unscaled_logits, labels, self.llm.vocab_size, **kwargs)
                loss = lm_loss

        # # Handle MoE auxiliary loss
        aux_loss = None
        # if self.is_moe and hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
        #     aux_loss = load_balancing_loss_func(
        #         outputs.router_logits,
        #         self.llm.num_experts,
        #         self.llm.num_experts_per_tok,
        #         attention_mask,
        #     )
        #     if labels is not None:
        #         loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return AutoDecoOutputWithPast(
            loss=loss,
            temp_loss=temp_loss,
            top_p_loss=top_p_loss,
            aux_loss=aux_loss,
            lm_loss=lm_loss,
            logits=unscaled_logits,
            temp_logits=temp_logits,
            top_p_logits=top_p_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            router_logits=outputs.router_logits if hasattr(outputs, 'router_logits') else None,
        )

    def generate(self, *args, **kwargs):
        """
        Generate using the base model's generate method.
        Note: This uses the base model's generation, not the AutoDeco heads.
        For generation with dynamic temperature/top-p, you'll need custom generation logic.
        """
        return self.llm.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation"""
        return self.llm.prepare_inputs_for_generation(*args, **kwargs)


# Register the config and model
from transformers import AutoModel, AutoModelForCausalLM as AutoModelForCausalLMClass

try:
    AutoConfig.register("autodeco", AutoDecoModelForCausalLMConfig)
    AutoModel.register(AutoDecoModelForCausalLMConfig, AutoDecoModelForCausalLM)
    AutoModelForCausalLMClass.register(AutoDecoModelForCausalLMConfig, AutoDecoModelForCausalLM)

    print("AutoDeco model registered with transformers (AutoConfig, AutoModel, AutoModelForCausalLM)")
except ValueError as e:
    print(f"Failed to register AutoDecoModelForCausalLM: {e}")


__all__ = [
    'AutoDecoModelForCausalLM',
    'AutoDecoModelForCausalLMConfig',
    'AutoDecoOutputWithPast',
    'TempHead',
    'TopPHead',
]
