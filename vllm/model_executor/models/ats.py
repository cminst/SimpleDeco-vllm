# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .ats_head import BaseATSHead, build_ats_head
from .interfaces import SupportsLoRA, SupportsPP
from .utils import AutoWeightsLoader, maybe_prefix

logger = init_logger(__name__)


class ATSModelForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        if not hasattr(config, "base_model_type"):
            raise ValueError(
                "ATS vLLM checkpoints require config.base_model_type. "
                "Use script/merge_ats.py to create a merged checkpoint."
            )
        self.config = config
        self.quant_config = quant_config
        self.lora_config = lora_config
        self.llm = self._get_base_model_class(config.base_model_type)(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "llm"),
        )
        self.ats_head: BaseATSHead = build_ats_head(config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self._runtime_hidden_states: Optional[torch.Tensor] = None
        self._runtime_metadata: Optional[dict[str, torch.Tensor]] = None
        self._request_hidden_cache: dict[int, torch.Tensor] = {}
        if hasattr(self.llm, "make_empty_intermediate_tensors"):
            self.make_empty_intermediate_tensors = self.llm.make_empty_intermediate_tensors
        if hasattr(self.llm, "packed_modules_mapping"):
            self.packed_modules_mapping = self.llm.packed_modules_mapping

    def _get_base_model_class(self, base_model_type: str):
        model_registry = {
            "qwen2": ("qwen2", "Qwen2ForCausalLM"),
            "qwen3": ("qwen3", "Qwen3ForCausalLM"),
            "qwen2_moe": ("qwen2_moe", "Qwen2MoeForCausalLM"),
            "qwen3_moe": ("qwen3_moe", "Qwen3MoeForCausalLM"),
            "gpt_oss": ("gpt_oss", "GptOssForCausalLM"),
            "llama": ("llama", "LlamaForCausalLM"),
            "mistral": ("mistral", "MistralForCausalLM"),
            "mixtral": ("mixtral", "MixtralForCausalLM"),
            "deepseek_v3": ("deepseek_v2", "DeepseekV3ForCausalLM"),
            "gpt2": ("gpt2", "GPT2LMHeadModel"),
        }
        if base_model_type not in model_registry:
            raise ValueError(f"Unsupported ATS base_model_type: {base_model_type}")
        module_name, class_name = model_registry[base_model_type]
        module = __import__(f"vllm.model_executor.models.{module_name}", fromlist=[class_name])
        return getattr(module, class_name)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings(input_ids)

    def _capture_runtime_metadata(self) -> Optional[dict[str, torch.Tensor]]:
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = next(iter(attn_metadata.values()), None)
        if attn_metadata is None:
            return None
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
        block_table_tensor = getattr(attn_metadata, "block_table_tensor", None)
        num_computed_tokens_cpu = getattr(attn_metadata, "num_computed_tokens_cpu", None)
        if query_start_loc is None or block_table_tensor is None or num_computed_tokens_cpu is None:
            return None
        return {
            "query_start_loc": query_start_loc.clone(),
            "block_table_tensor": block_table_tensor.clone(),
            "num_computed_tokens_cpu": num_computed_tokens_cpu.clone(),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.llm.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        if get_pp_group().is_last_rank and not isinstance(hidden_states, IntermediateTensors):
            self._runtime_hidden_states = hidden_states
            self._runtime_metadata = self._capture_runtime_metadata()
        else:
            self._runtime_hidden_states = None
            self._runtime_metadata = None
        return hidden_states

    def _compute_base_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
    ) -> torch.Tensor:
        return self.logits_processor(
            lm_head=self.llm.lm_head,
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            temp_head=None,
            top_p_head=None,
        )

    def _compute_simple_head_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata],
    ) -> torch.Tensor:
        base_logits = self._compute_base_logits(hidden_states, sampling_metadata)
        batch = hidden_states.shape[0]
        seq_hidden_states = hidden_states.unsqueeze(1)
        seq_logits = base_logits.unsqueeze(1)
        attention_mask = torch.ones((batch, 1), device=hidden_states.device, dtype=hidden_states.dtype)
        position_ids = torch.zeros((batch, 1), device=hidden_states.device, dtype=torch.long)
        temperature_scale = self.ats_head.get_temperature_scale(
            hidden_states=seq_hidden_states,
            logits=seq_logits,
            lm_head_weight=self.llm.lm_head.weight,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).squeeze(1)
        return self.ats_head.apply_scale(base_logits, temperature_scale)

    def _request_key(self, request_idx: int) -> int:
        if self._runtime_metadata is None:
            return request_idx
        row = self._runtime_metadata["block_table_tensor"][request_idx]
        return int(row[0].item()) if row.numel() > 0 else request_idx

    def _compute_transformer_logits(
        self,
        sample_hidden_states: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata],
    ) -> torch.Tensor:
        if self._runtime_hidden_states is None or self._runtime_metadata is None:
            return self._compute_simple_head_logits(sample_hidden_states, sampling_metadata)
        base_logits = self._compute_base_logits(sample_hidden_states, sampling_metadata)
        query_start_loc = self._runtime_metadata["query_start_loc"]
        num_computed = self._runtime_metadata["num_computed_tokens_cpu"]
        request_scales = []
        for request_idx in range(query_start_loc.numel() - 1):
            start = int(query_start_loc[request_idx].item())
            end = int(query_start_loc[request_idx + 1].item())
            current_chunk = self._runtime_hidden_states[start:end].unsqueeze(0)
            request_key = self._request_key(request_idx)
            if int(num_computed[request_idx].item()) == 0:
                self._request_hidden_cache.pop(request_key, None)
            cached_hidden = self._request_hidden_cache.get(request_key)
            if cached_hidden is None:
                full_hidden = current_chunk
            else:
                full_hidden = torch.cat([cached_hidden.to(current_chunk.device), current_chunk], dim=1)
            full_logits = self._compute_base_logits(full_hidden.squeeze(0), None).unsqueeze(0)
            attention_mask = torch.ones(
                (1, full_hidden.size(1)),
                device=full_hidden.device,
                dtype=full_hidden.dtype,
            )
            position_ids = torch.arange(full_hidden.size(1), device=full_hidden.device).unsqueeze(0)
            temperature_scale = self.ats_head.get_temperature_scale(
                hidden_states=full_hidden,
                logits=full_logits,
                lm_head_weight=self.llm.lm_head.weight,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[:, -1:, :]
            request_scales.append(temperature_scale)
            self._request_hidden_cache[request_key] = full_hidden.detach()
        stacked_scale = torch.cat(request_scales, dim=0).squeeze(1)
        return self.ats_head.apply_scale(base_logits, stacked_scale)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if self.config.calibration_type == "transformer":
            return self._compute_transformer_logits(hidden_states, sampling_metadata)
        return self._compute_simple_head_logits(hidden_states, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ):
        return self.sampler(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        loaded_params = loader.load_weights(weights)
        logger.info(
            "Loaded ATS checkpoint params: llm=%d ats_head=%d",
            sum(1 for name in loaded_params if name.startswith("llm.")),
            sum(1 for name in loaded_params if name.startswith("ats_head.")),
        )
        return loaded_params


__all__ = ["ATSModelForCausalLM"]
