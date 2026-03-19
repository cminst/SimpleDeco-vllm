# SPDX-License-Identifier: Apache-2.0
"""
Unified AutoDeco Model for vLLM Inference

This model loads a complete AutoDeco checkpoint (base model + heads) and
provides dynamic temperature and top-p sampling for inference.

The checkpoint should contain:
- config.json with model_type="autodeco" and base_model_type
- Complete model weights including:
  - llm.* (base model weights)
  - temp_head.* (temperature head weights)
  - top_p_head.* (top-p head weights)

Usage:
    # 1. First merge heads with base model (before vLLM deployment):
    python merge_autodeco.py --mode lightweight_to_full \\
        --autodeco-checkpoint ./lightweight-checkpoint \\
        --output ./full-checkpoint

    # 2. Then load with vLLM:
    for example:
    from vllm import LLM
    llm = LLM(model="./full-checkpoint", trust_remote_code=True)
"""

from typing import Iterable, Set, Tuple, Union

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .autodeco_heads import TempHead, TopPHead
from .interfaces import SupportsLoRA, SupportsPP
from .utils import init_vllm_registered_model

logger = init_logger(__name__)


def maybe_prefix(prefix: str, name: str) -> str:
    return name if not prefix else f"{prefix}.{name}"


def _build_base_model_config(config: PretrainedConfig) -> PretrainedConfig:
    base_model_type = getattr(config, "base_model_type", None)
    if not base_model_type:
        raise ValueError(
            "Merged AutoDeco checkpoint is missing base_model_type in config.json"
        )

    config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
    for key in (
        "enable_temperature_head",
        "enable_top_p_head",
        "use_enhanced_features",
        "train_temp",
        "train_top_p",
        "base_model_name_or_path",
        "base_model_type",
        "model_type",
        "architectures",
    ):
        config_dict.pop(key, None)

    try:
        base_model_config = AutoConfig.for_model(base_model_type, **config_dict)
    except Exception as exc:
        raise ValueError(
            "Failed to reconstruct base model config for merged AutoDeco "
            f"checkpoint with base_model_type={base_model_type!r}"
        ) from exc

    base_model_name_or_path = getattr(config, "base_model_name_or_path", None)
    if base_model_name_or_path is not None:
        base_model_config._name_or_path = base_model_name_or_path
    return base_model_config


class AutoDecoModelForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """
    Unified AutoDeco model for vLLM inference.

    This loads a complete checkpoint containing:
    - Base LLM model (self.llm)
    - Temperature prediction head (self.temp_head)
    - Top-p prediction head (self.top_p_head)

    The checkpoint must be created by merging heads with base model using:
        python merge_autodeco.py --mode lightweight_to_full ...
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        # Validate AutoDeco config
        if not hasattr(config, 'base_model_type'):
            raise ValueError(
                "This model requires an AutoDeco config with 'base_model_type'. "
                "Please use merge_autodeco.py to create a complete AutoDeco checkpoint."
            )

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        base_model_type = config.base_model_type
        use_enhanced_features = getattr(config, 'use_enhanced_features', True)
        self.enable_temperature_head = getattr(config, 'enable_temperature_head', True)
        self.enable_top_p_head = getattr(config, 'enable_top_p_head', True)

        logger.info("="*80)
        logger.info("Initializing AutoDeco model for vLLM:")
        logger.info(f"  - base_model_type: {base_model_type}")
        logger.info(f"  - use_enhanced_features: {use_enhanced_features}")
        logger.info(f"  - hidden_size: {config.hidden_size}")
        logger.info(f"  - enable_temperature_head: {self.enable_temperature_head}")
        logger.info(f"  - enable_top_p_head: {self.enable_top_p_head}")
        logger.info("="*80)

        # Get base model class
        base_model_class = self._get_base_model_class(base_model_type)

        if base_model_class is None:
            raise ValueError(
                f"Unsupported base model type: {base_model_type}. "
                f"Supported types: qwen2, qwen3, qwen2_moe, qwen3_moe, gpt_oss, llama, mistral, mixtral, deepseek_v3"
            )

        logger.info(f"  - Loading base model class: {base_model_class.__name__}")
        base_model_config = _build_base_model_config(config)

        # Initialize the wrapped base model through vLLM's nested-model
        # helper so it sees the correct hf_config/architecture context during
        # layer construction and post-load processing.
        self.llm = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=base_model_config,
            architectures=[base_model_class.__name__],
            prefix=maybe_prefix(prefix, "llm"),
        )

        # Get hidden size
        hidden_size = base_model_config.hidden_size

        # Initialize AutoDeco heads
        self.temp_head = TempHead(hidden_size) if self.enable_temperature_head else None
        self.top_p_head = None
        if self.enable_top_p_head:
            self.top_p_head = TopPHead(
                hidden_size,
                vocab_size=base_model_config.vocab_size,
                use_enhanced_features=use_enhanced_features
            )
            if not self.enable_temperature_head:
                logger.warning(
                    "AutoDeco config enables top-p head without temperature head. "
                    "Using a constant temperature input (1.0) for top-p features."
                )

        # Initialize logits processor
        self.logits_processor = LogitsProcessor(base_model_config.vocab_size)

        # Copy useful attributes from base model
        if hasattr(self.llm, 'make_empty_intermediate_tensors'):
            self.make_empty_intermediate_tensors = (
                self.llm.make_empty_intermediate_tensors
            )

        # For LoRA support
        if hasattr(self.llm, 'packed_modules_mapping'):
            self.packed_modules_mapping = self.llm.packed_modules_mapping

        logger.info("✓ AutoDeco model initialized successfully")
        logger.info("="*80)

    def _get_base_model_class(self, base_model_type: str):
        """
        Dynamically import and return the base model class.

        Args:
            base_model_type: Model type from config (e.g., 'qwen2', 'qwen3', 'gpt_oss')

        Returns:
            Base model class or None if not supported
        """
        # Mapping of model types to their module and class names
        MODEL_REGISTRY = {
            'qwen2': ('qwen2', 'Qwen2ForCausalLM'),
            'qwen3': ('qwen3', 'Qwen3ForCausalLM'),
            'qwen2_moe': ('qwen2_moe', 'Qwen2MoeForCausalLM'),
            'qwen3_moe': ('qwen3_moe', 'Qwen3MoeForCausalLM'),
            'gpt_oss': ('gpt_oss', 'GptOssForCausalLM'),
            'llama': ('llama', 'LlamaForCausalLM'),
            'mistral': ('mistral', 'MistralForCausalLM'),
            'mixtral': ('mixtral', 'MixtralForCausalLM'),
            "deepseek_v3": ("deepseek_v2", "DeepseekV3ForCausalLM"),
        }

        if base_model_type not in MODEL_REGISTRY:
            logger.warning(f"Unknown base model type: {base_model_type}")
            return None

        module_name, class_name = MODEL_REGISTRY[base_model_type]

        try:
            # Import the module
            module = __import__(
                f'vllm.model_executor.models.{module_name}',
                fromlist=[class_name]
            )

            # Get the class
            model_class = getattr(module, class_name)
            return model_class

        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load {class_name} from {module_name}: {e}")
            return None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings from base model"""
        return self.llm.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Expose the standard vLLM embedding hook via the wrapped base model."""
        return self.llm.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """
        Forward pass through base model to get hidden states.

        This is called during prefill and decode phases.
        """
        hidden_states = self.llm.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute logits with dynamic temperature and top-p.

        This applies the AutoDeco heads to predict temperature and top_p
        for each token, then uses these for sampling.

        Args:
            hidden_states: Hidden states from forward pass
            sampling_metadata: Sampling parameters

        Returns:
            Tuple of (logits, temperatures, top_ps) when using AutoDeco heads,
            or just logits for standard models
        """
        # Use the logits processor with AutoDeco heads
        # This will call temp_head and top_p_head internally and return
        # (logits, temp, top_p) tuple
        result = self.logits_processor(
            lm_head=self.llm.lm_head,
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            temp_head=self.temp_head,
            top_p_head=self.top_p_head,
        )
        return result

    def compute_base_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        """Compute logits without running the AutoDeco heads."""
        return self.logits_processor(
            lm_head=self.llm.lm_head,
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            temp_head=None,
            top_p_head=None,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """
        Load weights for complete AutoDeco model.

        Expected weight structure (from merged checkpoint):
        - llm.* : Base model weights
        - temp_head.* : Temperature head weights
        - top_p_head.* : Top-p head weights

        Args:
            weights: Iterable of (name, tensor) pairs from checkpoint

        Returns:
            Set of loaded parameter names
        """
        logger.info("Loading AutoDeco weights from merged checkpoint...")
        from .utils import AutoWeightsLoader

        # Filter out head weights that are disabled in config.
        # This avoids load failures when checkpoints contain unused heads.
        filtered_weights: list[tuple[str, torch.Tensor]] = []
        skipped_temp = 0
        skipped_top_p = 0
        for name, tensor in weights:
            if name.startswith("temp_head.") and not self.enable_temperature_head:
                skipped_temp += 1
                continue
            if name.startswith("top_p_head.") and not self.enable_top_p_head:
                skipped_top_p += 1
                continue
            filtered_weights.append((name, tensor))
        if skipped_temp > 0:
            logger.warning(
                "Skipping %d temp_head.* weights (temperature head disabled).",
                skipped_temp,
            )
        if skipped_top_p > 0:
            logger.warning(
                "Skipping %d top_p_head.* weights (top-p head disabled).",
                skipped_top_p,
            )

        # Use AutoWeightsLoader to handle all weights automatically
        # It will match prefixes: llm.*, temp_head.*, top_p_head.*
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )

        loaded_params = loader.load_weights(filtered_weights)

        logger.info(f"✓ Successfully loaded {len(loaded_params)} parameters")

        # Log breakdown by component
        llm_params = sum(1 for p in loaded_params if p.startswith('llm.'))
        temp_head_params = sum(1 for p in loaded_params if p.startswith('temp_head.'))
        top_p_head_params = sum(1 for p in loaded_params if p.startswith('top_p_head.'))

        logger.info(f"  - Base model (llm.*): {llm_params} parameters")
        logger.info(f"  - Temperature head (temp_head.*): {temp_head_params} parameters")
        logger.info(f"  - Top-p head (top_p_head.*): {top_p_head_params} parameters")

        return loaded_params


# Export for vLLM model registry
__all__ = ['AutoDecoModelForCausalLM', 'TempHead', 'TopPHead']
