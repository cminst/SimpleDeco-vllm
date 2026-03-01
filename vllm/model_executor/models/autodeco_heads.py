# SPDX-License-Identifier: Apache-2.0
"""Shared AutoDeco heads used by vLLM and transformers wrappers."""

import torch
from torch import nn


class TopPHead(nn.Module):
    """Top-P prediction head with enhanced features."""

    def __init__(self, hidden_size, vocab_size=None, use_enhanced_features=True):
        super().__init__()
        self.use_enhanced_features = use_enhanced_features
        input_dim = hidden_size + 1 + (4 if use_enhanced_features else 0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.vocab_size = vocab_size

    def compute_prob_stats(self, logits):
        """Compute probability distribution statistics."""
        probs = torch.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True)[0]
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        prob_var = probs.var(dim=-1, keepdim=True)
        top5_probs, _ = torch.topk(probs, min(5, probs.size(-1)), dim=-1)
        top5_sum = top5_probs.sum(dim=-1, keepdim=True)
        return torch.cat([max_prob, entropy, prob_var, top5_sum], dim=-1)

    def forward(self, hidden_states, temp_logits, unscaled_logits=None):
        if self.use_enhanced_features:
            features = [hidden_states, temp_logits]
            if unscaled_logits is not None:
                scaled_logits = unscaled_logits / (temp_logits + 1e-8)
                prob_stats = self.compute_prob_stats(scaled_logits)
            else:
                batch_size, seq_len = hidden_states.shape[:2]
                prob_stats = torch.zeros(
                    batch_size,
                    seq_len,
                    4,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            features.append(prob_stats)
            combined_features = torch.cat(features, dim=-1)
        else:
            combined_features = torch.cat([hidden_states, temp_logits], dim=-1)
        return self.mlp(combined_features)


class TempHead(nn.Module):
    """Temperature prediction head"""

    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states):
        sigmoid_output = self.mlp(hidden_states)
        return sigmoid_output * 2

# class TempHead(nn.Module):
#     """Temperature prediction head."""

#     def __init__(self, hidden_size):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_size, 1024),
#             nn.GELU(),
#             nn.Linear(1024, 512),
#             nn.GELU(),
#             nn.Linear(512, 256),
#             nn.GELU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, hidden_states):
#         sigmoid_output = self.mlp(hidden_states)
#         return sigmoid_output * 2


__all__ = ["TempHead", "TopPHead"]
