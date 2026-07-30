"""SnapKV (Li et al.): use the last few prompt tokens (the observation window) to
score which earlier tokens matter, keep those plus the window, and compress once
right after the prompt is read."""

import torch

from .base import Policy


class SnapKVPolicy(Policy):
    name = "snapkv"
    needs_attention = True

    def __init__(self, window=32):
        self.window = window

    def keep_indices(self, num_tokens, budget, stats=None):
        if num_tokens <= budget:
            return list(range(num_tokens))
        importance = stats["importance"]
        window = min(self.window, budget)
        keep_budget = budget - window
        window_positions = list(range(num_tokens - window, num_tokens))
        older = num_tokens - window
        top = []
        if keep_budget > 0 and older > 0:
            k = min(keep_budget, older)
            top = torch.topk(importance[:older], k).indices.tolist()
        return sorted(set(window_positions + top))
