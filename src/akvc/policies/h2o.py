"""H2O (Zhang et al.): keep the tokens that have drawn the most attention so far
(the heavy hitters) plus a recent window. Our version pools attention across heads
and layers into one score per token rather than tracking each head separately."""

import torch

from .base import Policy


class H2OPolicy(Policy):
    name = "h2o"
    needs_attention = True

    def __init__(self, recent=None):
        self.recent = recent

    def keep_indices(self, num_tokens, budget, stats=None):
        if num_tokens <= budget:
            return list(range(num_tokens))
        importance = stats["importance"]
        recent = self.recent if self.recent is not None else budget // 2
        recent = min(recent, budget)
        heavy_budget = budget - recent
        recent_positions = list(range(num_tokens - recent, num_tokens))
        older = num_tokens - recent
        top = []
        if heavy_budget > 0 and older > 0:
            k = min(heavy_budget, older)
            top = torch.topk(importance[:older], k).indices.tolist()
        return sorted(set(recent_positions + top))
