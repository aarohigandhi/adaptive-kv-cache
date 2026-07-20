"""Baseline: H2O — Heavy-Hitter Oracle (Zhang et al.).

Keep the "heavy hitter" tokens that have accumulated the most attention over
time, plus a window of recent tokens; evict the rest. Unlike StreamingLLM
(which keeps tokens by *position*), H2O keeps tokens by *importance*.

We use a simplified, single-budget variant: attention is aggregated across
heads and layers into one importance score per token, and the same positions
are evicted everywhere. (True H2O tracks heavy hitters per head.)
"""

from typing import List, Optional

import torch

from .base import Policy


class H2OPolicy(Policy):
    name = "h2o"
    needs_attention = True  # tells the decode loop to collect attention scores

    def __init__(self, recent: Optional[int] = None):
        # size of the always-keep recent window; defaults to half the budget
        self.recent = recent

    def keep_indices(
        self,
        num_tokens: int,
        budget: int,
        stats: Optional[dict] = None,
    ) -> List[int]:
        if num_tokens <= budget:
            return list(range(num_tokens))

        importance = stats["importance"]  # 1D tensor, one score per token position

        recent = self.recent if self.recent is not None else budget // 2
        recent = min(recent, budget)
        heavy_budget = budget - recent

        # Always keep the most recent `recent` tokens.
        recent_positions = list(range(num_tokens - recent, num_tokens))

        # From the older tokens, keep the `heavy_budget` highest-attention ones.
        older_count = num_tokens - recent
        if heavy_budget > 0 and older_count > 0:
            older_importance = importance[:older_count]
            k = min(heavy_budget, older_count)
            top = torch.topk(older_importance, k).indices.tolist()
        else:
            top = []

        return sorted(set(recent_positions + top))
