"""Baseline: SnapKV (Li et al.).

The rule: at the end of the prompt, use an "observation window" (the last few
prompt tokens) to judge which earlier tokens matter -- keep those plus the
window, drop the rest. Compression happens once, right after prefill.

Unlike H2O (which tallies attention over the whole generation), SnapKV scores
tokens using only the observation window's attention, computed at prefill. The
keep rule below is the same shape either way; what differs is where the
`importance` scores come from (see scripts/snapkv_demo.py).
"""

from typing import List, Optional

import torch

from .base import Policy


class SnapKVPolicy(Policy):
    name = "snapkv"
    needs_attention = True  # scores come from observation-window attention

    def __init__(self, window: int = 32):
        self.window = window  # size of the observation window (always kept)

    def keep_indices(
        self,
        num_tokens: int,
        budget: int,
        stats: Optional[dict] = None,
    ) -> List[int]:
        if num_tokens <= budget:
            return list(range(num_tokens))

        importance = stats["importance"]  # one score per token position

        window = min(self.window, budget)
        keep_budget = budget - window

        # Always keep the observation window (the most recent tokens).
        window_positions = list(range(num_tokens - window, num_tokens))

        # From the older tokens, keep the highest-scoring `keep_budget`.
        older_count = num_tokens - window
        if keep_budget > 0 and older_count > 0:
            older_importance = importance[:older_count]
            k = min(keep_budget, older_count)
            top = torch.topk(older_importance, k).indices.tolist()
        else:
            top = []

        return sorted(set(window_positions + top))
