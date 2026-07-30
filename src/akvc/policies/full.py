"""Full cache baseline: keep everything, evict nothing. The reference point."""

from .base import Policy


class FullCachePolicy(Policy):
    name = "full"

    def keep_indices(self, num_tokens, budget, stats=None):
        return list(range(num_tokens))
