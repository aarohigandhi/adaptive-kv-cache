"""Baseline: full cache — keep every token, evict nothing.

The reference point. All quality/memory/latency numbers are measured relative
to this. Simplest possible policy: it ignores the budget and keeps everything.
"""

from typing import List, Optional

from .base import Policy


class FullCachePolicy(Policy):
    name = "full"

    def keep_indices(
        self,
        num_tokens: int,
        budget: int,
        stats: Optional[dict] = None,
    ) -> List[int]:
        return list(range(num_tokens))  # keep them all
