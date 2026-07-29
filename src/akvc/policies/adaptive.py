r"""OUR method (v1) — "Anchored Skeleton" adaptive compression.

The novelty hook: existing baselines apply one static heuristic uniformly.
StreamingLLM in particular keeps anchors + recent tokens but throws away the
*entire middle* -- so any fact buried mid-context is lost. Our method spends the
same budget more cleverly by also retaining a thin "skeleton" of the middle:

    positions:  [0 1 2 3 . . x . . . x . . . x . . . n-3 n-2 n-1]
                 \_anchors_/   \___ strided skeleton ___/  \_recent_/

    anchors  : first few tokens the model uses as an attention sink
    skeleton : every k-th middle token -> a coarse memory of the whole context
    recent   : the most recent tokens (local coherence)

This needs no attention scores, so it runs in fast float16 on the same decode
path as StreamingLLM. Later versions can make the split adaptive per layer/head
or score the skeleton by attention (that's the roadmap).
"""

from typing import List, Optional

from .base import Policy


class AdaptivePolicy(Policy):
    name = "adaptive"

    def __init__(self, sinks: int = 4, recent_frac: float = 0.25, chunk_size: int = 16):
        # Defaults tuned on the needle task (scripts/tune_adaptive.py): more budget
        # to the middle skeleton (recent_frac=0.25) in bigger chunks (16) retrieves
        # mid-context facts best, since the needle lives in the middle.
        self.sinks = sinks              # number of anchor tokens to always keep
        self.recent_frac = recent_frac  # fraction of the leftover budget for recency
        self.chunk_size = chunk_size    # skeleton is kept as contiguous chunks this big

    def keep_indices(
        self,
        num_tokens: int,
        budget: int,
        stats: Optional[dict] = None,
    ) -> List[int]:
        if num_tokens <= budget:
            return list(range(num_tokens))

        sinks = min(self.sinks, budget)
        remaining = budget - sinks
        recent = int(remaining * self.recent_frac)
        skeleton_budget = remaining - recent

        anchor_pos = list(range(sinks))
        recent_pos = list(range(num_tokens - recent, num_tokens))

        # Skeleton across the middle [sinks, num_tokens - recent): a handful of
        # evenly-spaced CHUNKS of contiguous tokens, so kept phrases stay readable.
        mid_start, mid_end = sinks, num_tokens - recent
        mid_len = mid_end - mid_start
        skeleton_pos: List[int] = []
        if skeleton_budget > 0 and mid_len > 0:
            chunk = min(self.chunk_size, skeleton_budget)
            n_chunks = max(1, skeleton_budget // chunk)
            for i in range(n_chunks):
                # center each chunk in its evenly-spaced slice of the middle
                center = int(mid_start + (i + 0.5) * mid_len / n_chunks)
                start = max(mid_start, min(center - chunk // 2, mid_end - chunk))
                skeleton_pos.extend(range(start, min(start + chunk, mid_end)))
            skeleton_pos = sorted(set(skeleton_pos))[:skeleton_budget]

        return sorted(set(anchor_pos + skeleton_pos + recent_pos))


if __name__ == "__main__":
    # Pure-logic demo (no GPU): 40 tokens, budget 12, 4 anchors.
    policy = AdaptivePolicy(sinks=4, recent_frac=0.5)
    kept = policy.keep_indices(num_tokens=40, budget=12)
    print("kept positions:", kept)
    print("count:", len(kept), "(<= budget 12)")
    print("has anchors [0-3]?  ", all(p in kept for p in range(4)))
    print("has middle samples? ", any(4 <= p < 36 for p in kept))
    print("has recent tail?    ", any(p >= 36 for p in kept))
