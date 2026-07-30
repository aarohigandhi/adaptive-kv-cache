"""Our method: Anchored Skeleton.

The baselines keep a fixed shape of the cache. StreamingLLM keeps the start and
the recent tokens but drops the whole middle, so a fact buried in the middle is
gone. We split the same budget three ways: a few anchor tokens at the start, a
recent window at the end, and a thin skeleton of evenly spaced chunks through the
middle, so a mid context fact can survive.

AdaptiveQAPolicy is the query aware version. Instead of spacing the middle chunks
evenly, it puts them where attention is highest.
"""

import torch

from .base import Policy


class AdaptivePolicy(Policy):
    name = "adaptive"

    def __init__(self, sinks=4, recent_frac=0.25, chunk_size=16):
        self.sinks = sinks
        self.recent_frac = recent_frac
        self.chunk_size = chunk_size

    def keep_indices(self, num_tokens, budget, stats=None):
        if num_tokens <= budget:
            return list(range(num_tokens))

        sinks = min(self.sinks, budget)
        remaining = budget - sinks
        recent = int(remaining * self.recent_frac)
        skeleton_budget = remaining - recent

        anchors = list(range(sinks))
        recent_pos = list(range(num_tokens - recent, num_tokens))

        mid_start, mid_end = sinks, num_tokens - recent
        mid_len = mid_end - mid_start
        skeleton = []
        if skeleton_budget > 0 and mid_len > 0:
            chunk = min(self.chunk_size, skeleton_budget)
            n_chunks = max(1, skeleton_budget // chunk)
            for i in range(n_chunks):
                center = int(mid_start + (i + 0.5) * mid_len / n_chunks)
                start = max(mid_start, min(center - chunk // 2, mid_end - chunk))
                skeleton.extend(range(start, min(start + chunk, mid_end)))
            skeleton = sorted(set(skeleton))[:skeleton_budget]

        return sorted(set(anchors + skeleton + recent_pos))


class AdaptiveQAPolicy(Policy):
    name = "adaptive_qa"
    needs_attention = True

    def __init__(self, sinks=4, recent_frac=0.25, chunk_size=16):
        self.sinks = sinks
        self.recent_frac = recent_frac
        self.chunk_size = chunk_size

    def keep_indices(self, num_tokens, budget, stats=None):
        if num_tokens <= budget:
            return list(range(num_tokens))

        importance = stats["importance"]
        sinks = min(self.sinks, budget)
        remaining = budget - sinks
        recent = int(remaining * self.recent_frac)
        skeleton_budget = remaining - recent

        anchors = list(range(sinks))
        recent_pos = list(range(num_tokens - recent, num_tokens))

        mid_start, mid_end = sinks, num_tokens - recent
        skeleton = set()
        if skeleton_budget > 0 and mid_end > mid_start:
            chunk = min(self.chunk_size, skeleton_budget)
            order = torch.argsort(importance[mid_start:mid_end], descending=True).tolist()
            for local in order:
                if len(skeleton) >= skeleton_budget:
                    break
                center = mid_start + local
                start = max(mid_start, min(center - chunk // 2, mid_end - chunk))
                for p in range(start, min(start + chunk, mid_end)):
                    if len(skeleton) >= skeleton_budget:
                        break
                    skeleton.add(p)

        return sorted(set(anchors) | skeleton | set(recent_pos))


if __name__ == "__main__":
    kept = AdaptivePolicy().keep_indices(40, 12)
    print("kept:", kept, "count:", len(kept))
