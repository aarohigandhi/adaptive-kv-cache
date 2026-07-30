"""StreamingLLM (Xiao et al.): keep a few sink tokens at the start plus a window
of recent tokens, and drop the middle."""

from .base import Policy


class StreamingLLMPolicy(Policy):
    name = "streaming_llm"

    def __init__(self, sinks=4):
        self.sinks = sinks

    def keep_indices(self, num_tokens, budget, stats=None):
        if num_tokens <= budget:
            return list(range(num_tokens))
        sinks = min(self.sinks, budget)
        recent = budget - sinks
        keep = list(range(sinks)) + list(range(num_tokens - recent, num_tokens))
        return sorted(set(keep))


if __name__ == "__main__":
    kept = StreamingLLMPolicy().keep_indices(20, 8)
    print("kept:", kept, "count:", len(kept))
