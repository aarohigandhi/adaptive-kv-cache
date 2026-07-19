r"""Baseline: StreamingLLM (Xiao et al.).

The rule: keep a few "attention sink" tokens at the very start, plus a sliding
window of the most recent tokens; throw away the middle. It's the easy baseline
and a surprisingly strong one -- those first few tokens act as an anchor the
model relies on, so keeping them matters a lot.

    positions:  [0 1 2 3 ............................ n-3 n-2 n-1]
                 \___sinks___/         (evicted)      \__recent__/
"""

from typing import List, Optional

from .base import Policy


class StreamingLLMPolicy(Policy):
    name = "streaming_llm"

    def __init__(self, sinks: int = 4):
        self.sinks = sinks  # how many earliest "anchor" tokens to always keep

    def keep_indices(
        self,
        num_tokens: int,
        budget: int,
        stats: Optional[dict] = None,
    ) -> List[int]:
        # If everything already fits under budget, keep it all.
        if num_tokens <= budget:
            return list(range(num_tokens))

        sinks = min(self.sinks, budget)      # don't let sinks exceed the budget
        recent = budget - sinks              # the rest of the budget = recent window

        keep = list(range(sinks))                          # first `sinks` tokens
        keep += list(range(num_tokens - recent, num_tokens))  # last `recent` tokens
        return sorted(set(keep))             # sorted, de-duplicated positions


if __name__ == "__main__":
    # Quick demo (pure logic, no GPU): 20 tokens, budget 8, 4 sink tokens.
    policy = StreamingLLMPolicy(sinks=4)
    kept = policy.keep_indices(num_tokens=20, budget=8)
    print("kept positions:", kept)
    print("kept count:", len(kept), "(should be <= budget of 8)")
