"""The shared Policy interface.

A policy answers one question, repeatedly: given how many tokens are currently
in the cache and a memory budget, which token positions do we KEEP (and, by
omission, which do we throw away)? Every baseline and our own method implement
this same tiny interface, so the eval harness can swap them in and out and the
comparison stays fair by construction.

Positions are plain integers 0, 1, 2, ... in the order tokens arrived. A policy
returns the sorted list of positions to keep.
"""

from typing import List, Optional


class Policy:
    """Base class every eviction policy inherits from."""

    name = "policy"

    # Smart policies (e.g. H2O) need the model's attention scores; simple ones
    # (full, StreamingLLM) don't. The decode loop checks this flag to decide
    # whether to collect attention (which is slower) during generation.
    needs_attention = False

    def keep_indices(
        self,
        num_tokens: int,
        budget: int,
        stats: Optional[dict] = None,
    ) -> List[int]:
        """Return the sorted token positions to keep.

        Args:
            num_tokens: how many tokens are in the cache right now.
            budget:     the most tokens we're allowed to keep.
            stats:      optional extra info (e.g. attention scores) that smarter
                        policies need; simple policies ignore it.
        """
        raise NotImplementedError
