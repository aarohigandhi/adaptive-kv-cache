"""The interface every eviction policy implements."""


class Policy:
    """Base class for eviction policies.

    Set needs_attention = True if keep_indices reads attention scores from stats.
    """

    name = "policy"
    needs_attention = False

    def keep_indices(self, num_tokens, budget, stats=None):
        """Return the sorted positions to keep, at most budget out of num_tokens."""
        raise NotImplementedError
