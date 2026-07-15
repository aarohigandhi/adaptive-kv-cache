"""Baseline: SnapKV (Li et al.).

At prefill, use an "observation window" of the last prompt tokens to score
which earlier tokens matter, then keep only those. The moderate baseline.

TODO (Phase 2): implement observation-window attention selection.
"""
