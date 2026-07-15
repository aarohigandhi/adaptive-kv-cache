"""Baseline: H2O — Heavy-Hitter Oracle (Zhang et al.).

Keep the "heavy hitter" tokens that have accumulated the most attention over
time, plus recent tokens; evict the rest.

TODO (Phase 2): implement cumulative-attention heavy-hitter eviction.
"""
