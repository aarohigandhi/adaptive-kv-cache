"""Baseline: StreamingLLM (Xiao et al.).

Keep a few "attention sink" tokens at the very start plus a sliding window of
the most recent tokens; evict the middle. The easy baseline to reproduce first.

TODO (Phase 2): implement sink + recency-window eviction.
"""
