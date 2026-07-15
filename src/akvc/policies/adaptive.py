"""OUR method — adaptive KV cache compression.

The novelty hook: existing baselines apply one static heuristic uniformly. We
allocate the memory budget *adaptively* across heads, layers, and decode phases.
Exact direction (entropy-guided per-head budgets / layer-wise pyramid / hybrid
score with recovery) is chosen after the Phase 2 profiling study.

TODO (Phase 3): implement the adaptive budget allocation policy.
"""
