"""Measurement tools — honest numbers are the whole point.

    - peak GPU memory:   torch.cuda.max_memory_allocated
    - per-token latency: with proper CUDA synchronization
    - attention maps:    sampled, never stored in full (that explodes memory)

TODO (Phase 1): implement memory + latency timers and the attention sampler.
"""
