"""The cache manager — the heart of the project.

Owns the KV tensors (per layer, per head) and exposes a small, clean interface:
    - stats():          summary the policy needs to decide what to keep
    - evict(indices):   drop the chosen tokens from the cache
This is the component interviewers will probe, so the interface comes first,
before any policy is written.

TODO (Phase 1): design and implement the CacheManager interface.
"""
