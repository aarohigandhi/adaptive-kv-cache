"""The shared Policy interface.

A policy answers one question, repeatedly: given the current cache stats and a
memory budget, which tokens do we keep and which do we evict? Baselines and our
method all implement this same interface so the eval harness can swap them.

TODO (Phase 2): define the Policy base class / interface.
"""
