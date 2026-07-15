"""Eval harness — deterministic runner that ties everything together.

Loads a config, runs a (policy x budget x task) combination with fixed seeds,
and dumps results as JSON so the plots regenerate from raw data.

TODO (Phase 1): minimal runner; grows as tasks land.
"""
