"""akvc: adaptive KV cache compression.

model.py loads the model and runs the decode loops. cache_manager.py trims the
cache. instrumentation.py measures memory and speed. policies/ holds the eviction
rules (the baselines and our method). eval/ scores them on the needle task.
"""
