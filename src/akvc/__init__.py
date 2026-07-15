"""akvc — Adaptive KV Cache Compression.

Package layout:
    model.py           model + tokenizer loading and our manual decode loop
    cache_manager.py   owns the KV tensors; the evict()/stats() interface
    instrumentation.py memory / latency / attention measurement
    policies/          pluggable eviction policies (baselines + our method)
    eval/              scoring harness (needle, LongBench, perplexity)
"""
