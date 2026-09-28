"""Measuring memory and time.

reset_peak_memory and peak_memory_mb read the memory high water mark. timer times
a block of work, syncing the GPU first so the number reflects real compute rather
than the time it took to queue the work. On CPU the sync is a no-op and the memory
readings come back as 0.0, so the same code paths run in CI.
"""

import time
from contextlib import contextmanager

import torch


def _cuda():
    return torch.cuda.is_available()


def reset_peak_memory():
    if _cuda():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb():
    return torch.cuda.max_memory_allocated() / 1e6 if _cuda() else 0.0


@contextmanager
def timer():
    """Time a block. Synchronizes the GPU on both ends when there is one."""
    if _cuda():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        if _cuda():
            torch.cuda.synchronize()
        result["seconds"] = time.perf_counter() - start



def kv_cache_bytes(past):
    """Bytes held by the KV cache itself, which is what a policy actually shrinks.

    Peak process memory also carries the weights and activations, so it moves very
    little when the cache is trimmed. This counts only the key and value tensors.
    """
    total = 0
    if hasattr(past, "layers"):
        for layer in past.layers:
            total += layer.keys.numel() * layer.keys.element_size()
            total += layer.values.numel() * layer.values.element_size()
    elif hasattr(past, "key_cache"):
        for k, v in zip(past.key_cache, past.value_cache):
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


if __name__ == "__main__":
    reset_peak_memory()
    with timer() as t:
        x = torch.randn(2048, 2048)
        _ = x @ x
    print(f"Matmul time: {t['seconds'] * 1000:.1f} ms")
    print(f"Peak GPU memory: {peak_memory_mb():.1f} MB")
