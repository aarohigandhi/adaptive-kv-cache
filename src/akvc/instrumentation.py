"""Measuring GPU memory and time.

reset_peak_memory and peak_memory_mb read the memory high water mark. cuda_timer
times a block of GPU work, syncing first so the number reflects real compute
rather than the time it took to queue the work.
"""

import time
from contextlib import contextmanager

import torch


def reset_peak_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_memory_mb():
    return torch.cuda.max_memory_allocated() / 1e6


@contextmanager
def cuda_timer():
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        torch.cuda.synchronize()
        result["seconds"] = time.perf_counter() - start


if __name__ == "__main__":
    reset_peak_memory()
    with cuda_timer() as t:
        x = torch.randn(4096, 4096, device="cuda")
        _ = x @ x
    print(f"Matmul time:     {t['seconds'] * 1000:.1f} ms")
    print(f"Peak GPU memory: {peak_memory_mb():.1f} MB")
