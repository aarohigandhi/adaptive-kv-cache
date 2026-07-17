"""Measurement tools — honest numbers are the whole point.

    - peak GPU memory:   torch.cuda.max_memory_allocated
    - per-token latency: with proper CUDA synchronization
    - attention maps:    sampled, never stored in full (that explodes memory)

Run a quick self-test in Colab (with a GPU):
    !python src/akvc/instrumentation.py
"""

import time
from contextlib import contextmanager

import torch


# --- Memory -----------------------------------------------------------------

def reset_peak_memory():
    """Zero the GPU's 'high-water mark' so the next peak reading starts fresh."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_memory_mb():
    """Highest GPU memory used since the last reset, in megabytes."""
    return torch.cuda.max_memory_allocated() / 1e6


# --- Latency ----------------------------------------------------------------

@contextmanager
def cuda_timer():
    """Time a block of GPU work in seconds, accounting for the GPU running async.

    The GPU does work in the background, so a plain stopwatch would stop before
    the GPU actually finished -- giving a fake, too-fast time. torch.cuda.
    synchronize() waits for the GPU to catch up so we measure the real duration.

    Usage:
        with cuda_timer() as t:
            ... GPU work ...
        print(t["seconds"])
    """
    torch.cuda.synchronize()          # make sure earlier GPU work is done first
    start = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        torch.cuda.synchronize()      # wait for THIS block's GPU work to finish
        result["seconds"] = time.perf_counter() - start


if __name__ == "__main__":
    # Self-test: measure a big matrix multiply's time and memory.
    reset_peak_memory()
    with cuda_timer() as t:
        x = torch.randn(4096, 4096, device="cuda")
        _ = x @ x                     # a chunky GPU computation
    print(f"Matmul time:     {t['seconds'] * 1000:.1f} ms")
    print(f"Peak GPU memory: {peak_memory_mb():.1f} MB")
