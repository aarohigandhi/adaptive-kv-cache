"""Phase 1: sweep context length and measure the full-cache baseline.

This produces the project's "before" picture: how peak GPU memory and decode
latency climb as the prompt gets longer. It saves the raw numbers to
results/baseline_sweep.json and a two-panel chart to results/baseline_sweep.png.

Run in Colab (with a GPU), from the repo root:
    !python scripts/baseline_sweep.py
"""

import json
import os
import sys

import torch

# Make the src/ package importable when this file is run as a plain script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model                                    # noqa: E402
from akvc.instrumentation import (                                   # noqa: E402
    reset_peak_memory,
    peak_memory_mb,
    cuda_timer,
)

# Prompt lengths to test. Kept <= 8192 so it runs comfortably on a free T4.
CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192]
DECODE_STEPS = 16  # generate this many tokens to get an average decode speed


@torch.no_grad()
def measure(model, n_tokens):
    """Measure prefill time, decode speed, and peak memory for one prompt length.

    We use random token ids to hit an exact length -- for memory/speed we don't
    care what the text says, only how long it is.
    """
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, n_tokens), device="cuda")
    attn = torch.ones_like(input_ids)

    # --- Prefill: build the cache for the whole prompt, timed + memory-tracked
    reset_peak_memory()
    with cuda_timer() as t_prefill:
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # --- Decode: time a handful of one-token steps and average them
    with cuda_timer() as t_decode:
        for _ in range(DECODE_STEPS):
            attn = torch.cat([attn, torch.ones_like(next_token)], dim=1)
            out = model(
                input_ids=next_token,
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return {
        "context_length": n_tokens,
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "prefill_seconds": round(t_prefill["seconds"], 4),
        "decode_ms_per_token": round(t_decode["seconds"] / DECODE_STEPS * 1000, 2),
    }


def plot(results, out_path):
    """Two panels (memory | latency) -- different units never share one axis."""
    import matplotlib
    matplotlib.use("Agg")  # no screen in Colab; just save a file
    import matplotlib.pyplot as plt

    xs = [r["context_length"] for r in results]
    mem = [r["peak_memory_mb"] for r in results]
    dec = [r["decode_ms_per_token"] for r in results]

    fig, (ax_mem, ax_lat) = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, ys, title, ylabel, color in [
        (ax_mem, mem, "Peak GPU memory grows with context", "peak memory (MB)", "#2a6fdb"),
        (ax_lat, dec, "Decode slows down with context", "latency (ms / token)", "#e8710a"),
    ]:
        ax.plot(xs, ys, marker="o", linewidth=2, color=color)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)          # recessive grid
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Full-cache baseline — the 'before' picture", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print("Saved", out_path)


def main():
    _, model = load_model()

    results = []
    for n in CONTEXT_LENGTHS:
        r = measure(model, n)
        print(r)
        results.append(r)

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results/baseline_sweep.json")

    plot(results, "results/baseline_sweep.png")


if __name__ == "__main__":
    main()
