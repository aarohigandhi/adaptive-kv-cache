"""Measure the full cache baseline: how peak memory and decode latency grow with
context length. Saves the numbers and a two panel chart.

    python scripts/baseline_sweep.py
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model
from akvc.instrumentation import reset_peak_memory, peak_memory_mb, cuda_timer

CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192]
DECODE_STEPS = 16


@torch.no_grad()
def measure(model, n_tokens):
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, n_tokens), device="cuda")
    attn = torch.ones_like(input_ids)

    reset_peak_memory()
    with cuda_timer() as t_prefill:
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    with cuda_timer() as t_decode:
        for _ in range(DECODE_STEPS):
            attn = torch.cat([attn, torch.ones_like(next_token)], dim=1)
            out = model(input_ids=next_token, attention_mask=attn, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return {
        "context_length": n_tokens,
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "prefill_seconds": round(t_prefill["seconds"], 4),
        "decode_ms_per_token": round(t_decode["seconds"] / DECODE_STEPS * 1000, 2),
    }


def plot(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["context_length"] for r in results]
    panels = [
        ([r["peak_memory_mb"] for r in results], "Peak GPU memory grows with context", "peak memory (MB)", "#2a6fdb"),
        ([r["decode_ms_per_token"] for r in results], "Decode slows down with context", "latency (ms per token)", "#e8710a"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (ys, title, ylabel, color) in zip(axes, panels):
        ax.plot(xs, ys, marker="o", linewidth=2, color=color)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Full cache baseline", fontsize=13)
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
