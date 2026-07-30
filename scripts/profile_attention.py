"""Measure attention entropy per head. Low entropy means a head focuses on a few
tokens, high entropy means it spreads its attention out. A big spread across heads
is the case for giving different heads different budgets.

    python scripts/profile_attention.py
"""

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model, build_inputs

PROMPT = (
    "The lighthouse keeper recorded the weather every morning: the wind speed, "
    "the tide height, the temperature, and the number of ships that passed. "
    "Over the years these logs became a detailed history of the coast. One "
    "winter, a storm knocked out the lamp, and he had to climb the tower in the "
    "dark to relight it by hand. Describe how he felt afterwards, and why the "
    "logs mattered to the town."
)


@torch.no_grad()
def entropy_grid(model, inputs):
    out = model(**inputs, use_cache=False, output_attentions=True)
    attentions = out.attentions
    print(f"Captured attention: {len(attentions)} layers, shape {tuple(attentions[0].shape)}")
    rows = []
    for a in attentions:
        last = a[0, :, -1, :].float()
        ent = -(last * last.clamp_min(1e-9).log()).sum(-1)
        rows.append(ent)
    return torch.stack(rows)


def summarize(grid, n_keys):
    layers, heads = grid.shape
    print(f"\nGrid: {layers} layers by {heads} heads (max entropy about {math.log(n_keys):.2f})")
    print(f"Mean entropy:      {grid.mean():.2f}")
    print(f"Most focused head: {grid.min():.2f} at layer {grid.argmin() // heads}, head {grid.argmin() % heads}")
    print(f"Most diffuse head: {grid.max():.2f} at layer {grid.argmax() // heads}, head {grid.argmax() % heads}")
    print(f"Spread:            {grid.max() - grid.min():.2f}")


def plot(grid, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid.cpu().numpy(), aspect="auto", cmap="viridis")
    ax.set_title("Attention entropy per head\n(dark is focused, bright is diffuse)")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    fig.colorbar(im, ax=ax, label="entropy (nats)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print("Saved", out_path)


def main():
    tokenizer, model = load_model(attn_implementation="eager", dtype=torch.float32)
    inputs = build_inputs(tokenizer, PROMPT)
    grid = entropy_grid(model, inputs)
    summarize(grid, inputs["input_ids"].shape[1])
    os.makedirs("results", exist_ok=True)
    with open("results/attention_entropy.json", "w") as f:
        json.dump(grid.cpu().tolist(), f)
    print("Saved results/attention_entropy.json")
    plot(grid, "results/attention_entropy.png")


if __name__ == "__main__":
    main()
