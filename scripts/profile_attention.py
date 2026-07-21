"""Phase 2 -> 3 bridge: profile per-head attention entropy.

The novel-method hypothesis is "different heads deserve different budgets." This
script tests it: it runs the model over a real paragraph, then for every layer
and head measures the ENTROPY of that head's attention at the last position.

    low entropy  = focused head  (a few tokens carry the signal -> needs little cache)
    high entropy = diffuse head  (attention spread out          -> needs more cache)

If heads differ a lot, that's the evidence for an adaptive per-head budget.

Outputs:
    results/attention_entropy.json  raw [layers x heads] entropy grid
    results/attention_entropy.png   heatmap of the grid

Needs eager attention + float32. Run in Colab (GPU), from the repo root:
    !python scripts/profile_attention.py
"""

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model, build_inputs  # noqa: E402

# A content-rich prompt so there's real context to attend to.
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
    """Return a [layers x heads] tensor of attention entropy at the last token."""
    out = model(**inputs, use_cache=False, output_attentions=True)
    attentions = out.attentions  # tuple[layers] of [batch, heads, queries, keys]

    # Sanity check that the attention machinery actually gave us numbers.
    a0 = attentions[0]
    print(f"Captured attention: {len(attentions)} layers, per-layer shape {tuple(a0.shape)}")

    rows = []
    for a in attentions:
        last = a[0, :, -1, :].float()                       # [heads, keys]
        ent = -(last * last.clamp_min(1e-9).log()).sum(-1)  # entropy per head
        rows.append(ent)
    return torch.stack(rows)                                # [layers, heads]


def summarize(grid, n_keys):
    layers, heads = grid.shape
    max_possible = math.log(n_keys)  # entropy if attention were perfectly uniform
    print(f"\nGrid: {layers} layers x {heads} heads   (max possible entropy ~ {max_possible:.2f})")
    print(f"Mean entropy:        {grid.mean():.2f}")
    print(f"Most FOCUSED head:   entropy {grid.min():.2f} "
          f"at layer {grid.argmin() // heads}, head {grid.argmin() % heads}")
    print(f"Most DIFFUSE head:   entropy {grid.max():.2f} "
          f"at layer {grid.argmax() // heads}, head {grid.argmax() % heads}")
    print(f"Spread (max - min):  {grid.max() - grid.min():.2f}  "
          f"<- big spread => heads really do want different budgets")


def plot(grid, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid.cpu().numpy(), aspect="auto", cmap="viridis")
    ax.set_title("Attention entropy per head\n(dark = focused, bright = diffuse)")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    fig.colorbar(im, ax=ax, label="entropy (nats)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print("Saved", out_path)


def main():
    tokenizer, model = load_model(attn_implementation="eager", dtype=torch.float32)
    inputs = build_inputs(tokenizer, PROMPT)
    n_keys = inputs["input_ids"].shape[1]

    grid = entropy_grid(model, inputs)
    summarize(grid, n_keys)

    os.makedirs("results", exist_ok=True)
    with open("results/attention_entropy.json", "w") as f:
        json.dump(grid.cpu().tolist(), f)
    print("Saved results/attention_entropy.json")
    plot(grid, "results/attention_entropy.png")


if __name__ == "__main__":
    main()
