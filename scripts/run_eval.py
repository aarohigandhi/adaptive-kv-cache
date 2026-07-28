"""Phase 4: headline evaluation — needle retrieval vs cache budget.

For every policy and several cache budgets, we run the needle test across a few
needles x depths and record retrieval accuracy. The result is an accuracy-vs-
budget curve per policy: it shows exactly where each method breaks and recovers.

Saved to results/eval_needle.json and results/needle_accuracy.png.

All policies run on one eager/float32 model. Run in Colab (GPU), from repo root:
    !python scripts/run_eval.py
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model                          # noqa: E402
from akvc.eval.harness import run_policy, POLICIES         # noqa: E402
from akvc.eval.needle import make_prompt, found, NEEDLES   # noqa: E402

BUDGETS = [48, 96, 192, 288]
DEPTHS = [0.1, 0.5, 0.9]

# fixed colors per policy (our method highlighted); baselines muted-but-distinct
COLORS = {
    "full": "#9aa4b2", "streaming_llm": "#2a6fdb", "h2o": "#e8710a",
    "snapkv": "#2ca02c", "adaptive": "#d62728",
}


def run_sweep(model, tokenizer):
    records = []
    for policy in POLICIES:
        line = [f"  {policy:>14}:"]
        for budget in BUDGETS:
            hits = 0
            for needle in NEEDLES:
                for depth in DEPTHS:
                    prompt = make_prompt(needle, depth)
                    ans = run_policy(model, tokenizer, prompt, policy, budget)
                    hit = found(ans, needle)
                    hits += hit
                    records.append({"policy": policy, "budget": budget,
                                    "needle": needle, "depth": depth, "found": hit})
            acc = hits / (len(NEEDLES) * len(DEPTHS))
            line.append(f"b{budget}={acc:.0%}")
        print("  ".join(line))
    return records


def accuracy(records, policy, budget):
    rs = [r for r in records if r["policy"] == policy and r["budget"] == budget]
    return sum(r["found"] for r in rs) / len(rs) if rs else 0.0


def plot(records, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for policy in POLICIES:
        ys = [accuracy(records, policy, b) * 100 for b in BUDGETS]
        lw = 2.8 if policy == "adaptive" else 1.8
        ax.plot(BUDGETS, ys, marker="o", linewidth=lw,
                color=COLORS[policy], label=policy)

    ax.set_title("Needle retrieval vs cache budget")
    ax.set_xlabel("cache budget (tokens kept)")
    ax.set_ylabel("retrieval accuracy (%)")
    ax.set_ylim(-3, 103)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print("Saved", out_path)


def main():
    tokenizer, model = load_model(attn_implementation="eager", dtype=torch.float32)
    print(f"Needle sweep: {len(POLICIES)} policies x {len(BUDGETS)} budgets "
          f"x {len(NEEDLES)} needles x {len(DEPTHS)} depths "
          f"= {len(POLICIES)*len(BUDGETS)*len(NEEDLES)*len(DEPTHS)} runs\n")

    records = run_sweep(model, tokenizer)

    os.makedirs("results", exist_ok=True)
    with open("results/eval_needle.json", "w") as f:
        json.dump(records, f, indent=2)
    print("\nSaved results/eval_needle.json")
    plot(records, "results/needle_accuracy.png")


if __name__ == "__main__":
    main()
