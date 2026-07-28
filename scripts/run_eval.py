"""Phase 4: the headline evaluation — needle retrieval across all policies.

For every policy, every needle, and every depth, we compress to a fixed budget
and check whether the model can still retrieve the hidden code. Results are
saved to results/eval_needle.json and summarized in results/needle_accuracy.png.

All policies run on one eager/float32 model (needed by H2O/SnapKV; the others
are fine on it too). Run in Colab (GPU), from the repo root:
    !python scripts/run_eval.py
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model                                   # noqa: E402
from akvc.eval.harness import run_policy, POLICIES                  # noqa: E402
from akvc.eval.needle import make_prompt, found, NEEDLES, DEPTHS    # noqa: E402

BUDGET = 96


def run_sweep(model, tokenizer):
    records = []
    for policy in POLICIES:
        for needle in NEEDLES:
            for depth in DEPTHS:
                prompt = make_prompt(needle, depth)
                answer = run_policy(model, tokenizer, prompt, policy, BUDGET)
                hit = found(answer, needle)
                records.append({
                    "policy": policy, "needle": needle, "depth": depth,
                    "found": hit, "answer": answer[:40],
                })
        acc = sum(r["found"] for r in records if r["policy"] == policy) / (len(NEEDLES) * len(DEPTHS))
        print(f"  {policy:>14}: needle accuracy {acc:.0%}")
    return records


def accuracy_by_policy(records):
    accs = {}
    for policy in POLICIES:
        rs = [r for r in records if r["policy"] == policy]
        accs[policy] = sum(r["found"] for r in rs) / len(rs)
    return accs


def plot(accs, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(accs.keys())
    values = [accs[p] * 100 for p in labels]
    # highlight our method; baselines share one muted color
    colors = ["#2a6fdb" if p == "adaptive" else "#9aa4b2" for p in labels]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(labels, values, color=colors)
    ax.set_title(f"Needle retrieval under a {BUDGET}-token cache budget")
    ax.set_ylabel("retrieval accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(values):          # direct labels
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print("Saved", out_path)


def main():
    tokenizer, model = load_model(attn_implementation="eager", dtype=torch.float32)
    print(f"Running needle sweep at budget {BUDGET} "
          f"({len(POLICIES)} policies x {len(NEEDLES)} needles x {len(DEPTHS)} depths)...\n")

    records = run_sweep(model, tokenizer)

    os.makedirs("results", exist_ok=True)
    with open("results/eval_needle.json", "w") as f:
        json.dump(records, f, indent=2)
    print("\nSaved results/eval_needle.json")

    accs = accuracy_by_policy(records)
    plot(accs, "results/needle_accuracy.png")

    winner = max(accs, key=accs.get)
    print(f"\nBest policy under budget {BUDGET}: {winner} ({accs[winner]:.0%})")


if __name__ == "__main__":
    main()
