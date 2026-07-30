"""Query aware skeleton (adaptive_qa) vs snapkv vs adaptive on the needle task.
Tests whether steering the middle chunks toward high attention tokens closes the
gap to SnapKV.

    python scripts/adaptive_qa_demo.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model
from akvc.eval.harness import run_policy
from akvc.eval.needle import make_prompt, found, NEEDLES, DEPTHS

BUDGETS = [96, 192]
METHODS = ["snapkv", "adaptive", "adaptive_qa"]


def accuracy(model, tokenizer, method, budget):
    hits = 0
    for needle in NEEDLES:
        for depth in DEPTHS:
            ans = run_policy(model, tokenizer, make_prompt(needle, depth), method, budget)
            hits += found(ans, needle)
    return hits / (len(NEEDLES) * len(DEPTHS))


def main():
    tokenizer, model = load_model(attn_implementation="eager", dtype=torch.float32)
    print(f"Needle accuracy | budgets {BUDGETS}\n")
    print(f"  {'method':>12}  " + "  ".join(f"b{b}" for b in BUDGETS))
    for method in METHODS:
        accs = [accuracy(model, tokenizer, method, b) for b in BUDGETS]
        print(f"  {method:>12}  " + "  ".join(f"{a:.0%}" for a in accs))


if __name__ == "__main__":
    main()
