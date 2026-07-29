"""Phase 3: query-aware skeleton (adaptive_qa) vs snapkv vs adaptive.

Tests whether steering the skeleton chunks toward high-attention tokens (instead
of even spacing) closes the gap to SnapKV on needle retrieval. Runs the needle
set at a couple of budgets and prints accuracy per method.

Uses the eager/float32 model (adaptive_qa and snapkv read attention). Run in
Colab (GPU), from the repo root:
    !python scripts/adaptive_qa_demo.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model                              # noqa: E402
from akvc.eval.harness import run_policy                       # noqa: E402
from akvc.eval.needle import make_prompt, found, NEEDLES, DEPTHS  # noqa: E402

BUDGETS = [96, 192]
METHODS = ["snapkv", "adaptive", "adaptive_qa"]


def accuracy(model, tokenizer, method, budget):
    hits = 0
    for needle in NEEDLES:
        for depth in DEPTHS:
            prompt = make_prompt(needle, depth)
            ans = run_policy(model, tokenizer, prompt, method, budget)
            hits += found(ans, needle)
    return hits / (len(NEEDLES) * len(DEPTHS))


def main():
    tokenizer, model = load_model(attn_implementation="eager", dtype=torch.float32)
    print(f"Needle accuracy | budgets {BUDGETS} | {len(NEEDLES)}x{len(DEPTHS)} trials\n")
    print(f"  {'method':>12}  " + "  ".join(f"b{b}" for b in BUDGETS))
    for method in METHODS:
        accs = [accuracy(model, tokenizer, method, b) for b in BUDGETS]
        print(f"  {method:>12}  " + "  ".join(f"{a:.0%}" for a in accs))


if __name__ == "__main__":
    main()
