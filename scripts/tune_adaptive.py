"""Search AdaptivePolicy hyperparameters (recent_frac and chunk_size) on the
needle task and report accuracy per config, to find the setting that retrieves
best.

    python scripts/tune_adaptive.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model, build_inputs, decode_with_policy
from akvc.policies.adaptive import AdaptivePolicy
from akvc.eval.needle import make_prompt, found, NEEDLES, DEPTHS

BUDGETS = [96, 192]
RECENT_FRACS = [0.25, 0.5, 0.75]
CHUNK_SIZES = [4, 8, 16]


def accuracy(model, tokenizer, policy, budget):
    hits = 0
    for needle in NEEDLES:
        for depth in DEPTHS:
            inputs = build_inputs(tokenizer, make_prompt(needle, depth))
            n_prompt = inputs["input_ids"].shape[1]
            ids = decode_with_policy(model, tokenizer, inputs, policy, budget=budget, max_new_tokens=16)
            ans = tokenizer.decode(ids[0, n_prompt:], skip_special_tokens=True)
            hits += found(ans, needle)
    return hits / (len(NEEDLES) * len(DEPTHS))


def main():
    tokenizer, model = load_model()
    print(f"Tuning AdaptivePolicy | budgets {BUDGETS} | {len(RECENT_FRACS)} by {len(CHUNK_SIZES)} configs\n")
    print("  recent_frac  chunk_size  " + "  ".join(f"b{b}" for b in BUDGETS))

    best = None
    for rf in RECENT_FRACS:
        for cs in CHUNK_SIZES:
            policy = AdaptivePolicy(sinks=4, recent_frac=rf, chunk_size=cs)
            accs = [accuracy(model, tokenizer, policy, b) for b in BUDGETS]
            print(f"  {rf:>10}  {cs:>10}  " + "  ".join(f"{a:.0%}" for a in accs))
            if best is None or accs[-1] > best[0]:
                best = (accs[-1], rf, cs, accs)

    _, rf, cs, accs = best
    print(f"\nBest config: recent_frac={rf}, chunk_size={cs} -> {dict(zip(BUDGETS, [f'{a:.0%}' for a in accs]))}")


if __name__ == "__main__":
    main()
