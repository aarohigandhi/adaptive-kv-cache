"""Phase 3 tuning: search AdaptivePolicy hyperparameters on the needle task.

Your method has two knobs:
    recent_frac  -- share of the budget spent on the recent window vs the skeleton
    chunk_size   -- how many contiguous tokens each middle chunk keeps
This sweeps combinations at a couple of budgets and reports needle-retrieval
accuracy, so we can pick the setting that retrieves best (goal: beat SnapKV's
100% at budget 192). Runs on the fast float16 path.

Run in Colab (GPU), from the repo root:
    !python scripts/tune_adaptive.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model, build_inputs, decode_with_policy  # noqa: E402
from akvc.policies.adaptive import AdaptivePolicy                    # noqa: E402
from akvc.eval.needle import make_prompt, found, NEEDLES, DEPTHS     # noqa: E402

BUDGETS = [96, 192]
RECENT_FRACS = [0.25, 0.5, 0.75]
CHUNK_SIZES = [4, 8, 16]


def accuracy(model, tokenizer, policy, budget):
    hits = 0
    for needle in NEEDLES:
        for depth in DEPTHS:
            prompt = make_prompt(needle, depth)
            inputs = build_inputs(tokenizer, prompt)
            n_prompt = inputs["input_ids"].shape[1]
            ids = decode_with_policy(model, tokenizer, inputs, policy,
                                     budget=budget, max_new_tokens=16)
            ans = tokenizer.decode(ids[0, n_prompt:], skip_special_tokens=True)
            hits += found(ans, needle)
    return hits / (len(NEEDLES) * len(DEPTHS))


def main():
    tokenizer, model = load_model()  # fast float16 path (adaptive needs no attention)

    print(f"Tuning AdaptivePolicy | budgets {BUDGETS} | "
          f"{len(RECENT_FRACS)}x{len(CHUNK_SIZES)} configs\n")
    header = "  recent_frac  chunk_size  " + "  ".join(f"b{b}" for b in BUDGETS)
    print(header)

    best = None
    for rf in RECENT_FRACS:
        for cs in CHUNK_SIZES:
            policy = AdaptivePolicy(sinks=4, recent_frac=rf, chunk_size=cs)
            accs = [accuracy(model, tokenizer, policy, b) for b in BUDGETS]
            row = f"  {rf:>10}  {cs:>10}  " + "  ".join(f"{a:.0%}" for a in accs)
            print(row)
            score = accs[-1]  # rank by accuracy at the largest budget
            if best is None or score > best[0]:
                best = (score, rf, cs, accs)

    _, rf, cs, accs = best
    print(f"\nBest config: recent_frac={rf}, chunk_size={cs} "
          f"-> {dict(zip(BUDGETS, [f'{a:.0%}' for a in accs]))}")


if __name__ == "__main__":
    main()
