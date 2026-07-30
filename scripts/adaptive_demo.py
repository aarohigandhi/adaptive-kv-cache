"""Our method vs StreamingLLM on one needle. A code is hidden in the middle of a
long prompt and the budget is small enough that the middle must be evicted. Full
cache is the reference that should always retrieve it.

    python scripts/adaptive_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model, build_inputs, manual_decode, decode_with_policy
from akvc.policies.streaming_llm import StreamingLLMPolicy
from akvc.policies.adaptive import AdaptivePolicy

NEEDLE = "7391"
FILLER = ("The sea was calm and the sky was clear over the northern coast. "
          "The keeper walked the shore and noted the quiet tide. ")
PROMPT = (
    FILLER * 8
    + f"Important administrative note: the vault passcode is {NEEDLE}. "
    + FILLER * 8
    + "Question: what is the vault passcode? Answer with just the number."
)
BUDGET = 96
NEW_TOKENS = 24


def answer(tokenizer, ids, n_prompt):
    return tokenizer.decode(ids[0, n_prompt:], skip_special_tokens=True).strip()


def main():
    tokenizer, model = load_model()
    inputs = build_inputs(tokenizer, PROMPT)
    n_prompt = inputs["input_ids"].shape[1]
    print(f"Prompt length: {n_prompt} tokens | budget: {BUDGET} (the middle must be evicted)\n")

    full = manual_decode(model, tokenizer, inputs, max_new_tokens=NEW_TOKENS)
    stream = decode_with_policy(model, tokenizer, inputs, StreamingLLMPolicy(sinks=4),
                                budget=BUDGET, max_new_tokens=NEW_TOKENS)
    adapt = decode_with_policy(model, tokenizer, inputs, AdaptivePolicy(sinks=4),
                               budget=BUDGET, max_new_tokens=NEW_TOKENS)

    for label, ids in [("full cache", full), ("StreamingLLM", stream), ("Adaptive (yours)", adapt)]:
        text = answer(tokenizer, ids, n_prompt)
        status = "FOUND" if NEEDLE in text else "missed"
        print(f"[{label:>16}] {status} needle | answer: {text[:60]!r}")

    print(f"\nThe needle is {NEEDLE}. This is one example; run_eval.py runs the full sweep.")


if __name__ == "__main__":
    main()
