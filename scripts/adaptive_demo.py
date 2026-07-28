"""Phase 3: your method (Anchored Skeleton) vs StreamingLLM on a needle test.

A secret code is hidden in the MIDDLE of a long prompt, then we ask for it, with
a budget small enough that the middle must be evicted. StreamingLLM throws the
whole middle away; your method keeps a strided skeleton of it, so it has a chance
to retain the needle. Full cache is shown as the "should definitely get it"
reference.

This runs on the fast float16 path (no attention needed). Run in Colab (GPU):
    !python scripts/adaptive_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import (                                    # noqa: E402
    load_model, build_inputs, manual_decode, decode_with_policy,
)
from akvc.policies.streaming_llm import StreamingLLMPolicy  # noqa: E402
from akvc.policies.adaptive import AdaptivePolicy           # noqa: E402

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
    tokenizer, model = load_model()  # fast float16 path
    inputs = build_inputs(tokenizer, PROMPT)
    n_prompt = inputs["input_ids"].shape[1]
    print(f"Prompt length: {n_prompt} tokens | budget: {BUDGET} "
          f"(so the middle MUST be evicted)\n")

    # Full cache reference
    full = manual_decode(model, tokenizer, inputs, max_new_tokens=NEW_TOKENS)
    a_full = answer(tokenizer, full, n_prompt)

    # StreamingLLM (drops the whole middle)
    s_ids = decode_with_policy(model, tokenizer, inputs, StreamingLLMPolicy(sinks=4),
                               budget=BUDGET, max_new_tokens=NEW_TOKENS)
    a_stream = answer(tokenizer, s_ids, n_prompt)

    # Your method (keeps a skeleton of the middle)
    a_ids = decode_with_policy(model, tokenizer, inputs, AdaptivePolicy(sinks=4),
                               budget=BUDGET, max_new_tokens=NEW_TOKENS)
    a_adapt = answer(tokenizer, a_ids, n_prompt)

    for label, text in [("full cache", a_full),
                        ("StreamingLLM", a_stream),
                        ("Adaptive (yours)", a_adapt)]:
        found = "FOUND" if NEEDLE in text else "missed"
        print(f"[{label:>16}] {found} needle | answer: {text[:60]!r}")

    print(f"\n(The needle is {NEEDLE}. This is one example; Phase 4 runs the full "
          "sweep across many needle positions to measure it properly.)")


if __name__ == "__main__":
    main()
