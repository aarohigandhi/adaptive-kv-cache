"""Phase 2: H2O (heavy-hitter) policy vs StreamingLLM, both under one budget.

Both policies cap the cache at `BUDGET`, but they choose *which* tokens to keep
differently: StreamingLLM by position, H2O by accumulated attention. This runs
both and prints the capped cache size plus a text snippet from each.

Needs the model in "eager" attention mode so H2O can read attention scores.

Run in Colab (with a GPU), from the repo root:
    !python scripts/h2o_demo.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model, build_inputs, decode_with_policy  # noqa: E402
from akvc.policies.streaming_llm import StreamingLLMPolicy           # noqa: E402
from akvc.policies.h2o import H2OPolicy                              # noqa: E402

PROMPT = "Tell me a long, detailed story about a lighthouse keeper and the sea."
NEW_TOKENS = 300
BUDGET = 128


def snippet(tokenizer, ids, n_prompt, chars=220):
    return tokenizer.decode(ids[0, n_prompt:], skip_special_tokens=True)[:chars]


def main():
    # eager attention is required for H2O to see attention scores; use float32
    # because eager attention overflows to NaN in float16.
    tokenizer, model = load_model(attn_implementation="eager", dtype=torch.float32)
    inputs = build_inputs(tokenizer, PROMPT)
    n_prompt = inputs["input_ids"].shape[1]

    runs = {
        "StreamingLLM": StreamingLLMPolicy(sinks=4),
        "H2O": H2OPolicy(),
    }

    print(f"\nPrompt: {n_prompt} tokens | generated: {NEW_TOKENS} | budget: {BUDGET}\n")
    for label, policy in runs.items():
        ids, trace = decode_with_policy(
            model, tokenizer, inputs, policy, budget=BUDGET,
            max_new_tokens=NEW_TOKENS, return_trace=True,
        )
        print(f"[{label}] cache capped at {max(trace)} tokens")
        print("   " + snippet(tokenizer, ids, n_prompt) + "\n")


if __name__ == "__main__":
    main()
