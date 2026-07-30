"""H2O against StreamingLLM under the same budget, on a long story.

Both cap the cache at BUDGET but pick different tokens: StreamingLLM by position,
H2O by how much attention a token has collected. The needle tests show H2O losing
on retrieval; this one shows what the generated text itself looks like.

Needs eager attention so H2O can read attention scores.

    python scripts/h2o_demo.py
"""

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model, build_inputs, decode_with_policy
from akvc.policies.streaming_llm import StreamingLLMPolicy
from akvc.policies.h2o import H2OPolicy

PROMPT = "Tell me a long, detailed story about a lighthouse keeper and the sea."
NEW_TOKENS = 300
BUDGET = 128


def main():
    # float32 because eager attention overflows to NaN in float16
    tokenizer, model = load_model(attn_implementation="eager", dtype=torch.float32)
    inputs = build_inputs(tokenizer, PROMPT)
    n_prompt = inputs["input_ids"].shape[1]

    print(f"\nPrompt: {n_prompt} tokens | generated: {NEW_TOKENS} | budget: {BUDGET}\n")
    for label, policy in [("StreamingLLM", StreamingLLMPolicy(sinks=4)), ("H2O", H2OPolicy())]:
        ids, trace = decode_with_policy(model, tokenizer, inputs, policy, budget=BUDGET,
                                        max_new_tokens=NEW_TOKENS, return_trace=True)
        text = tokenizer.decode(ids[0, n_prompt:], skip_special_tokens=True)
        print(f"[{label}] cache capped at {max(trace)} tokens")
        print("   " + text[:220] + "\n")


if __name__ == "__main__":
    main()
