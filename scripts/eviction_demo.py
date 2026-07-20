"""Phase 2: watch a policy actually bound the cache during generation.

Generates the same continuation two ways -- full cache vs StreamingLLM with a
fixed budget -- and reports how big the cache got each way, the peak memory, and
a snippet of the text so you can eyeball that it's still coherent.

Run in Colab (with a GPU), from the repo root:
    !python scripts/eviction_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import (                                            # noqa: E402
    load_model,
    build_inputs,
    manual_decode,
    decode_with_policy,
)
from akvc.instrumentation import reset_peak_memory, peak_memory_mb  # noqa: E402
from akvc.policies.streaming_llm import StreamingLLMPolicy          # noqa: E402

PROMPT = "Tell me a long, detailed story about a lighthouse keeper and the sea."
NEW_TOKENS = 300
BUDGET = 128


def snippet(tokenizer, ids, n_prompt, chars=220):
    return tokenizer.decode(ids[0, n_prompt:], skip_special_tokens=True)[:chars]


def main():
    tokenizer, model = load_model()
    inputs = build_inputs(tokenizer, PROMPT)
    n_prompt = inputs["input_ids"].shape[1]

    # --- Full cache (keep everything) ---
    reset_peak_memory()
    full_ids = manual_decode(model, tokenizer, inputs, max_new_tokens=NEW_TOKENS)
    full_mem = peak_memory_mb()
    full_len = full_ids.shape[1]  # prompt + generated = final cache size

    # --- StreamingLLM (bounded budget) ---
    reset_peak_memory()
    policy = StreamingLLMPolicy(sinks=4)
    sllm_ids, trace = decode_with_policy(
        model, tokenizer, inputs, policy, budget=BUDGET,
        max_new_tokens=NEW_TOKENS, return_trace=True,
    )
    sllm_mem = peak_memory_mb()

    print(f"\nPrompt length: {n_prompt} tokens | generated: {NEW_TOKENS} tokens\n")
    print("                 final cache size   peak GPU memory")
    print(f"  full cache:        {full_len:>5} tokens        {full_mem:6.1f} MB")
    print(f"  StreamingLLM:      {max(trace):>5} tokens        {sllm_mem:6.1f} MB   (budget {BUDGET})")
    print(f"\n  -> full cache grew to {full_len}; StreamingLLM stayed capped at {max(trace)}.")

    print("\n--- Full-cache text ---")
    print(snippet(tokenizer, full_ids, n_prompt))
    print("\n--- StreamingLLM text (should still read coherently) ---")
    print(snippet(tokenizer, sllm_ids, n_prompt))


if __name__ == "__main__":
    main()
