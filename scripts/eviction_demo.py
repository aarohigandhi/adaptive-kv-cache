"""Show a policy bounding the cache during generation. Generates the same text
with the full cache and with StreamingLLM at a fixed budget, reports how big the
cache got each way, and plots the two cache sizes over time.

    python scripts/eviction_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model, build_inputs, manual_decode, decode_with_policy
from akvc.instrumentation import reset_peak_memory, peak_memory_mb
from akvc.policies.streaming_llm import StreamingLLMPolicy

PROMPT = "Tell me a long, detailed story about a lighthouse keeper and the sea."
NEW_TOKENS = 300
BUDGET = 128


def snippet(tokenizer, ids, n_prompt, chars=220):
    return tokenizer.decode(ids[0, n_prompt:], skip_special_tokens=True)[:chars]


def plot_cache_sizes(n_prompt, streaming_trace, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = list(range(len(streaming_trace)))
    full_trace = [n_prompt + s for s in steps]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(steps, full_trace, linewidth=2, color="#2a6fdb", label="full cache")
    ax.plot(steps, streaming_trace, linewidth=2, color="#e8710a", label="StreamingLLM")
    ax.set_title("The cache stops growing", fontsize=13)
    ax.set_xlabel("decode step (tokens generated)")
    ax.set_ylabel("cache size (tokens)")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print("Saved", out_path)


def main():
    tokenizer, model = load_model()
    inputs = build_inputs(tokenizer, PROMPT)
    n_prompt = inputs["input_ids"].shape[1]

    reset_peak_memory()
    full_ids = manual_decode(model, tokenizer, inputs, max_new_tokens=NEW_TOKENS)
    full_mem = peak_memory_mb()
    full_len = full_ids.shape[1]

    reset_peak_memory()
    sllm_ids, trace = decode_with_policy(
        model, tokenizer, inputs, StreamingLLMPolicy(sinks=4),
        budget=BUDGET, max_new_tokens=NEW_TOKENS, return_trace=True,
    )
    sllm_mem = peak_memory_mb()

    print(f"\nPrompt length: {n_prompt} tokens | generated: {NEW_TOKENS} tokens\n")
    print("                 final cache size   peak GPU memory")
    print(f"  full cache:        {full_len:>5} tokens        {full_mem:6.1f} MB")
    print(f"  StreamingLLM:      {max(trace):>5} tokens        {sllm_mem:6.1f} MB   (budget {BUDGET})")
    print(f"\n  full cache grew to {full_len}; StreamingLLM stayed capped at {max(trace)}.")

    print("\nFull cache text:")
    print(snippet(tokenizer, full_ids, n_prompt))
    print("\nStreamingLLM text:")
    print(snippet(tokenizer, sllm_ids, n_prompt))

    os.makedirs("results", exist_ok=True)
    plot_cache_sizes(n_prompt, trace, "results/eviction_cache_size.png")


if __name__ == "__main__":
    main()
