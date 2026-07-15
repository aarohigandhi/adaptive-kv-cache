"""
Phase 0 — model bring-up and a first look at the KV cache.

Goal of this file: prove the environment works and make the KV cache *visible*.
Run this in a Colab notebook with a GPU (Runtime -> Change runtime type -> T4 GPU).

The mental model:
    words --(tokenizer)--> numbers --(model)--> predicted numbers --(tokenizer)--> words
The "KV cache" is the running memory the model keeps for every token it has seen.
It grows by one entry per token, at every layer -- that growth is the problem this
whole project attacks.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # small, openly available, fits a free T4


def load_model():
    """Load the translator (tokenizer) and the brain (model) onto the GPU."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,   # half-precision numbers so it fits the free GPU
        device_map="cuda",     # put the brain on the GPU
    )
    return tokenizer, model


def build_inputs(tokenizer, user_message):
    """Wrap a message in chat tags, then translate it into numbers on the GPU."""
    messages = [{"role": "user", "content": user_message}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return tokenizer(text, return_tensors="pt").to("cuda")


def generate_reply(tokenizer, model, inputs, max_new_tokens=40):
    """Let the library generate a reply (the baseline we will later reproduce by hand)."""
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def peek_at_cache(model, inputs):
    """Run a single forward pass and report the shape + size of the KV cache."""
    with torch.no_grad():                        # just looking, not training
        out = model(**inputs, use_cache=True)    # ask the model to build + return the cache
    cache = out.past_key_values

    # Grab the stored Keys from layer 0 (handle a few transformers versions safely)
    try:
        k = cache.layers[0].keys
    except AttributeError:
        try:
            k = cache.key_cache[0]
        except AttributeError:
            k = cache[0][0]

    layers = model.config.num_hidden_layers
    _, kv_heads, tokens, head_dim = k.shape

    print("Cache type:            ", type(cache).__name__)
    print("Layers:                ", layers)
    print("Layer-0 Keys shape:    ", tuple(k.shape), "= (batch, kv_heads, tokens, head_dim)")
    print("Tokens in prompt:      ", inputs["input_ids"].shape[1])

    # How big does the full cache get? (Keys + Values, all layers, in float16 = 2 bytes)
    def cache_megabytes(n_tokens):
        numbers = layers * 2 * kv_heads * n_tokens * head_dim
        return numbers * 2 / 1e6

    print(f"Cache for {tokens} tokens:    {cache_megabytes(tokens):.1f} MB")
    print(f"Cache for 32,000 tokens: {cache_megabytes(32000):.0f} MB  <- the memory problem")


if __name__ == "__main__":
    tokenizer, model = load_model()

    inputs = build_inputs(tokenizer, "Say hello in exactly five words.")

    print("=== Library-generated reply ===")
    print(generate_reply(tokenizer, model, inputs))

    print("\n=== A first look at the KV cache ===")
    peek_at_cache(model, inputs)
