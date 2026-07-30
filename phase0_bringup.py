"""Bring the model up and look at the KV cache.

Loads the model, generates a reply, and prints the shape and size of the cache
so the growth problem is concrete.

    python phase0_bringup.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="cuda")
    return tokenizer, model


def build_inputs(tokenizer, user_message):
    messages = [{"role": "user", "content": user_message}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(text, return_tensors="pt").to("cuda")


def generate_reply(tokenizer, model, inputs, max_new_tokens=40):
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def peek_at_cache(model, inputs):
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    cache = out.past_key_values
    try:
        k = cache.layers[0].keys
    except AttributeError:
        try:
            k = cache.key_cache[0]
        except AttributeError:
            k = cache[0][0]

    layers = model.config.num_hidden_layers
    _, kv_heads, tokens, head_dim = k.shape
    print("Cache type:         ", type(cache).__name__)
    print("Layers:             ", layers)
    print("Layer 0 keys shape: ", tuple(k.shape), "= (batch, kv_heads, tokens, head_dim)")
    print("Tokens in prompt:   ", inputs["input_ids"].shape[1])

    def cache_megabytes(n_tokens):
        return layers * 2 * kv_heads * n_tokens * head_dim * 2 / 1e6

    print(f"Cache for {tokens} tokens:    {cache_megabytes(tokens):.1f} MB")
    print(f"Cache for 32,000 tokens: {cache_megabytes(32000):.0f} MB")


if __name__ == "__main__":
    tokenizer, model = load_model()
    inputs = build_inputs(tokenizer, "Say hello in exactly five words.")
    print("Library reply:")
    print(generate_reply(tokenizer, model, inputs))
    print("\nKV cache:")
    peek_at_cache(model, inputs)
