"""Model + tokenizer loading, and our own token-by-token decode loop.

Why our own loop instead of model.generate(): the whole project is about
controlling the KV cache during decoding. generate() hides the cache from us,
so in Phase 0 we rewrite the loop ourselves and verify it matches generate()
token for token.

Run in Colab (with a GPU) to verify:
    !python src/akvc/model.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the cache helpers whether this file is imported as a package
# (akvc.model) or run directly as a loose script.
try:
    from akvc.cache_manager import cache_length, evict
except ModuleNotFoundError:
    from cache_manager import cache_length, evict

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model():
    """Load the tokenizer (words <-> numbers) and the model (the brain) on GPU."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()  # inference mode, not training
    return tokenizer, model


def build_inputs(tokenizer, user_message):
    """Wrap a message in chat tags and translate it into numbers on the GPU."""
    messages = [{"role": "user", "content": user_message}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return tokenizer(text, return_tensors="pt").to("cuda")


@torch.no_grad()  # we're only running the model, never training -> saves memory
def manual_decode(model, tokenizer, inputs, max_new_tokens=40):
    """Our own greedy decode loop, managing the KV cache by hand.

    Greedy = always pick the single most-likely next token (no randomness), so
    the result is deterministic and we can compare it against generate().
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # --- Phase 1: PREFILL -------------------------------------------------
    # Run the whole prompt once. This builds the initial KV cache and gives us
    # a prediction for the first new token.
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values                          # the KV cache so far
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy pick
    generated = [next_token]

    # --- Phase 2: DECODE --------------------------------------------------
    # Feed ONE token at a time, reusing the cache instead of re-reading history.
    for _ in range(max_new_tokens - 1):
        # the new token also needs to be "visible" in the attention mask
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=1
        )
        out = model(
            input_ids=next_token,        # just the one newest token
            attention_mask=attention_mask,
            past_key_values=past,        # hand back the cache we built
            use_cache=True,
        )
        past = out.past_key_values                       # cache grew by one token
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)

        if next_token.item() == tokenizer.eos_token_id:  # model said "I'm done"
            break

    generated_ids = torch.cat(generated, dim=1)
    return torch.cat([input_ids, generated_ids], dim=1)  # prompt + reply


@torch.no_grad()
def decode_with_policy(
    model, tokenizer, inputs, policy, budget, max_new_tokens=64, return_trace=False
):
    """Greedy decode, but apply an eviction `policy` to the cache each step.

    Before each new token, we ask the policy which positions to keep (given the
    current cache length and the budget) and physically evict the rest. The
    result: the cache never grows past `budget`, instead of growing forever.
    """
    input_ids = inputs["input_ids"]
    attn = inputs["attention_mask"]

    # Prefill (same as before): build the cache, predict the first new token.
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    trace = [cache_length(past)]

    for _ in range(max_new_tokens - 1):
        # 1) ask the policy what to keep, and evict the rest
        n = cache_length(past)
        keep = policy.keep_indices(n, budget)
        if len(keep) < n:
            evict(past, keep)

        # 2) attention mask must match the (possibly trimmed) cache + the new token
        n = cache_length(past)
        attn = torch.ones((1, n + 1), dtype=torch.long, device=input_ids.device)

        # 3) one decode step
        out = model(
            input_ids=next_token,
            attention_mask=attn,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        trace.append(cache_length(past))

        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_ids = torch.cat(generated, dim=1)
    full = torch.cat([input_ids, generated_ids], dim=1)
    return (full, trace) if return_trace else full


def verify_against_generate(user_message="Explain what a KV cache is in one sentence."):
    """Sanity check: our loop must produce the same tokens as model.generate()."""
    tokenizer, model = load_model()
    inputs = build_inputs(tokenizer, user_message)

    ours = manual_decode(model, tokenizer, inputs, max_new_tokens=40)
    ref = model.generate(**inputs, max_new_tokens=40, do_sample=False)  # greedy too

    # compare only the newly generated tokens, trimmed to the same length
    n_prompt = inputs["input_ids"].shape[1]
    ours_new = ours[0, n_prompt:]
    ref_new = ref[0, n_prompt:]
    length = min(len(ours_new), len(ref_new))
    match = torch.equal(ours_new[:length], ref_new[:length])

    print("Ours    :", tokenizer.decode(ours_new, skip_special_tokens=True))
    print("generate:", tokenizer.decode(ref_new, skip_special_tokens=True))
    print("\nTokens match:", match)
    return match


if __name__ == "__main__":
    verify_against_generate()
