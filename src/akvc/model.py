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


def load_model(attn_implementation=None, dtype=torch.float16):
    """Load the tokenizer (words <-> numbers) and the model (the brain) on GPU.

    attn_implementation="eager" is required for policies that read attention
    scores (like H2O); the default (faster) mode doesn't expose them. Note:
    eager attention is numerically unstable in float16 (it can overflow to NaN),
    so pass dtype=torch.float32 whenever you use eager.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    kwargs = {}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map="cuda",
        **kwargs,
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


def _init_importance(attentions):
    """Per-token importance from the prefill attention: for each token, how much
    attention did it receive, summed over every query, head, and layer."""
    importance = None
    for a in attentions:                 # a: [batch, heads, queries, keys]
        # sum over heads (dim 1) then over queries (dim 1 of what's left) -> [keys]
        score = a[0].sum(dim=0).sum(dim=0).float()
        importance = score if importance is None else importance + score
    return importance


def _update_importance(importance, attentions):
    """Add the newest token's attention (over all cached keys) to the tally,
    and grow the tally by one slot for the token we just added."""
    n_keys = attentions[0].shape[-1]
    new = torch.zeros(n_keys, device=importance.device, dtype=importance.dtype)
    for a in attentions:                 # a: [batch, heads, 1, keys]
        new += a[0, :, 0, :].sum(dim=0).float()   # sum over heads -> [keys]
    # importance currently covers the old keys; pad for the new token, then add.
    pad = n_keys - importance.shape[0]
    if pad > 0:
        importance = torch.cat(
            [importance, torch.zeros(pad, device=importance.device, dtype=importance.dtype)]
        )
    return importance + new


@torch.no_grad()
def decode_with_policy(
    model, tokenizer, inputs, policy, budget, max_new_tokens=64, return_trace=False
):
    """Greedy decode, but apply an eviction `policy` to the cache each step.

    Before each new token, we ask the policy which positions to keep (given the
    current cache length and the budget) and physically evict the rest. The
    result: the cache never grows past `budget`, instead of growing forever.

    If the policy needs attention scores (policy.needs_attention), we collect
    them each step and keep a running per-token importance tally to hand it.
    """
    needs_attn = getattr(policy, "needs_attention", False)
    input_ids = inputs["input_ids"]
    attn = inputs["attention_mask"]

    # Prefill: build the cache, predict the first new token.
    out = model(
        input_ids=input_ids, attention_mask=attn,
        use_cache=True, output_attentions=needs_attn,
    )
    past = out.past_key_values
    importance = _init_importance(out.attentions) if needs_attn else None
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    trace = [cache_length(past)]

    for _ in range(max_new_tokens - 1):
        # 1) ask the policy what to keep, and evict the rest (from cache + tally)
        n = cache_length(past)
        stats = {"importance": importance} if needs_attn else None
        keep = policy.keep_indices(n, budget, stats)
        if len(keep) < n:
            evict(past, keep)
            if needs_attn:
                keep_idx = torch.as_tensor(keep, dtype=torch.long, device=importance.device)
                importance = importance.index_select(0, keep_idx)

        # 2) attention mask must match the (possibly trimmed) cache + the new token
        n = cache_length(past)
        attn = torch.ones((1, n + 1), dtype=torch.long, device=input_ids.device)

        # 3) one decode step
        out = model(
            input_ids=next_token,
            attention_mask=attn,
            past_key_values=past,
            use_cache=True,
            output_attentions=needs_attn,
        )
        past = out.past_key_values
        if needs_attn:
            importance = _update_importance(importance, out.attentions)
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
