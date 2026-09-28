"""Loading the model and running the decode loops.

manual_decode is a plain greedy loop that reproduces model.generate, so we can
trust it. decode_with_policy is the same loop but it asks a policy which cache
positions to keep at each step and evicts the rest, keeping the cache within a
budget. When a policy needs attention scores we collect them and keep a running
importance tally per token.

Everything here runs on whatever device is available. A GPU is used when there is
one; otherwise it falls back to CPU, which is how the tests run.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from akvc.cache_manager import cache_length, evict
    from akvc.instrumentation import timer, kv_cache_bytes
except ModuleNotFoundError:
    from cache_manager import cache_length, evict
    from instrumentation import timer, kv_cache_bytes

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Queries per prefill chunk when attention scores are needed. See chunked_prefill.
PREFILL_CHUNK = 256


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(attn_implementation=None, dtype=None, device=None, model_name=MODEL_NAME):
    """Load the tokenizer and model onto the best available device.

    Pass attn_implementation="eager" for policies that read attention scores.
    Eager attention overflows to NaN in float16, so eager forces float32 unless a
    dtype is given explicitly. CPU is float32 either way.
    """
    device = device or pick_device()
    if dtype is None:
        dtype = torch.float32 if (device == "cpu" or attn_implementation == "eager") else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model


def build_inputs(tokenizer, user_message, device=None):
    """Turn a user message into token ids, wrapped in the chat template."""
    messages = [{"role": "user", "content": user_message}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(text, return_tensors="pt").to(device or pick_device())


def _greedy(logits):
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def _record(metrics, t_prefill, t_decode, n_generated, peak_kv_bytes,
            decode_peak_kv_bytes, final_cache_len):
    """Fill a metrics dict, if the caller passed one.

    prefill_seconds is time to first token. decode_ms_per_token divides by the
    tokens after the first, since the first one comes out of prefill.

    Two memory numbers, because they answer different questions. peak_kv_bytes is
    the high water mark over the whole run, which every policy here pays in full:
    prefill builds the entire cache before anything can be evicted. decode_peak_kv
    _bytes is the high water mark after the first eviction, which is what the
    policy actually buys you, and the number that matters for long generations.
    """
    if metrics is None:
        return
    steps = max(1, n_generated - 1)
    metrics.update({
        "prefill_seconds": t_prefill["seconds"],
        "decode_seconds": t_decode["seconds"],
        "decode_ms_per_token": t_decode["seconds"] / steps * 1000,
        "tokens_generated": n_generated,
        "peak_kv_bytes": peak_kv_bytes,
        "decode_peak_kv_bytes": decode_peak_kv_bytes,
        "final_cache_length": final_cache_len,
    })


@torch.no_grad()
def manual_decode(model, tokenizer, inputs, max_new_tokens=40, metrics=None):
    """Greedy decode by hand, keeping the whole cache. Reproduces model.generate.

    Pass a dict as metrics to have prefill time, decode time and peak cache size
    written into it.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with timer() as t_prefill:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values
    next_token = _greedy(out.logits)
    generated = [next_token]
    peak_kv = kv_cache_bytes(past)

    with timer() as t_decode:
        for _ in range(max_new_tokens - 1):
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            out = model(input_ids=next_token, attention_mask=attention_mask,
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            peak_kv = max(peak_kv, kv_cache_bytes(past))
            next_token = _greedy(out.logits)
            generated.append(next_token)
            if next_token.item() == tokenizer.eos_token_id:
                break

    _record(metrics, t_prefill, t_decode, len(generated), peak_kv, peak_kv,
            cache_length(past))
    return torch.cat([input_ids] + generated, dim=1)


def _init_importance(attentions):
    """Total attention each token received during prefill, summed over queries, heads, layers."""
    importance = None
    for a in attentions:
        score = a[0].sum(dim=0).sum(dim=0).float()
        importance = score if importance is None else importance + score
    return importance


def _add_importance(importance, attentions):
    """Fold one chunk's attention into a running per token tally, growing it as the
    context does."""
    chunk = _init_importance(attentions)
    if importance is None:
        return chunk
    pad = chunk.shape[0] - importance.shape[0]
    if pad > 0:
        importance = torch.cat([importance, torch.zeros(pad, device=importance.device,
                                                        dtype=importance.dtype)])
    return importance + chunk


@torch.no_grad()
def chunked_prefill(model, input_ids, attention_mask=None, chunk_size=None,
                    need_importance=False, observation_window=None):
    """Prefill the prompt a chunk at a time, returning (past, logits, importance,
    window_importance).

    Reading attention scores means asking for output_attentions, and that returns a
    tensor of shape (heads, queries, keys) for every layer at once. For this model at
    a 4K prompt that is about 22 GB, which does not fit on the GPU this project is
    meant to run on. Prefilling in chunks holds one chunk of queries at a time and
    folds each into a running tally, which costs about 1.4 GB at a 256 token chunk
    and gives the identical result: every query still attends to every earlier key,
    and summing over queries does not care what order they arrive in.

    observation_window is SnapKV's: it needs the attention from only the last few
    prompt tokens, so the prompt is split to put exactly those in the final chunk.

    With need_importance False this is a plain prefill and runs in one pass.
    """
    device = input_ids.device
    n = input_ids.shape[1]
    chunk_size = chunk_size or PREFILL_CHUNK

    if not need_importance:
        mask = attention_mask if attention_mask is not None else torch.ones(
            (1, n), dtype=torch.long, device=device)
        out = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
        return out.past_key_values, out.logits, None, None

    bounds = []
    tail = min(observation_window, n) if observation_window else 0
    head_end = n - tail
    start = 0
    while start < head_end:
        bounds.append((start, min(start + chunk_size, head_end)))
        start += chunk_size
    if tail:
        bounds.append((head_end, n))

    past, logits, importance, window_importance = None, None, None, None
    for start, end in bounds:
        piece = input_ids[:, start:end]
        mask = torch.ones((1, end), dtype=torch.long, device=device)
        out = model(input_ids=piece, attention_mask=mask, past_key_values=past,
                    use_cache=True, output_attentions=True)
        past = out.past_key_values
        logits = out.logits
        importance = _add_importance(importance, out.attentions)
        if tail and (start, end) == bounds[-1]:
            window_importance = _init_importance(out.attentions)
        del out

    return past, logits, importance, window_importance


def _update_importance(importance, attentions):
    """Add the newest token's attention over the cached keys, growing the tally by one slot."""
    n_keys = attentions[0].shape[-1]
    new = torch.zeros(n_keys, device=importance.device, dtype=importance.dtype)
    for a in attentions:
        new += a[0, :, 0, :].sum(dim=0).float()
    pad = n_keys - importance.shape[0]
    if pad > 0:
        new_slots = torch.zeros(pad, device=importance.device, dtype=importance.dtype)
        importance = torch.cat([importance, new_slots])
    return importance + new


@torch.no_grad()
def decode_with_policy(model, tokenizer, inputs, policy, budget,
                       max_new_tokens=64, return_trace=False, metrics=None):
    """Greedy decode that evicts the cache down to budget each step using policy.

    Each new token is given its true absolute position. The kept keys keep their
    original positions, so if the position shrank with the cache the RoPE
    distances would break and the output would fall apart.
    """
    needs_attn = getattr(policy, "needs_attention", False)
    input_ids = inputs["input_ids"]
    attn = inputs["attention_mask"]

    with timer() as t_prefill:
        past, logits, importance, _ = chunked_prefill(
            model, input_ids, attention_mask=attn, need_importance=needs_attn)
    next_token = _greedy(logits)
    generated = [next_token]
    trace = [cache_length(past)]
    abs_pos = cache_length(past)
    peak_kv = kv_cache_bytes(past)
    decode_peak_kv = 0

    with timer() as t_decode:
        for _ in range(max_new_tokens - 1):
            n = cache_length(past)
            stats = {"importance": importance} if needs_attn else None
            keep = policy.keep_indices(n, budget, stats)
            if len(keep) < n:
                evict(past, keep)
                if needs_attn:
                    keep_idx = torch.as_tensor(keep, dtype=torch.long, device=importance.device)
                    importance = importance.index_select(0, keep_idx)

            n = cache_length(past)
            attn = torch.ones((1, n + 1), dtype=torch.long, device=input_ids.device)
            position_ids = torch.tensor([[abs_pos]], dtype=torch.long, device=input_ids.device)

            out = model(input_ids=next_token, attention_mask=attn, past_key_values=past,
                        position_ids=position_ids, use_cache=True, output_attentions=needs_attn)
            past = out.past_key_values
            held = kv_cache_bytes(past)
            peak_kv = max(peak_kv, held)
            decode_peak_kv = max(decode_peak_kv, held)
            abs_pos += 1
            if needs_attn:
                importance = _update_importance(importance, out.attentions)
            next_token = _greedy(out.logits)
            generated.append(next_token)
            trace.append(cache_length(past))
            if next_token.item() == tokenizer.eos_token_id:
                break

    _record(metrics, t_prefill, t_decode, len(generated), peak_kv,
            decode_peak_kv or peak_kv, cache_length(past))
    full = torch.cat([input_ids] + generated, dim=1)
    return (full, trace) if return_trace else full


def verify_against_generate(user_message="Explain what a KV cache is in one sentence."):
    """Check that manual_decode produces the same tokens as model.generate."""
    tokenizer, model = load_model()
    inputs = build_inputs(tokenizer, user_message, device=model.device)

    ours = manual_decode(model, tokenizer, inputs, max_new_tokens=40)
    ref = model.generate(**inputs, max_new_tokens=40, do_sample=False)

    n_prompt = inputs["input_ids"].shape[1]
    ours_new, ref_new = ours[0, n_prompt:], ref[0, n_prompt:]
    length = min(len(ours_new), len(ref_new))
    match = torch.equal(ours_new[:length], ref_new[:length])

    print("Ours    :", tokenizer.decode(ours_new, skip_special_tokens=True))
    print("generate:", tokenizer.decode(ref_new, skip_special_tokens=True))
    print("\nTokens match:", match)
    return match


if __name__ == "__main__":
    verify_against_generate()
