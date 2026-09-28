"""Running any policy on one prompt through a single call.

Full cache never evicts, SnapKV compresses once after the prompt, and the rest
evict at every step, so run_policy dispatches to the right flow and returns the
answer text together with its latency and cache size.

Budgets are given as a fraction of the prompt, not an absolute token count. A
budget of 0.25 means keep a quarter of the cache, which is the axis the published
baselines report on, so a number here can be compared to a number there. The
absolute token count each policy actually received is recorded alongside.
"""

import torch

from akvc.model import (build_inputs, chunked_prefill, manual_decode,
                        decode_with_policy)
from akvc.cache_manager import cache_length, evict
from akvc.instrumentation import timer, kv_cache_bytes
from akvc.policies.streaming_llm import StreamingLLMPolicy
from akvc.policies.h2o import H2OPolicy
from akvc.policies.full import FullCachePolicy
from akvc.policies.snapkv import SnapKVPolicy
from akvc.policies.adaptive import AdaptivePolicy, AdaptiveQAPolicy

POLICIES = ["full", "streaming_llm", "h2o", "snapkv", "adaptive"]

# The ratios the spec preregistered. 0.25 is the one the success criterion is
# stated at; the other two bracket it.
BUDGET_RATIOS = [0.5, 0.25, 0.125]

# Below this, the anchors and the recent window alone eat the whole budget and the
# comparison stops meaning anything.
MIN_BUDGET = 16


SNAPKV_WINDOW = 32


def make_policy(name):
    """Build a policy by name.

    full and snapkv still take their own paths in run_policy, since one never
    evicts and the other compresses once instead of every step, but both are
    constructible here so the policy logic can be tested on its own.
    """
    return {
        "full": FullCachePolicy(),
        "streaming_llm": StreamingLLMPolicy(sinks=4),
        "h2o": H2OPolicy(),
        "snapkv": SnapKVPolicy(window=SNAPKV_WINDOW),
        "adaptive": AdaptivePolicy(sinks=4),
        "adaptive_qa": AdaptiveQAPolicy(sinks=4),
    }[name]


def budget_for(n_prompt, ratio):
    """Absolute token budget for a prompt at a given keep ratio."""
    return max(MIN_BUDGET, int(round(ratio * n_prompt)))


@torch.no_grad()
def _run_snapkv(model, tokenizer, inputs, budget, window, max_new_tokens, metrics=None):
    device = inputs["input_ids"].device
    n_prompt = inputs["input_ids"].shape[1]

    with timer() as t_prefill:
        past, logits, _, importance = chunked_prefill(
            model, inputs["input_ids"], attention_mask=inputs["attention_mask"],
            need_importance=True, observation_window=window)

    # Measured before the compression, so this is comparable to the other policies:
    # prefill materializes the whole cache either way.
    peak_kv = kv_cache_bytes(past)
    keep = SnapKVPolicy(window=window).keep_indices(n_prompt, budget, {"importance": importance})
    if len(keep) < n_prompt:
        evict(past, keep)
    decode_peak_kv = kv_cache_bytes(past)

    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    abs_pos = n_prompt
    with timer() as t_decode:
        for _ in range(max_new_tokens - 1):
            n = cache_length(past)
            attn = torch.ones((1, n + 1), dtype=torch.long, device=device)
            position_ids = torch.tensor([[abs_pos]], dtype=torch.long, device=device)
            out = model(input_ids=next_token, attention_mask=attn, past_key_values=past,
                        position_ids=position_ids, use_cache=True)
            past = out.past_key_values
            held = kv_cache_bytes(past)
            peak_kv = max(peak_kv, held)
            decode_peak_kv = max(decode_peak_kv, held)
            abs_pos += 1
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if next_token.item() == tokenizer.eos_token_id:
                break

    if metrics is not None:
        steps = max(1, len(generated) - 1)
        metrics.update({
            "prefill_seconds": t_prefill["seconds"],
            "decode_seconds": t_decode["seconds"],
            "decode_ms_per_token": t_decode["seconds"] / steps * 1000,
            "tokens_generated": len(generated),
            "peak_kv_bytes": peak_kv,
            "decode_peak_kv_bytes": decode_peak_kv,
            "final_cache_length": cache_length(past),
        })
    return torch.cat([inputs["input_ids"]] + generated, dim=1)


def run_policy(model, tokenizer, prompt, policy_name, budget=None, ratio=None,
               max_new_tokens=16, snapkv_window=SNAPKV_WINDOW, metrics=None):
    """Run one policy on one prompt and return the generated answer text.

    Give either budget (absolute tokens) or ratio (a fraction of the prompt). Pass
    a dict as metrics to collect latency and cache size for the run.
    """
    if (budget is None) == (ratio is None):
        raise ValueError("Pass exactly one of budget or ratio.")

    inputs = build_inputs(tokenizer, prompt, device=model.device)
    n_prompt = inputs["input_ids"].shape[1]
    if budget is None:
        budget = budget_for(n_prompt, ratio)

    if metrics is not None:
        metrics.update({"prompt_tokens": n_prompt, "budget": budget})

    if policy_name == "full":
        ids = manual_decode(model, tokenizer, inputs, max_new_tokens=max_new_tokens,
                            metrics=metrics)
    elif policy_name == "snapkv":
        ids = _run_snapkv(model, tokenizer, inputs, budget, snapkv_window, max_new_tokens,
                          metrics=metrics)
    else:
        ids = decode_with_policy(model, tokenizer, inputs, make_policy(policy_name),
                                 budget=budget, max_new_tokens=max_new_tokens,
                                 metrics=metrics)

    return tokenizer.decode(ids[0, n_prompt:], skip_special_tokens=True).strip()


# Old private name, kept so anything importing it keeps working.
_make_policy = make_policy
