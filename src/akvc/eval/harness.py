"""Eval harness — run any policy on a prompt through one interface.

Different policies need slightly different flows (full cache = no eviction,
SnapKV = one-shot prefill compression, the rest = per-step eviction), so
run_policy() hides that behind a single call and returns the decoded answer.
"""

import torch

from akvc.model import (
    build_inputs, manual_decode, decode_with_policy,
)
from akvc.cache_manager import cache_length, evict
from akvc.policies.streaming_llm import StreamingLLMPolicy
from akvc.policies.h2o import H2OPolicy
from akvc.policies.snapkv import SnapKVPolicy
from akvc.policies.adaptive import AdaptivePolicy

# Names in the fixed order we want them to appear in charts/tables.
POLICIES = ["full", "streaming_llm", "h2o", "snapkv", "adaptive"]


def _make_policy(name):
    return {
        "streaming_llm": StreamingLLMPolicy(sinks=4),
        "h2o": H2OPolicy(),
        "adaptive": AdaptivePolicy(sinks=4),
    }[name]


@torch.no_grad()
def _run_snapkv(model, tokenizer, inputs, budget, window, max_new_tokens):
    """SnapKV's one-shot flow: score prompt tokens from the observation window,
    compress the cache once, then decode from the compressed cache."""
    device = inputs["input_ids"].device
    n_prompt = inputs["input_ids"].shape[1]

    out = model(**inputs, use_cache=True, output_attentions=True)
    past = out.past_key_values

    w = min(window, n_prompt)
    importance = None
    for a in out.attentions:
        score = a[0, :, -w:, :].float().sum(dim=0).sum(dim=0)
        importance = score if importance is None else importance + score

    keep = SnapKVPolicy(window=window).keep_indices(n_prompt, budget, {"importance": importance})
    if len(keep) < n_prompt:
        evict(past, keep)

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    for _ in range(max_new_tokens - 1):
        n = cache_length(past)
        attn = torch.ones((1, n + 1), dtype=torch.long, device=device)
        out = model(input_ids=next_token, attention_mask=attn,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if next_token.item() == tokenizer.eos_token_id:
            break
    full = torch.cat([inputs["input_ids"], torch.cat(generated, dim=1)], dim=1)
    return full


def run_policy(model, tokenizer, prompt, policy_name, budget,
               max_new_tokens=16, snapkv_window=32):
    """Run one policy on one prompt; return just the generated answer text."""
    inputs = build_inputs(tokenizer, prompt)
    n_prompt = inputs["input_ids"].shape[1]

    if policy_name == "full":
        ids = manual_decode(model, tokenizer, inputs, max_new_tokens=max_new_tokens)
    elif policy_name == "snapkv":
        ids = _run_snapkv(model, tokenizer, inputs, budget, snapkv_window, max_new_tokens)
    else:
        ids = decode_with_policy(model, tokenizer, inputs, _make_policy(policy_name),
                                 budget=budget, max_new_tokens=max_new_tokens)

    return tokenizer.decode(ids[0, n_prompt:], skip_special_tokens=True).strip()
