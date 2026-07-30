"""Running any policy on one prompt through a single call.

Full cache never evicts, SnapKV compresses once after the prompt, and the rest
evict at every step, so run_policy dispatches to the right flow and returns the
answer text.
"""

import torch

from akvc.model import build_inputs, manual_decode, decode_with_policy
from akvc.cache_manager import cache_length, evict
from akvc.policies.streaming_llm import StreamingLLMPolicy
from akvc.policies.h2o import H2OPolicy
from akvc.policies.snapkv import SnapKVPolicy
from akvc.policies.adaptive import AdaptivePolicy, AdaptiveQAPolicy

POLICIES = ["full", "streaming_llm", "h2o", "snapkv", "adaptive"]


def _make_policy(name):
    return {
        "streaming_llm": StreamingLLMPolicy(sinks=4),
        "h2o": H2OPolicy(),
        "adaptive": AdaptivePolicy(sinks=4),
        "adaptive_qa": AdaptiveQAPolicy(sinks=4),
    }[name]


@torch.no_grad()
def _run_snapkv(model, tokenizer, inputs, budget, window, max_new_tokens):
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
    abs_pos = n_prompt
    for _ in range(max_new_tokens - 1):
        n = cache_length(past)
        attn = torch.ones((1, n + 1), dtype=torch.long, device=device)
        position_ids = torch.tensor([[abs_pos]], dtype=torch.long, device=device)
        out = model(input_ids=next_token, attention_mask=attn, past_key_values=past,
                    position_ids=position_ids, use_cache=True)
        past = out.past_key_values
        abs_pos += 1
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return torch.cat([inputs["input_ids"]] + generated, dim=1)


def run_policy(model, tokenizer, prompt, policy_name, budget, max_new_tokens=16, snapkv_window=32):
    """Run one policy on one prompt and return the generated answer text."""
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
