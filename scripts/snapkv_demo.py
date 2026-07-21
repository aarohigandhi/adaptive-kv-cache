"""Phase 2: SnapKV one-shot prefill compression.

Reads a long prompt, uses the observation window's attention to pick which
prompt tokens to keep, compresses the cache ONCE down to the budget, then
generates an answer from the compressed cache. Prints how much was dropped and
the answer, so you can see the model still responds sensibly.

Needs eager attention + float32. Run in Colab (GPU), from the repo root:
    !python scripts/snapkv_demo.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model, build_inputs        # noqa: E402
from akvc.cache_manager import cache_length, evict      # noqa: E402
from akvc.policies.snapkv import SnapKVPolicy            # noqa: E402

PROMPT = (
    "Here is a short report. The lighthouse on Bell Rock was built in 1810. "
    "It stands 35 metres tall and its light can be seen for 30 kilometres. "
    "The keeper logged the weather, the tides, and passing ships every day. "
    "In the winter of 1861 a great storm damaged the lamp room. "
    "Question: in what year was the Bell Rock lighthouse built, and how tall is it?"
)
BUDGET = 128
WINDOW = 32
NEW_TOKENS = 100


@torch.no_grad()
def snapkv_importance(attentions, window):
    """Per-token score from the observation window: how much the last `window`
    query tokens attended to each earlier token, summed over heads and layers."""
    importance = None
    for a in attentions:                        # a: [1, heads, queries, keys]
        w = a[0, :, -window:, :].float()        # last `window` queries
        score = w.sum(dim=0).sum(dim=0)         # sum over heads and window -> [keys]
        importance = score if importance is None else importance + score
    return importance


@torch.no_grad()
def main():
    tokenizer, model = load_model(attn_implementation="eager", dtype=torch.float32)
    inputs = build_inputs(tokenizer, PROMPT)
    device = inputs["input_ids"].device
    n_prompt = inputs["input_ids"].shape[1]

    # Prefill with attention so SnapKV can score the prompt tokens.
    out = model(**inputs, use_cache=True, output_attentions=True)
    past = out.past_key_values

    window = min(WINDOW, n_prompt)
    importance = snapkv_importance(out.attentions, window)

    policy = SnapKVPolicy(window=WINDOW)
    keep = policy.keep_indices(n_prompt, BUDGET, {"importance": importance})
    print(f"Prompt: {n_prompt} tokens -> SnapKV kept {len(keep)} (budget {BUDGET}), "
          f"dropped {n_prompt - len(keep)}.")
    if len(keep) < n_prompt:
        evict(past, keep)

    # Generate the answer from the compressed cache (plain greedy, no more eviction).
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    for _ in range(NEW_TOKENS - 1):
        n = cache_length(past)
        attn = torch.ones((1, n + 1), dtype=torch.long, device=device)
        out = model(input_ids=next_token, attention_mask=attn,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if next_token.item() == tokenizer.eos_token_id:
            break

    answer = tokenizer.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True)
    print("\nAnswer from the SnapKV-compressed cache:")
    print(answer[:300])
    print("\n(Correct answer is in there if it says 1810 and 35 metres.)")


if __name__ == "__main__":
    main()
