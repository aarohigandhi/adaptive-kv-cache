"""SnapKV compressing the prompt once. It scores the prompt tokens from the
observation window, keeps the top ones plus the window, and answers from the
compressed cache.

    python scripts/snapkv_demo.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.model import load_model, build_inputs
from akvc.cache_manager import cache_length, evict
from akvc.policies.snapkv import SnapKVPolicy

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
    importance = None
    for a in attentions:
        score = a[0, :, -window:, :].float().sum(dim=0).sum(dim=0)
        importance = score if importance is None else importance + score
    return importance


@torch.no_grad()
def main():
    tokenizer, model = load_model(attn_implementation="eager", dtype=torch.float32)
    inputs = build_inputs(tokenizer, PROMPT)
    device = inputs["input_ids"].device
    n_prompt = inputs["input_ids"].shape[1]

    out = model(**inputs, use_cache=True, output_attentions=True)
    past = out.past_key_values

    window = min(WINDOW, n_prompt)
    importance = snapkv_importance(out.attentions, window)
    keep = SnapKVPolicy(window=WINDOW).keep_indices(n_prompt, BUDGET, {"importance": importance})
    print(f"Prompt: {n_prompt} tokens, SnapKV kept {len(keep)} (budget {BUDGET}), dropped {n_prompt - len(keep)}.")
    if len(keep) < n_prompt:
        evict(past, keep)

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    abs_pos = n_prompt
    for _ in range(NEW_TOKENS - 1):
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

    text = tokenizer.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True)
    print("\nAnswer from the compressed cache:")
    print(text[:300])


if __name__ == "__main__":
    main()
