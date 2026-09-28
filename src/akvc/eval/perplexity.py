"""Perplexity on PG19 under a cache policy.

The needle test asks whether one fact survived eviction. Perplexity asks the
duller and broader question: with part of the cache thrown away, how much worse
does the model predict ordinary text.

The measurement is teacher forced. Prefill the first chunk of a book, then walk
through the next tokens one at a time, feeding the true token at every step and
adding up the log probability the model gave it. The policy evicts on every step,
exactly as it would during real generation, so the number reflects the policy and
not just the prompt. Lower is better.
"""

import math

import torch

from akvc.cache_manager import cache_length, evict
from akvc.instrumentation import timer, kv_cache_bytes
from akvc.model import chunked_prefill

DATASET = "deepmind/pg19"          # the Hub repo, for the test split file list
BUCKET = "https://storage.googleapis.com/deepmind-gutenberg"  # where the books live


def load_passages(tokenizer, n_passages, n_tokens, seed=0):
    """Token ids for a few PG19 books, each trimmed to n_tokens.

    The books come from the bucket DeepMind published them in, one file each, so
    only the handful we need is downloaded rather than the whole 11 GB corpus. The
    test split's file list lives on the Hub. Books shorter than n_tokens are skipped
    rather than padded, so every passage is the same length and the perplexities are
    averaged over the same amount of text.
    """
    import random
    import urllib.request

    from huggingface_hub import hf_hub_download

    listing = hf_hub_download(DATASET, "data/test_files.txt", repo_type="dataset")
    with open(listing, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    random.Random(seed).shuffle(names)

    passages = []
    for name in names:
        with urllib.request.urlopen(f"{BUCKET}/{name}", timeout=120) as response:
            text = response.read().decode("utf-8", errors="replace")
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < n_tokens:
            continue
        passages.append(ids[:n_tokens])
        if len(passages) >= n_passages:
            break
    if not passages:
        raise RuntimeError("No PG19 book was long enough. Lower n_tokens.")
    return passages


@torch.no_grad()
def passage_nll(model, ids, policy, budget, n_prefill, metrics=None,
                compress_once=False, observation_window=32):
    """Sum of negative log probabilities over the tokens after the prefill.

    Returns (total_nll, n_tokens_scored). policy may be None, which means keep the
    whole cache and gives the uncompressed reference number.

    compress_once is for SnapKV, which compresses a single time right after the
    prompt and then leaves the cache alone. Scoring it per step would turn it into
    a different method and make the comparison dishonest, so it gets its own path,
    scored off the observation window the way the paper does it.
    """
    device = model.device
    needs_attn = bool(policy is not None and getattr(policy, "needs_attention", False))
    input_ids = torch.tensor([ids[:n_prefill]], dtype=torch.long, device=device)

    with timer() as t_prefill:
        past, all_logits, prefill_importance, window_importance = chunked_prefill(
            model, input_ids, need_importance=needs_attn,
            observation_window=observation_window if compress_once else None)
    importance = window_importance if compress_once else prefill_importance
    logits = all_logits[:, -1, :]

    # Read before any compression, so every policy is charged for the prefill it
    # really built.
    peak_kv = kv_cache_bytes(past)
    if compress_once and policy is not None:
        n = cache_length(past)
        keep = policy.keep_indices(n, budget, {"importance": importance})
        if len(keep) < n:
            evict(past, keep)
        needs_attn = False  # nothing reads attention after the one compression
    decode_peak_kv = 0

    total_nll = 0.0
    scored = 0
    with timer() as t_decode:
        for target in ids[n_prefill:]:
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            total_nll += -logprobs[0, target].item()
            scored += 1

            if policy is not None and not compress_once:
                n = cache_length(past)
                stats = {"importance": importance} if needs_attn else None
                keep = policy.keep_indices(n, budget, stats)
                if len(keep) < n:
                    evict(past, keep)
                    if needs_attn:
                        idx = torch.as_tensor(keep, dtype=torch.long, device=importance.device)
                        importance = importance.index_select(0, idx)

            n = cache_length(past)
            attn = torch.ones((1, n + 1), dtype=torch.long, device=device)
            position_ids = torch.tensor([[n_prefill + scored - 1]], dtype=torch.long, device=device)
            token = torch.tensor([[target]], dtype=torch.long, device=device)

            out = model(input_ids=token, attention_mask=attn, past_key_values=past,
                        position_ids=position_ids, use_cache=True,
                        output_attentions=needs_attn)
            past = out.past_key_values
            held = kv_cache_bytes(past)
            peak_kv = max(peak_kv, held)
            decode_peak_kv = max(decode_peak_kv, held)
            if needs_attn:
                importance = _step_importance(importance, out.attentions)
            logits = out.logits[:, -1, :]

    if metrics is not None:
        metrics.update({
            "prefill_seconds": t_prefill["seconds"],
            "decode_seconds": t_decode["seconds"],
            "decode_ms_per_token": t_decode["seconds"] / max(1, scored) * 1000,
            "peak_kv_bytes": peak_kv,
            "decode_peak_kv_bytes": decode_peak_kv or peak_kv,
            "final_cache_length": cache_length(past),
        })
    return total_nll, scored


def perplexity(total_nll, n_tokens):
    return math.exp(total_nll / n_tokens) if n_tokens else float("nan")


def _step_importance(importance, attentions):
    n_keys = attentions[0].shape[-1]
    new = torch.zeros(n_keys, device=importance.device, dtype=importance.dtype)
    for a in attentions:
        new += a[0, :, 0, :].sum(dim=0).float()
    pad = n_keys - importance.shape[0]
    if pad > 0:
        importance = torch.cat([importance, torch.zeros(pad, device=importance.device,
                                                        dtype=importance.dtype)])
    return importance + new
