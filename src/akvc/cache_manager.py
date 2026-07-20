"""The cache manager — surgically trims the KV cache mid-generation.

Two helpers the decode loop relies on:
    cache_length(past)        -> how many tokens are in the cache right now
    evict(past, keep_indices) -> drop everything except keep_indices, in place

The model's cache is a "DynamicCache" object. Its exact internals changed across
transformers versions, so both helpers handle the couple of known layouts.
"""

import torch


def cache_length(past):
    """How many tokens are currently stored in the cache."""
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    if hasattr(past, "layers"):
        return past.layers[0].keys.shape[2]
    return past.key_cache[0].shape[2]


def _first_keys(past):
    """Grab any layer's Key tensor (just to read its device)."""
    if hasattr(past, "layers"):
        return past.layers[0].keys
    return past.key_cache[0]


def evict(past, keep_indices):
    """Keep only `keep_indices` (a list of token positions) in every layer.

    We slice each layer's Keys and Values along the token dimension (dim=2:
    shape is [batch, heads, tokens, head_dim]). index_select picks out exactly
    the positions we want to keep and drops the rest.
    """
    idx = torch.as_tensor(
        keep_indices, dtype=torch.long, device=_first_keys(past).device
    )

    if hasattr(past, "layers"):                    # newer transformers layout
        for layer in past.layers:
            layer.keys = layer.keys.index_select(2, idx)
            layer.values = layer.values.index_select(2, idx)
    elif hasattr(past, "key_cache"):               # older transformers layout
        for i in range(len(past.key_cache)):
            past.key_cache[i] = past.key_cache[i].index_select(2, idx)
            past.value_cache[i] = past.value_cache[i].index_select(2, idx)
    else:
        raise TypeError(f"Unsupported cache type: {type(past).__name__}")

    # keep the cache's own length bookkeeping in sync with what we kept
    if hasattr(past, "_seen_tokens"):
        past._seen_tokens = int(idx.numel())

    return past
