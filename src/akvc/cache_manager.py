"""Trimming the KV cache during generation.

cache_length reports how many tokens the cache holds. evict keeps only the given
positions in every layer. The cache is a DynamicCache whose internals shifted
across transformers versions, so both helpers handle the two known layouts.
"""

import torch


def cache_length(past):
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    if hasattr(past, "layers"):
        return past.layers[0].keys.shape[2]
    return past.key_cache[0].shape[2]


def _first_keys(past):
    if hasattr(past, "layers"):
        return past.layers[0].keys
    return past.key_cache[0]


def evict(past, keep_indices):
    """Keep only keep_indices along the token axis of every layer's keys and values."""
    idx = torch.as_tensor(keep_indices, dtype=torch.long, device=_first_keys(past).device)

    if hasattr(past, "layers"):
        for layer in past.layers:
            layer.keys = layer.keys.index_select(2, idx)
            layer.values = layer.values.index_select(2, idx)
    elif hasattr(past, "key_cache"):
        for i in range(len(past.key_cache)):
            past.key_cache[i] = past.key_cache[i].index_select(2, idx)
            past.value_cache[i] = past.value_cache[i].index_select(2, idx)
    else:
        raise TypeError(f"Unsupported cache type: {type(past).__name__}")

    if hasattr(past, "_seen_tokens"):
        past._seen_tokens = int(idx.numel())
    return past
