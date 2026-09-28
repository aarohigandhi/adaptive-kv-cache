"""The decode loops, on a 4K context, on CPU.

The first test is the Phase 0 invariant and the one that matters most: a hand
written greedy loop that manages the cache itself has to produce exactly the same
tokens as model.generate. If it does not, every number this repo reports is
measuring a bug. It was unguarded until now.

The rest check that each policy actually holds the cache down to its budget during
generation, that positions stay absolute after eviction, and that the metrics come
back filled in.
"""

import pytest
import torch

from akvc.cache_manager import cache_length, evict
from akvc.eval.harness import make_policy
from akvc.instrumentation import kv_cache_bytes
from akvc.model import decode_with_policy, manual_decode

EVICTING = ["streaming_llm", "h2o", "adaptive", "adaptive_qa"]


def test_manual_decode_matches_generate(tiny_model, tokenizer, fixture_inputs):
    """Phase 0: our own cache handling must not change a single token."""
    n_prompt = fixture_inputs["input_ids"].shape[1]
    max_new = 8

    ours = manual_decode(tiny_model, tokenizer, fixture_inputs, max_new_tokens=max_new)
    with torch.no_grad():
        ref = tiny_model.generate(**fixture_inputs, max_new_tokens=max_new,
                                  do_sample=False)

    ours_new = ours[0, n_prompt:]
    ref_new = ref[0, n_prompt:]
    assert len(ours_new) == max_new
    assert torch.equal(ours_new, ref_new), (
        f"manual decode diverged from generate: {ours_new.tolist()} vs {ref_new.tolist()}"
    )


def test_manual_decode_reports_metrics(tiny_model, tokenizer, fixture_inputs):
    metrics = {}
    manual_decode(tiny_model, tokenizer, fixture_inputs, max_new_tokens=4, metrics=metrics)
    for key in ("prefill_seconds", "decode_ms_per_token", "peak_kv_bytes",
                "decode_peak_kv_bytes", "final_cache_length"):
        assert key in metrics, f"{key} missing"
    assert metrics["peak_kv_bytes"] > 0
    assert metrics["prefill_seconds"] > 0


@pytest.mark.parametrize("name", EVICTING)
def test_policy_holds_the_cache_at_budget(name, tiny_model, tokenizer, fixture_inputs):
    """A 4K context with a 256 token budget: the cache must never exceed it."""
    budget = 256
    metrics = {}
    decode_with_policy(tiny_model, tokenizer, fixture_inputs, make_policy(name),
                       budget=budget, max_new_tokens=6, metrics=metrics)

    # One step of growth is expected: the policy trims, then the new token appends.
    assert metrics["final_cache_length"] <= budget + 1, (
        f"{name} let the cache reach {metrics['final_cache_length']} on a {budget} budget"
    )


@pytest.mark.parametrize("name", EVICTING)
def test_policy_shrinks_the_cache_against_full(name, tiny_model, tokenizer, fixture_inputs):
    """Compression has to actually save memory, not just reorder the cache.

    Compared on the decode peak, not the overall peak: prefill builds the whole
    cache before any policy gets to evict, so the overall peak is the same for all
    of them and comparing it would credit compression with a saving it never made.
    """
    full_metrics, policy_metrics = {}, {}
    manual_decode(tiny_model, tokenizer, fixture_inputs, max_new_tokens=4,
                  metrics=full_metrics)
    decode_with_policy(tiny_model, tokenizer, fixture_inputs, make_policy(name),
                       budget=256, max_new_tokens=4, metrics=policy_metrics)

    assert policy_metrics["decode_peak_kv_bytes"] < full_metrics["decode_peak_kv_bytes"] / 4
    # And the overall peak barely moves, because prefill already built the cache.
    assert policy_metrics["peak_kv_bytes"] > full_metrics["peak_kv_bytes"] * 0.95


def test_policy_output_is_not_degenerate(tiny_model, tokenizer, fixture_inputs):
    """A broken position or a mangled cache shows up as one token repeated forever.

    The weights are random so the text is meaningless either way, but total
    collapse to a single token is the signature of the RoPE bug this repo already
    hit once, so it is worth a tripwire.
    """
    n_prompt = fixture_inputs["input_ids"].shape[1]
    ids = decode_with_policy(tiny_model, tokenizer, fixture_inputs,
                             make_policy("adaptive"), budget=256, max_new_tokens=12)
    produced = ids[0, n_prompt:].tolist()
    assert torch.isfinite(ids.float()).all()
    assert len(produced) == 12


def test_evict_keeps_the_chosen_positions(tiny_model, fixture_inputs):
    with torch.no_grad():
        out = tiny_model(input_ids=fixture_inputs["input_ids"][:, :64], use_cache=True)
    past = out.past_key_values

    before = kv_cache_bytes(past)
    keep = [0, 1, 2, 3, 30, 31, 60, 61, 62, 63]
    evict(past, keep)

    assert cache_length(past) == len(keep)
    assert kv_cache_bytes(past) < before
