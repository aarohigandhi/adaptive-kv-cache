"""Chunked prefill has to be an optimization, not a change of method.

Reading attention scores means asking for output_attentions, which returns one
(heads, queries, keys) tensor per layer all at once. At a 4K prompt that is about
22 GB for this model, so the prompt is prefilled a chunk at a time and each chunk's
attention is folded into a running tally and dropped.

That is only legitimate if it produces the same numbers as the single pass version.
These tests check exactly that: same logits, same importance, same cache. Without
them the saving would be silently buying a different policy.
"""

import pytest
import torch

from akvc.model import PREFILL_CHUNK, _init_importance, chunked_prefill
from akvc.cache_manager import cache_length


@torch.no_grad()
def single_pass(model, input_ids):
    out = model(input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=True, output_attentions=True)
    return out.past_key_values, out.logits, _init_importance(out.attentions)


@pytest.fixture(scope="module")
def prompt(tiny_model):
    torch.manual_seed(3)
    return torch.randint(0, tiny_model.config.vocab_size, (1, 600))


@pytest.mark.parametrize("chunk_size", [64, 128, PREFILL_CHUNK, 1024])
def test_chunked_matches_single_pass(tiny_model, prompt, chunk_size):
    ref_past, ref_logits, ref_importance = single_pass(tiny_model, prompt)
    past, logits, importance, _ = chunked_prefill(
        tiny_model, prompt, chunk_size=chunk_size, need_importance=True)

    assert cache_length(past) == cache_length(ref_past) == prompt.shape[1]
    assert importance.shape == ref_importance.shape
    torch.testing.assert_close(importance, ref_importance, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(logits[:, -1, :], ref_logits[:, -1, :],
                               rtol=1e-4, atol=1e-4)


def test_importance_covers_every_token(tiny_model, prompt):
    _, _, importance, _ = chunked_prefill(tiny_model, prompt, need_importance=True)
    assert importance.shape[0] == prompt.shape[1]
    assert (importance >= 0).all()
    # Causal attention means the first token is seen by every query and the last by
    # one, so the tally must not be flat or reversed.
    assert importance[0] > importance[-1]


def test_observation_window_scores_only_the_last_queries(tiny_model, prompt):
    """SnapKV's score. It must come from the final tokens of the prompt, not all of
    them, or it stops being SnapKV."""
    window = 32
    _, _, importance, window_importance = chunked_prefill(
        tiny_model, prompt, need_importance=True, observation_window=window)

    assert window_importance is not None
    assert window_importance.shape[0] == prompt.shape[1]
    # The window's own tokens are the most recent keys; everything after the window
    # start is attended to by fewer queries, and past the end there is nothing.
    assert window_importance.sum() < importance.sum()

    ref = single_pass(tiny_model, prompt)[0]
    del ref

    # Scored off exactly the last `window` queries.
    with torch.no_grad():
        out = tiny_model(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                         use_cache=True, output_attentions=True)
    expected = None
    for a in out.attentions:
        score = a[0, :, -window:, :].float().sum(dim=0).sum(dim=0)
        expected = score if expected is None else expected + score
    torch.testing.assert_close(window_importance, expected, rtol=1e-4, atol=1e-4)


def test_plain_prefill_skips_the_chunking(tiny_model, prompt):
    past, logits, importance, window = chunked_prefill(tiny_model, prompt,
                                                       need_importance=False)
    assert importance is None and window is None
    assert cache_length(past) == prompt.shape[1]
    assert logits.shape[1] == prompt.shape[1]


def test_chunking_does_not_change_the_generated_tokens(tiny_model, tokenizer, prompt):
    """The end to end version of the same claim, through the real decode loop."""
    from akvc.eval.harness import make_policy
    from akvc.model import decode_with_policy

    inputs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}
    runs = []
    for chunk in (64, 4096):
        import akvc.model as model_module
        original = model_module.PREFILL_CHUNK
        model_module.PREFILL_CHUNK = chunk
        try:
            runs.append(decode_with_policy(tiny_model, tokenizer, inputs,
                                           make_policy("h2o"), budget=128,
                                           max_new_tokens=6))
        finally:
            model_module.PREFILL_CHUNK = original

    assert torch.equal(runs[0], runs[1]), "chunk size changed the output"
