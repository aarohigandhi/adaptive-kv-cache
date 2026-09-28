"""Shared fixtures.

The tests run on CPU with no network and no model download. A tiny randomly
initialized Qwen2 stands in for the real one: it is the same architecture, so the
cache layout, the position handling and the attention outputs are all genuine, and
it is small enough that a 4K context costs nothing. The weights are nonsense, which
does not matter, because these tests check mechanics rather than answer quality.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SEED = 0
FIXTURE_CONTEXT = 4096


class StubTokenizer:
    """Just enough tokenizer for the decode loops, which only ask for eos."""

    eos_token_id = -1
    pad_token_id = 0


@pytest.fixture(scope="session")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(SEED)
    config = Qwen2Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8192,
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    model.generation_config.pad_token_id = 0
    return model


@pytest.fixture(scope="session")
def tokenizer():
    return StubTokenizer()


@pytest.fixture(scope="session")
def fixture_inputs(tiny_model):
    """A 4K token context, the length the policies are supposed to handle."""
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, tiny_model.config.vocab_size, (1, FIXTURE_CONTEXT))
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
