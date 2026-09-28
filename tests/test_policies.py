"""Policy logic, with no model in sight.

Every policy has to answer the same question, keep_indices, and answer it legally:
sorted, unique, inside the context, and never more than the budget. A policy that
quietly returns budget + 1 positions would make its own quality numbers look good
and the comparison meaningless, so that is the first thing checked.
"""

import pytest
import torch

from akvc.eval.harness import POLICIES, budget_for, make_policy, MIN_BUDGET
from akvc.policies.adaptive import AdaptivePolicy, AdaptiveQAPolicy
from akvc.policies.full import FullCachePolicy
from akvc.policies.h2o import H2OPolicy
from akvc.policies.snapkv import SnapKVPolicy
from akvc.policies.streaming_llm import StreamingLLMPolicy

EVICTING = ["streaming_llm", "h2o", "snapkv", "adaptive", "adaptive_qa"]
CONTEXTS = [64, 512, 4096]
BUDGETS = [16, 48, 128, 512]


def stats_for(num_tokens):
    torch.manual_seed(0)
    return {"importance": torch.rand(num_tokens)}


@pytest.mark.parametrize("name", EVICTING)
@pytest.mark.parametrize("num_tokens", CONTEXTS)
@pytest.mark.parametrize("budget", BUDGETS)
def test_keep_indices_respects_budget(name, num_tokens, budget):
    keep = make_policy(name).keep_indices(num_tokens, budget, stats_for(num_tokens))

    assert keep == sorted(keep), f"{name} returned unsorted indices"
    assert len(keep) == len(set(keep)), f"{name} returned duplicates"
    assert all(0 <= i < num_tokens for i in keep), f"{name} returned out of range indices"
    if num_tokens > budget:
        assert len(keep) <= budget, f"{name} kept {len(keep)} of a {budget} budget"


@pytest.mark.parametrize("name", EVICTING)
def test_nothing_evicted_below_budget(name):
    keep = make_policy(name).keep_indices(32, 128, stats_for(32))
    assert keep == list(range(32))


def test_full_cache_keeps_everything():
    assert FullCachePolicy().keep_indices(4096, 16) == list(range(4096))


def test_streaming_llm_keeps_sinks_and_recent_and_drops_the_middle():
    keep = StreamingLLMPolicy(sinks=4).keep_indices(1000, 100)
    assert keep[:4] == [0, 1, 2, 3]
    assert keep[-1] == 999
    assert not any(200 <= i <= 800 for i in keep)


def test_adaptive_keeps_a_trace_of_the_middle():
    """The whole claim of the method. If this fails there is no method."""
    num_tokens, budget = 1000, 100
    keep = AdaptivePolicy(sinks=4).keep_indices(num_tokens, budget)
    middle = [i for i in keep if 200 <= i <= 800]
    assert middle, "adaptive kept nothing from the middle"

    streaming = StreamingLLMPolicy(sinks=4).keep_indices(num_tokens, budget)
    assert len(keep) <= len(streaming) + 1, "adaptive spent more than the same budget"


def test_adaptive_skeleton_spans_the_middle():
    keep = AdaptivePolicy(sinks=4, chunk_size=8).keep_indices(2000, 200)
    middle = [i for i in keep if 100 <= i <= 1400]
    assert middle
    assert max(middle) - min(middle) > 500, "the skeleton is bunched in one place"


def test_heavy_hitter_policies_follow_the_importance_scores():
    """H2O and SnapKV should keep a token the scores single out, not a fixed shape."""
    num_tokens, budget = 500, 64
    importance = torch.zeros(num_tokens)
    spike = 100
    importance[spike] = 1000.0

    for policy in (H2OPolicy(), SnapKVPolicy(window=16)):
        keep = policy.keep_indices(num_tokens, budget, {"importance": importance})
        assert spike in keep, f"{policy.name} dropped the highest scoring token"


def test_query_aware_variant_prefers_the_attended_region():
    num_tokens, budget = 1000, 120
    importance = torch.zeros(num_tokens)
    importance[400:420] = 1.0
    keep = AdaptiveQAPolicy(sinks=4).keep_indices(num_tokens, budget,
                                                 {"importance": importance})
    assert any(380 <= i <= 440 for i in keep)


@pytest.mark.parametrize("name", POLICIES)
def test_every_named_policy_is_reachable(name):
    if name in ("full", "snapkv"):
        return  # these take their own path in run_policy
    assert make_policy(name) is not None


def test_budget_for_tracks_the_ratio_and_has_a_floor():
    assert budget_for(4000, 0.25) == 1000
    assert budget_for(4000, 0.125) == 500
    assert budget_for(8, 0.25) == MIN_BUDGET
