"""The PG19 scoring loop and the reporting path, on the tiny model.

load_passages needs the network so it is not tested here; passage_nll, which is
the part with the logic in it, is tested on synthetic token ids.
"""

import json
import math

import pytest
import torch

from akvc.eval import phase4
from akvc.eval.harness import make_policy
from akvc.eval.perplexity import passage_nll, perplexity


@pytest.fixture(scope="module")
def ids(tiny_model):
    torch.manual_seed(1)
    return torch.randint(0, tiny_model.config.vocab_size, (320,)).tolist()


def test_perplexity_is_exp_of_mean_nll():
    assert perplexity(200.0, 100) == pytest.approx(math.exp(2.0))
    assert math.isnan(perplexity(0.0, 0))


def test_passage_nll_scores_every_token_after_the_prefill(tiny_model, ids):
    n_prefill = 256
    nll, count = passage_nll(tiny_model, ids, None, budget=0, n_prefill=n_prefill)
    assert count == len(ids) - n_prefill
    assert nll > 0 and math.isfinite(nll)


@pytest.mark.parametrize("name", ["streaming_llm", "h2o", "adaptive"])
def test_passage_nll_under_a_policy_stays_finite_and_reports_cost(tiny_model, ids, name):
    metrics = {}
    nll, count = passage_nll(tiny_model, ids, make_policy(name), budget=64,
                             n_prefill=256, metrics=metrics)
    assert count == len(ids) - 256
    assert math.isfinite(nll)
    assert metrics["peak_kv_bytes"] > 0
    assert metrics["decode_peak_kv_bytes"] > 0
    assert metrics["decode_ms_per_token"] > 0


def test_compressing_the_cache_costs_some_perplexity(tiny_model, ids):
    """Weights are random, so the absolute numbers are meaningless, but the full
    cache path and the compressed path must both produce usable numbers and the
    compressed one must hold less cache."""
    full_metrics, cut_metrics = {}, {}
    passage_nll(tiny_model, ids, None, 0, 256, metrics=full_metrics)
    passage_nll(tiny_model, ids, make_policy("adaptive"), 64, 256, metrics=cut_metrics)
    assert cut_metrics["decode_peak_kv_bytes"] < full_metrics["decode_peak_kv_bytes"]


def test_snapkv_compresses_once_and_then_leaves_the_cache_alone(tiny_model, ids):
    """Per step SnapKV would be a different method, so the flag has to change the
    behaviour. Compressing once lets the cache grow again afterwards; compressing
    every step pins it at the budget."""
    once, every = {}, {}
    passage_nll(tiny_model, ids, make_policy("snapkv"), 64, 256, metrics=once,
                compress_once=True)
    passage_nll(tiny_model, ids, make_policy("snapkv"), 64, 256, metrics=every,
                compress_once=False)
    assert once["final_cache_length"] > every["final_cache_length"]
    assert every["final_cache_length"] <= 64 + 1


def test_prefill_is_charged_to_every_policy(tiny_model, ids):
    """Prefill builds the whole cache before anything can be evicted, so a policy
    barely touches the overall peak however small its budget is. What it collapses
    is the cache held during decode. Reporting the overall peak as the saving would
    credit compression with memory it never saved, so both are measured."""
    full, cut = {}, {}
    passage_nll(tiny_model, ids, None, 0, 256, metrics=full)
    passage_nll(tiny_model, ids, make_policy("adaptive"), 64, 256, metrics=cut)

    overall_saving = full["peak_kv_bytes"] / cut["peak_kv_bytes"]
    decode_saving = full["decode_peak_kv_bytes"] / cut["decode_peak_kv_bytes"]
    assert overall_saving < 1.5, "the overall peak should hardly move"
    assert decode_saving > 2.0, "the decode peak should collapse"


def test_report_renders_tables_and_a_plot(tmp_path):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import report_phase4

    records = []
    for task in ["needle"] + phase4.QUALITY_TASKS:
        for policy in ["full", "snapkv", "adaptive"]:
            for ratio in [0.5, 0.25, 0.125]:
                records.append({"task": task, "policy": policy, "ratio": ratio,
                                "score": 0.5, "decode_ms_per_token": 10.0,
                                "peak_kv_bytes": 4e6, "decode_peak_kv_bytes": 1e6})

    md = report_phase4.quality_table(records, "qasper", ["snapkv", "adaptive"],
                                    [0.5, 0.25], markdown=True)
    assert md.startswith("| policy |")
    assert "0.500" in md

    cost = report_phase4.cost_table(records, "qasper", ["adaptive"], [0.25])
    assert "10.0 ms" in cost and "1.0 MB" in cost

    out = tmp_path / "plot.png"
    report_phase4.plot(records, ["snapkv", "adaptive"], [0.5, 0.25, 0.125], str(out))
    assert out.exists() and out.stat().st_size > 0


def test_full_cache_records_are_measured_once_and_fanned_out():
    """The full baseline ignores the budget, so it is run at one ratio and copied.
    The copies have to be marked, or a later reader would think it was measured
    three times."""
    mine = [{"task": "qasper", "policy": "full", "ratio": 0.5, "score": 0.42}]
    out = phase4._fan_out(mine, [0.5, 0.25, 0.125])
    assert len(out) == 3
    assert {r["ratio"] for r in out} == {0.5, 0.25, 0.125}
    assert all(r["score"] == 0.42 for r in out)
    copies = [r for r in out if "copied_from_ratio" in r]
    assert len(copies) == 2 and all(r["copied_from_ratio"] == 0.5 for r in copies)
    assert phase4._ratios_to_run("full", [0.5, 0.25, 0.125]) == [0.5]
    assert phase4._ratios_to_run("adaptive", [0.5, 0.25]) == [0.5, 0.25]


def test_records_round_trip_through_json(tmp_path):
    records = [{"task": "qasper", "policy": "adaptive", "ratio": 0.25, "score": 0.4}]
    path = tmp_path / "phase4.json"
    phase4.save(records, str(path))
    assert phase4.load(str(path)) == records
    assert json.loads(path.read_text())[0]["task"] == "qasper"
