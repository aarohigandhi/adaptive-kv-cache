"""The scoring functions, the needle prompts, and the verdict logic.

The verdict is the claim the whole repo rests on, so it is tested against handmade
records: a clear win, a quality win given back by latency, and an incomplete run
that must refuse to return a pass.
"""

import pytest

from akvc.eval import phase4
from akvc.eval.metrics import normalize, rouge_l, token_f1
from akvc.eval.needle import DEPTHS, NEEDLES, found, make_prompt


def test_normalize_strips_case_punctuation_and_articles():
    assert normalize("The Vault, passcode!") == "vault passcode"


def test_token_f1_edges():
    assert token_f1("the answer is 42", "answer is 42") == pytest.approx(1.0)
    assert token_f1("completely different", "answer is 42") == 0.0
    assert 0 < token_f1("the answer is 43", "the answer is 42") < 1


def test_rouge_l_rewards_order():
    assert rouge_l("a b c d", "a b c d") == pytest.approx(1.0)
    in_order = rouge_l("a b c", "a x b x c")
    shuffled = rouge_l("c b a", "a x b x c")
    assert in_order > shuffled


@pytest.mark.parametrize("depth", DEPTHS)
def test_needle_prompt_contains_the_needle_and_the_question(depth):
    needle = NEEDLES[0]
    prompt = make_prompt(needle, depth, n_filler=32)
    assert needle in prompt
    assert "passcode" in prompt
    assert found(f"The passcode is {needle}.", needle)
    assert not found("I do not know.", needle)


def test_needle_depth_moves_the_needle_through_the_prompt():
    shallow = make_prompt(NEEDLES[0], 0.1, n_filler=32).index(NEEDLES[0])
    deep = make_prompt(NEEDLES[0], 0.9, n_filler=32).index(NEEDLES[0])
    assert shallow < deep


def _records(ours_scores, baseline_scores, ours_ms=10.0, base_ms=10.0):
    out = []
    for task, score in ours_scores.items():
        out.append({"task": task, "policy": "adaptive", "ratio": 0.25, "score": score,
                    "decode_ms_per_token": ours_ms})
    for task, score in baseline_scores.items():
        out.append({"task": task, "policy": "snapkv", "ratio": 0.25, "score": score,
                    "decode_ms_per_token": base_ms})
    return out


def test_verdict_passes_on_two_clean_wins():
    ours = {"qasper": 0.6, "hotpotqa": 0.6, "gov_report": 0.2, "samsum": 0.2}
    base = {"qasper": 0.5, "hotpotqa": 0.5, "gov_report": 0.3, "samsum": 0.3}
    v = phase4.verdict(_records(ours, base))
    assert v["complete"] and v["passed"] and v["wins"] == 2


def test_verdict_fails_when_latency_eats_the_win():
    ours = {"qasper": 0.6, "hotpotqa": 0.6, "gov_report": 0.6, "samsum": 0.6}
    base = {"qasper": 0.5, "hotpotqa": 0.5, "gov_report": 0.5, "samsum": 0.5}
    v = phase4.verdict(_records(ours, base, ours_ms=20.0, base_ms=10.0))
    assert not v["passed"]
    assert all(r["quality_ok"] and not r["latency_ok"] for r in v["rows"])


def test_verdict_allows_latency_inside_the_five_percent_allowance():
    ours = {"qasper": 0.6, "hotpotqa": 0.6, "gov_report": 0.6, "samsum": 0.6}
    base = {"qasper": 0.5, "hotpotqa": 0.5, "gov_report": 0.5, "samsum": 0.5}
    v = phase4.verdict(_records(ours, base, ours_ms=10.4, base_ms=10.0))
    assert v["passed"]


def test_verdict_refuses_to_decide_on_a_partial_run():
    ours = {"qasper": 0.9, "hotpotqa": 0.9}
    base = {"qasper": 0.1, "hotpotqa": 0.1}
    v = phase4.verdict(_records(ours, base))
    assert not v["complete"]
    assert not v["passed"], "a partial run must never report a pass"


def test_pg19_aggregates_as_perplexity_not_as_a_mean():
    records = [
        {"task": "pg19", "policy": "adaptive", "ratio": 0.25, "nll": 200.0, "tokens": 100},
        {"task": "pg19", "policy": "adaptive", "ratio": 0.25, "nll": 100.0, "tokens": 100},
    ]
    # exp(300 / 200) = exp(1.5)
    assert phase4.mean_score(records, "pg19", "adaptive", 0.25) == pytest.approx(4.4816891)


def test_done_keys_and_pending_drive_resume():
    """A resumed run must skip what it already has and run what it does not."""
    records = _records({"qasper": 0.6}, {"qasper": 0.5})
    done = phase4.done_keys(records)
    assert done == {("qasper", "adaptive", 0.25), ("qasper", "snapkv", 0.25)}

    ratios = [0.5, 0.25, 0.125]
    assert phase4._pending("qasper", "adaptive", ratios, done) == [0.5, 0.125]
    assert phase4._pending("hotpotqa", "adaptive", ratios, done) == ratios
    assert phase4._pending("qasper", "adaptive", ratios, None) == ratios


def test_resume_treats_a_fanned_out_full_baseline_as_complete():
    """full is measured once and copied to every ratio, so seeing it at any ratio
    means it is done. Without this a resume would rerun the most expensive baseline."""
    ratios = [0.5, 0.25, 0.125]
    records = phase4._fan_out(
        [{"task": "gov_report", "policy": "full", "ratio": 0.5, "score": 0.3}], ratios)
    done = phase4.done_keys(records)
    assert phase4._pending("gov_report", "full", ratios, done) == []
    assert phase4._pending("gov_report", "adaptive", ratios, done) == ratios
