# Adaptive KV Cache Compression

[![ci](https://github.com/aarohigandhi/adaptive-kv-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/aarohigandhi/adaptive-kv-cache/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A KV cache compression method for long context inference, measured honestly against three published baselines (StreamingLLM, H2O, SnapKV) on a small open model.

When a language model generates text it keeps a running memory of every token it has seen, called the KV cache. That memory grows with the length of the context, so long prompts eat GPU memory and slow generation down. Compression throws away part of the cache to keep it small without hurting the answers too much. The three baselines each keep a fixed shape of the cache. StreamingLLM in particular keeps the start and the recent tokens but drops the whole middle, so a fact buried in the middle is lost. This project builds a method that spends the same budget more carefully.

Model: Qwen2.5 1.5B Instruct, which runs on a free Colab T4.

## The method: Anchored Skeleton

Within a fixed token budget it keeps three things:

* Anchors: the first few tokens, which the model leans on as an attention sink.
* Recent window: the most recent tokens, for local coherence.
* Skeleton: a handful of evenly spaced chunks through the middle, a coarse memory of the whole context.

The idea is simple. The other methods that never read the question forget the middle entirely, so a fact in the middle is unrecoverable. Keeping a thin skeleton of the middle gives that fact a chance to survive.

There is also a query aware version, `adaptive_qa`, that places the middle chunks where attention is highest instead of spacing them evenly.

## What counts as working

Preregistered in [docs/spec.md](docs/spec.md) before the evaluation was run, so the
bar could not move afterwards:

> At a 25% cache budget, the method matches or beats the best baseline on at least
> 2 of 4 quality tasks while staying within 5% of its latency.

The four quality tasks are the LongBench subsets qasper, hotpotqa, gov_report and
samsum. The needle sweep below and the PG19 perplexity are reported next to them but
do not vote: the needle task is the one the method was designed around, so letting it
score its own criterion would be marking my own homework.

**Verdict: not yet earned.** The grid that decides it is written and tested but has
not been run on a GPU. `scripts/run_phase4.py` produces it and
`scripts/report_phase4.py` prints the pass or fail; until those records exist the
verdict code refuses to return a pass. Whatever it comes back as goes in this README,
including a loss.

## Result so far: the needle sweep

The test hides a code in a long prompt and then asks for it. The budget is small enough that the middle has to be evicted. Each cell below is 9 runs: 3 codes at 3 depths (0.1, 0.5, 0.9 through the prompt).

| cache budget | full | streaming_llm | h2o | snapkv | adaptive (ours) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 48  | 100% | 0%  | 0%  | 0%   | 0%   |
| 96  | 100% | 33% | 0%  | 0%   | 0%   |
| 192 | 100% | 33% | 33% | 100% | 67%  |
| 288 | 100% | 67% | 33% | 100% | 100% |

![Needle retrieval accuracy vs cache budget](results/needle_accuracy.png)

What this says:

* Among the methods that never read the question (streaming_llm, h2o, ours), ours retrieves best at budgets 192 and 288, because it keeps a trace of the middle where the others drop it.
* At budget 288 it reaches 100%, matching SnapKV, even though SnapKV gets to peek at the question and ours does not.
* H2O does worst on this task, exactly as expected. It keeps the tokens the prompt paid attention to, but nobody attends to the code until the question is asked, so it evicts the code first.

## Layout

```
src/akvc/
  model.py            load the model and run the decode loops
  cache_manager.py    trim the cache (evict and cache_length)
  instrumentation.py  measure memory and time
  policies/           full, streaming_llm, h2o, snapkv, adaptive
  eval/               the tasks: needle, LongBench, PG19 perplexity, and the harness
scripts/              one runnable experiment each
tests/                CPU tests, no GPU and no model download
results/              json and plots
docs/spec.md          the full project spec
```

## Running it

Each script is standalone. Run in Colab with a GPU, or any CUDA machine.

```bash
git clone https://github.com/aarohigandhi/adaptive-kv-cache.git
cd adaptive-kv-cache

pip install -r requirements.txt

python scripts/run_phase4.py         # the preregistered grid: every task, every ratio
python scripts/report_phase4.py      # tables, chart and verdict, from the saved json
python scripts/run_eval.py           # the absolute budget needle sweep (the table above)
python scripts/baseline_sweep.py     # memory and latency vs context length
python scripts/eviction_demo.py      # StreamingLLM caps the cache
python scripts/profile_attention.py  # attention entropy per head
python scripts/h2o_demo.py           # H2O vs StreamingLLM on a long story
python scripts/snapkv_demo.py        # SnapKV's prefill compression
python scripts/adaptive_demo.py      # our method vs baselines on one needle
python scripts/adaptive_qa_demo.py   # the query aware variant
python scripts/tune_adaptive.py      # the sweep that picked our defaults
```

`run_eval.py` writes `results/eval_needle.json` and the chart above.
`run_phase4.py` writes `results/phase4.json`; everything after that regenerates from
it without a GPU, so the tables and the verdict are never typed in by hand.

The two needle scripts answer different questions and both are kept. `run_eval.py`
sweeps absolute token budgets, which shows where each policy breaks. `run_phase4.py`
sweeps budgets as a fraction of the prompt, which is the axis the published baselines
report on and the only one where a number here can be set next to a number there.

Tests run on CPU in about 20 seconds with no model download, since they build a tiny
Qwen2 from config:

```bash
pip install -r requirements-dev.txt
pytest tests/
```

The one that matters most is `test_manual_decode_matches_generate`: the hand written
decode loop has to produce the same tokens as `model.generate` on a 4K context. If
that ever fails, every number in this repo is measuring a bug instead of a policy.

## Caveats

* One small model, so the numbers may not carry over to larger ones.
* The needle sweep uses a handful of codes and depths. It shows the effect but is not a full statistical study.
* H2O and SnapKV use a simplified single budget version that pools attention across heads, not the per head original.
* Compression barely moves peak memory, and the repo measures both numbers rather than the flattering one. Prefill builds the whole cache before any policy gets to evict, so the high water mark for a run is roughly the prompt either way. What a policy collapses is the cache held during decode, which is what matters for long generations and what the cost tables report.
* The needle table above was produced at 3 depths. `needle.py` now sweeps 5, so rerunning widens it rather than reproducing it exactly.
