# Adaptive KV Cache Compression

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

## Result

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
  eval/               the needle task and the harness
scripts/              one runnable experiment each
results/              json and plots
docs/spec.md          the full project spec
```

## Running it

Each script is standalone. Run in Colab with a GPU, or any CUDA machine.

```bash
git clone https://github.com/aarohigandhi/adaptive-kv-cache.git
cd adaptive-kv-cache

pip install -r requirements.txt

python scripts/run_eval.py           # the full accuracy vs budget sweep (the table above)
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

## Caveats

* One small model, so the numbers may not carry over to larger ones.
* The needle sweep uses a handful of codes and depths. It shows the effect but is not a full statistical study.
* H2O and SnapKV use a simplified single budget version that pools attention across heads, not the per head original.
