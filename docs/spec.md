# Full Spec: Project A — Adaptive Context Compression for Long Context Inference

## One-line pitch

Design, implement, and rigorously benchmark an original KV cache compression policy that beats at least one published baseline (StreamingLLM, H2O, or SnapKV) on at least one axis of the quality vs memory vs latency triangle, on a real open model, with honest numbers.

## The novelty hook (what makes this not a clone)

Existing baselines mostly apply one static heuristic uniformly. StreamingLLM keeps sink tokens plus a recency window. H2O keeps heavy hitters by cumulative attention. SnapKV selects by observation-window attention at prefill. The open gap: **budget allocation is not adaptive across heads, layers, and workload phases.** Different heads want different retention policies, and the right budget shifts between prefill-heavy and decode-heavy phases.

Three candidate directions, pick one after Phase 2 profiling:

1. **Entropy-guided per-head budgets.** Measure attention entropy per head online. Concentrated heads (low entropy) get tiny budgets since a few tokens carry the signal. Diffuse heads keep more. Allocate a global memory budget across heads proportionally, recomputed every N decode steps.
2. **Layer-wise pyramid with learned breakpoints.** Later layers need less cache. Instead of a fixed pyramid schedule, fit the per-layer budget curve on a tiny calibration set per model, then freeze it. The finding is the fitted curve and how stable it is across tasks.
3. **Hybrid score with cheap recovery.** Combine recency, cumulative attention, and a novelty term (distance of the new key from retained keys), plus a small quarantine buffer so recently evicted entries can be recovered before final eviction. Claim: recovery reduces the catastrophic misses that pure heavy-hitter policies suffer on needle tasks.

Any one, validated honestly, is a genuine result. If your method loses everywhere, that converts cleanly into a negative-result writeup.

## Scope and non-goals

- Small open model only: Qwen2.5-1.5B-Instruct or Llama-3.2-1B.
- Decode path focus. No training, no finetuning, no RLHF.
- Not a production server. Single request first, simple batching as a stretch goal.
- No CUDA kernel work required. Pure PyTorch. A Triton fused kernel for eviction scoring is an optional bridge to a later project.

## Architecture

```
prompt -> tokenizer -> prefill (build KV cache, capture attention stats)
      -> compression controller (your policy lives here)
      -> decode loop (paged KV read, incremental eviction)
      -> metrics tap (memory, latency, attention logs)
      -> eval harness (LongBench subset, needle test, perplexity)
```

Key components:

- **Cache manager:** owns KV tensors per layer/head, exposes `evict(indices)` and `stats()`. Design this interface first — it's the thing interviewers probe.
- **Policy module:** pluggable. Baselines and your method implement the same interface, so comparisons are apples-to-apples by construction.
- **Instrumentation:** peak memory via `torch.cuda.max_memory_allocated`, per-token latency with proper CUDA synchronization, attention maps sampled (not stored fully).
- **Eval harness:** deterministic seeds, config files per run, results dumped as JSON so plots regenerate from raw data.

## Phase plan (8–10 weeks)

- **Phase 0, week 1 — environment and model bring-up.** Load the model in HuggingFace, write your own greedy decode loop that manages the KV cache manually instead of `model.generate`. Verify outputs match `generate` token-for-token.
- **Phase 1, weeks 2–3 — baseline + instrumentation.** Full-cache baseline with clean measurements. First plot: memory and latency vs context length (1K–32K). Establish the eval harness now.
- **Phase 2, weeks 4–5 — reimplement two baselines.** StreamingLLM (easy) and H2O or SnapKV (moderate). Reproduce their claims. Log per-head attention entropy across tasks; that profiling decides the novelty direction.
- **Phase 3, weeks 6–7 — your method.** Implement the chosen direction. Iterate on a small fixed slice, then lock the method. Preregister success criteria in the repo before big runs.
- **Phase 4, week 8 — full evaluation.** Grid over compression ratios (keep 50/25/12.5%). Tasks: needle-in-a-haystack sweep, 3–4 LongBench subsets (qasper, hotpotqa, gov_report, samsum), perplexity on PG19. Metrics: task score, peak KV memory, decode tok/s, TTFT.
- **Phase 5, weeks 9–10 — writeup and polish.** Short paper-style writeup: claim, method, results, wins, losses, why. Clean README with the headline plot. Optional blog post.

## Success criteria (preregister these)

- **Primary:** at a 25% cache budget, your method matches or beats the best baseline on ≥2 of 4 quality tasks while staying within 5% of its latency.
- **Secondary:** a clear, explainable failure-mode analysis.
- **Engineering bar:** reproducible from a fresh clone with one script per experiment.

## Compute budget

Colab/Kaggle free tier covers Phases 0–3 with a 1B model. Phase 4 grids may want ~10–20 rented A100 hours (~$15–30). Keep dev loops on a 4K context slice so iteration stays free.

## Deliverables

1. Public repo: cache manager, policy interface, three baselines, your method, eval harness.
2. Results writeup with quality-vs-memory curves and an honest one-paragraph verdict.
3. Headline README plot: quality (y) vs cache budget (x), your curve vs baselines.
4. Optional: short post; a note to SnapKV/H2O authors if numbers are interesting.

## Interview talking points to rehearse

- Why manual cache management instead of `model.generate`, and what broke first.
- The eviction interface design: why policy is pluggable, what state each policy needs, how comparisons stay fair.
- A tradeoff story: something that made latency worse, and how measurement caught it.
- The failure mode of heavy-hitter policies on needle tasks, and whether your method fixed it.
- What you'd do with 100x compute.

## Risks and fallbacks

- **Method underperforms everywhere:** pivot to a rigorous negative result. Still a flagship.
- **HuggingFace internals fight you:** drop to a smaller pure-PyTorch reimplementation of the forward pass.
- **Time crunch:** ship Phases 0–2 + the profiling study as a standalone artifact, finish the method later.

## Definition of done

A stranger can clone the repo, run one script, regenerate the headline plot, and read a writeup that states in one sentence what your method does that StreamingLLM, H2O, and SnapKV do not.
