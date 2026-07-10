# Adaptive KV Cache Compression

An original KV cache compression policy for long-context LLM inference, benchmarked honestly against published baselines (StreamingLLM, H2O, SnapKV) on the quality / memory / latency triangle.

**The gap this attacks:** existing methods apply one static heuristic uniformly. Real workloads want *adaptive* budget allocation — different heads, layers, and phases (prefill vs decode) deserve different retention. This project measures that and builds a policy around it.

**Model:** Qwen2.5-1.5B-Instruct or Llama-3.2-1B (both run on free-tier GPUs).
**Focus:** decode path, no training/finetuning, pure PyTorch.

## Status

🚧 Phase 0 — model bring-up and manual decode loop.

## Where things are

- [`docs/spec.md`](docs/spec.md) — full project spec, phase plan, and success criteria.

## Reproducing (once there's something to reproduce)

_Coming as phases land: one script per experiment, deterministic seeds, results dumped as JSON so the headline plot regenerates from raw data._
