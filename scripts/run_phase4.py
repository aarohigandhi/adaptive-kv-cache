"""The Phase 4 evaluation: every policy on every task at keep ratios of 50, 25 and
12.5 percent, with latency and cache size recorded on the same runs.

Writes results/phase4.json, then prints the tables and the verdict against the
preregistered success criterion and saves the headline chart. Nothing after the run
needs the GPU: scripts/report_phase4.py regenerates all of it from the JSON.

    python scripts/run_phase4.py                      # everything, the defaults below
    python scripts/run_phase4.py --tasks needle       # just the needle sweep
    python scripts/run_phase4.py --samples 5 --max-context 2048    # a cheap dry run

The LongBench and PG19 tasks need the datasets package and a network connection the
first time. Start with --tasks needle if you only have a free tier GPU; the rest is
where the real compute goes.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.eval import phase4
from akvc.eval.harness import POLICIES, BUDGET_RATIOS
from akvc.model import load_model
from report_phase4 import report

ALL_TASKS = ["needle"] + phase4.QUALITY_TASKS + ["pg19"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default=",".join(ALL_TASKS),
                   help=f"comma separated, from {ALL_TASKS}")
    p.add_argument("--policies", default=",".join(POLICIES))
    p.add_argument("--ratios", default=",".join(str(r) for r in BUDGET_RATIOS))
    p.add_argument("--samples", type=int, default=20,
                   help="LongBench samples per subset")
    p.add_argument("--max-context", type=int, default=4096,
                   help="truncate LongBench contexts to this many tokens")
    p.add_argument("--needle-filler", type=int, default=64,
                   help="filler repeats in the needle prompt, about 30 tokens each")
    p.add_argument("--pg19-passages", type=int, default=3)
    p.add_argument("--pg19-prefill", type=int, default=2048)
    p.add_argument("--pg19-eval-tokens", type=int, default=512)
    p.add_argument("--out", default="results/phase4.json")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    ratios = [float(r) for r in args.ratios.split(",") if r.strip()]

    unknown = set(tasks) - set(ALL_TASKS)
    if unknown:
        raise SystemExit(f"Unknown tasks: {sorted(unknown)}. Choose from {ALL_TASKS}.")

    tokenizer, model = load_model(attn_implementation="eager")
    print(f"Device {model.device}, dtype {model.dtype}")
    print(f"Tasks {tasks} | policies {policies} | ratios {ratios}\n")

    records = []
    if "needle" in tasks:
        records += phase4.run_needle(model, tokenizer, policies, ratios,
                                     n_filler=args.needle_filler)

    lb = [t for t in tasks if t in phase4.QUALITY_TASKS]
    if lb:
        records += phase4.run_longbench(model, tokenizer, policies, ratios, lb,
                                        n_samples=args.samples,
                                        max_context_tokens=args.max_context)

    if "pg19" in tasks:
        records += phase4.run_pg19(model, tokenizer, policies, ratios,
                                   n_passages=args.pg19_passages,
                                   n_tokens=args.pg19_prefill + args.pg19_eval_tokens,
                                   n_prefill=args.pg19_prefill)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    phase4.save(records, args.out)
    print(f"\nSaved {args.out} ({len(records)} records)")

    report(records, policies=policies, ratios=ratios)


if __name__ == "__main__":
    main()
