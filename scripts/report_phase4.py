"""Tables, chart and verdict from results/phase4.json. No GPU needed.

Run this after run_phase4.py, or on its own to redraw everything from saved records.
It prints the quality table, the cost table, and the pass or fail against the
preregistered criterion, then writes results/phase4_quality.png.

    python scripts/report_phase4.py
    python scripts/report_phase4.py --markdown    # tables ready to paste in the README
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from akvc.eval import phase4
from akvc.eval.harness import POLICIES, BUDGET_RATIOS

COLORS = {
    "full": "#9aa4b2", "streaming_llm": "#2a6fdb", "h2o": "#e8710a",
    "snapkv": "#2ca02c", "adaptive": "#d62728", "adaptive_qa": "#9467bd",
}


def _fmt(value, task):
    if value is None:
        return "-"
    return f"{value:.2f}" if task == "pg19" else f"{value:.3f}"


def quality_table(records, task, policies, ratios, markdown=False):
    header = ["policy"] + [f"keep {r:.1%}" for r in ratios]
    rows = [[p] + [_fmt(phase4.mean_score(records, task, p, r), task) for r in ratios]
            for p in policies]
    return _render(header, rows, markdown)


def cost_table(records, task, policies, ratios, markdown=False):
    """Decode latency and the cache held during decode. Not the overall peak: prefill
    builds the full cache for every policy, so that column would be identical."""
    header = ["policy"] + [f"keep {r:.1%}" for r in ratios]
    rows = []
    for p in policies:
        cells = []
        for r in ratios:
            ms = phase4.mean_latency(records, task, p, r)
            mb = phase4.mean_kv_mb(records, task, p, r)
            if ms is None:
                cells.append("-")
            elif mb is None:
                cells.append(f"{ms:.1f} ms")
            else:
                cells.append(f"{ms:.1f} ms / {mb:.1f} MB" if mb < 10
                             else f"{ms:.1f} ms / {mb:.0f} MB")
        rows.append([p] + cells)
    return _render(header, rows, markdown)


def _render(header, rows, markdown):
    if markdown:
        align = "|:---" + "|:---:" * (len(header) - 1) + "|"
        lines = ["| " + " | ".join(header) + " |", align]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    out = ["  ".join(h.ljust(w) for h, w in zip(header, widths))]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def verdict_text(v):
    """The pass or fail sentence plus the per task working."""
    if not v["complete"]:
        missing = [r["task"] for r in v["rows"] if r["status"] == "missing"]
        return (f"VERDICT: not yet decided. Missing results for {missing} at "
                f"keep {v['ratio']:.1%}. Run those tasks before claiming anything.")

    lines = [f"VERDICT: {'PASS' if v['passed'] else 'FAIL'}. "
             f"Won {v['wins']} of {len(v['tasks'])} quality tasks at keep "
             f"{v['ratio']:.1%}, needed {v['needed']}."]
    for r in v["rows"]:
        why = []
        if not r["quality_ok"]:
            why.append("quality below best baseline")
        if not r["latency_ok"]:
            why.append("latency over the 5% allowance")
        note = "won" if r["status"] == "won" else "lost: " + ", ".join(why)
        lines.append(f"  {r['task']:>12}  ours {r['ours']:.3f} vs "
                     f"{r['best_baseline']} {r['best_baseline_score']:.3f}  ({note})")
    return "\n".join(lines)


def plot(records, policies, ratios, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = [t for t in ["needle"] + phase4.QUALITY_TASKS
             if any(r["task"] == t for r in records)]
    if not tasks:
        print("Nothing to plot.")
        return

    cols = min(3, len(tasks))
    rows = (len(tasks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    xs = [r * 100 for r in ratios]

    for ax, task in zip(axes.flat, tasks):
        for policy in policies:
            ys = [phase4.mean_score(records, task, policy, r) for r in ratios]
            if all(y is None for y in ys):
                continue
            ax.plot(xs, ys, marker="o", label=policy,
                    linewidth=2.8 if policy == phase4.OURS else 1.8,
                    color=COLORS.get(policy))
        ax.set_title(task, fontsize=11)
        ax.set_xlabel("cache kept (%)")
        ax.set_ylabel("needle accuracy" if task == "needle" else
                      phase4.longbench.TASKS[task]["metric"])
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(tasks):]:
        ax.set_visible(False)

    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Quality vs cache budget", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print("Saved", out_path)


def report(records, policies=None, ratios=None, markdown=False,
           out_path="results/phase4_quality.png"):
    policies = policies or [p for p in POLICIES if any(r["policy"] == p for r in records)]
    ratios = ratios or sorted({r["ratio"] for r in records}, reverse=True)
    tasks = [t for t in ["needle"] + phase4.QUALITY_TASKS + ["pg19"]
             if any(r["task"] == t for r in records)]

    for task in tasks:
        label = "perplexity, lower is better" if task == "pg19" else "score"
        print(f"\n{task} ({label})")
        print(quality_table(records, task, policies, ratios, markdown))
        print(f"\n{task} cost (decode latency / peak KV cache)")
        print(cost_table(records, task, policies, ratios, markdown))

    print()
    print(verdict_text(phase4.verdict(records)))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plot(records, policies, ratios, out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", default="results/phase4.json")
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.records):
        raise SystemExit(f"{args.records} not found. Run scripts/run_phase4.py first.")
    report(phase4.load(args.records), markdown=args.markdown)


if __name__ == "__main__":
    main()
