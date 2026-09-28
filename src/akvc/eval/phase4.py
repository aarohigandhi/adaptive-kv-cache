"""The preregistered evaluation: every policy, every task, at keep ratios.

One record per run, written to JSON, so the tables and the verdict regenerate from
raw data without touching the GPU again. Every record carries the quality score and
the latency and cache size measured on the same run, because the success criterion
is about both at once.

The criterion, copied from docs/spec.md so it cannot drift:

    at a 25% cache budget, your method matches or beats the best baseline on >= 2
    of 4 quality tasks while staying within 5% of its latency.

The four quality tasks are the four LongBench subsets. The needle sweep and the
PG19 perplexity are reported next to them but are not part of the count: the
needle task is the one the method was designed around, so letting it vote on its
own criterion would be marking your own homework, and perplexity is a different
kind of number from a task score.
"""

import json

from akvc.eval import longbench
from akvc.eval.harness import POLICIES, BUDGET_RATIOS, run_policy, budget_for, make_policy
from akvc.eval.needle import make_prompt as make_needle_prompt, found, NEEDLES, DEPTHS
from akvc.eval.perplexity import load_passages, passage_nll, perplexity

QUALITY_TASKS = ["qasper", "hotpotqa", "gov_report", "samsum"]
OURS = "adaptive"
BASELINES = ["streaming_llm", "h2o", "snapkv"]
LATENCY_TOLERANCE = 1.05
TARGET_RATIO = 0.25


def done_keys(records):
    """The (task, policy, ratio) combinations a record file already covers."""
    return {(r["task"], r["policy"], r["ratio"]) for r in records}


def _pending(task, policy, ratios, done):
    """Ratios still to run for this task and policy, after the full cache shortcut
    and after anything a resumed run already has."""
    wanted = _ratios_to_run(policy, ratios)
    if not done:
        return wanted
    # The full baseline is stored at every ratio, so if any is present it is done.
    if policy == "full" and any((task, policy, r) in done for r in ratios):
        return []
    return [r for r in wanted if (task, policy, r) not in done]


def _ratios_to_run(policy, ratios):
    """The full cache baseline ignores the budget, so it is measured once and the
    record copied to the other ratios. On gov_report that is two thirds of the
    baseline's GPU time saved for an identical number."""
    return ratios[:1] if policy == "full" else ratios


def _fan_out(records, ratios):
    """Copy a full cache record across the remaining ratios."""
    out = []
    for r in records:
        for ratio in ratios:
            if ratio == r["ratio"]:
                out.append(r)
            else:
                out.append({**r, "ratio": ratio, "copied_from_ratio": r["ratio"]})
    return out


def run_needle(model, tokenizer, policies, ratios, n_filler, max_new_tokens=16,
               log=print, done=None, on_progress=None):
    """Needle retrieval at each keep ratio. One record per policy, ratio, needle, depth."""
    records = []
    for policy in policies:
        pending = _pending("needle", policy, ratios, done)
        if not pending:
            log(f"  needle {policy:>14}:  already done, skipping")
            continue
        line = [f"  needle {policy:>14}:"]
        mine = []
        for ratio in pending:
            hits, runs = 0, 0
            for needle in NEEDLES:
                for depth in DEPTHS:
                    prompt = make_needle_prompt(needle, depth, n_filler=n_filler)
                    metrics = {}
                    answer = run_policy(model, tokenizer, prompt, policy, ratio=ratio,
                                        max_new_tokens=max_new_tokens, metrics=metrics)
                    hit = found(answer, needle)
                    hits += hit
                    runs += 1
                    mine.append({"task": "needle", "policy": policy, "ratio": ratio,
                                 "needle": needle, "depth": depth, "score": float(hit),
                                 **metrics})
            line.append(f"r{ratio}={hits / max(1, runs):.0%}")
        records += _fan_out(mine, ratios) if policy == "full" else mine
        log("  ".join(line))
        if on_progress:
            on_progress(records)
    return records


def run_longbench(model, tokenizer, policies, ratios, tasks, n_samples,
                  max_context_tokens, log=print, done=None, on_progress=None):
    """The LongBench subsets. Samples are drawn once per task so every policy sees
    the same documents."""
    records = []
    for task in tasks:
        if all(not _pending(task, p, ratios, done) for p in policies):
            log(f"  {task}: already done, skipping")
            continue
        samples = longbench.load_samples(task, n_samples)
        spec = longbench.TASKS[task]
        prompts = [longbench.make_prompt(tokenizer, task, s, max_context_tokens)
                   for s in samples]
        for policy in policies:
            pending = _pending(task, policy, ratios, done)
            if not pending:
                log(f"  {task} {policy:>14}:  already done, skipping")
                continue
            line = [f"  {task} {policy:>14}:"]
            mine = []
            for ratio in pending:
                scores = []
                for prompt, sample in zip(prompts, samples):
                    metrics = {}
                    answer = run_policy(model, tokenizer, prompt, policy, ratio=ratio,
                                        max_new_tokens=spec["max_new_tokens"],
                                        metrics=metrics)
                    s = longbench.score(task, answer, sample["answers"])
                    scores.append(s)
                    mine.append({"task": task, "policy": policy, "ratio": ratio,
                                 "metric": spec["metric"], "score": s, **metrics})
                mean = sum(scores) / max(1, len(scores))
                line.append(f"r{ratio}={mean:.3f}")
            records += _fan_out(mine, ratios) if policy == "full" else mine
            log("  ".join(line))
            if on_progress:
                on_progress(records)
    return records


def run_pg19(model, tokenizer, policies, ratios, n_passages, n_tokens, n_prefill,
             log=print, done=None, on_progress=None):
    """Perplexity on PG19 passages, teacher forced with the policy evicting each step."""
    if all(not _pending("pg19", p, ratios, done) for p in policies):
        log("  pg19: already done, skipping")
        return []
    passages = load_passages(tokenizer, n_passages, n_tokens)
    records = []
    for policy_name in policies:
        pending = _pending("pg19", policy_name, ratios, done)
        if not pending:
            log(f"  pg19 {policy_name:>14}:  already done, skipping")
            continue
        policy = None if policy_name == "full" else make_policy(policy_name)
        line = [f"  pg19 {policy_name:>14}:"]
        mine = []
        for ratio in pending:
            budget = budget_for(n_prefill, ratio)
            nll, count = 0.0, 0
            for ids in passages:
                metrics = {}
                p_nll, p_count = passage_nll(model, ids, policy, budget, n_prefill,
                                             metrics=metrics,
                                             compress_once=(policy_name == "snapkv"))
                nll += p_nll
                count += p_count
                mine.append({"task": "pg19", "policy": policy_name, "ratio": ratio,
                             "metric": "nll_sum", "nll": p_nll, "tokens": p_count,
                             "budget": budget, **metrics})
            line.append(f"r{ratio}={perplexity(nll, count):.2f}")
        records += _fan_out(mine, ratios) if policy_name == "full" else mine
        log("  ".join(line))
        if on_progress:
            on_progress(records)
    return records


# --- reading the records back -------------------------------------------------

def mean_score(records, task, policy, ratio):
    rs = [r for r in records if r["task"] == task and r["policy"] == policy
          and r["ratio"] == ratio]
    if not rs:
        return None
    if task == "pg19":
        nll = sum(r["nll"] for r in rs)
        tokens = sum(r["tokens"] for r in rs)
        return perplexity(nll, tokens)
    return sum(r["score"] for r in rs) / len(rs)


def mean_latency(records, task, policy, ratio):
    rs = [r for r in records if r["task"] == task and r["policy"] == policy
          and r["ratio"] == ratio and "decode_ms_per_token" in r]
    return sum(r["decode_ms_per_token"] for r in rs) / len(rs) if rs else None


def mean_kv_mb(records, task, policy, ratio, key="decode_peak_kv_bytes"):
    """Mean cache size in MB. Defaults to the decode peak, which is the number a
    policy moves; pass key="peak_kv_bytes" for the whole run including prefill."""
    rs = [r for r in records if r["task"] == task and r["policy"] == policy
          and r["ratio"] == ratio and key in r]
    return sum(r[key] for r in rs) / len(rs) / 1e6 if rs else None


def verdict(records, ratio=TARGET_RATIO, tasks=None, ours=OURS, baselines=None):
    """Score the preregistered criterion. Returns a dict, pass or fail, with the
    per task reasoning so the README can show the work."""
    tasks = tasks or QUALITY_TASKS
    baselines = baselines or BASELINES
    rows, wins = [], 0

    for task in tasks:
        ours_score = mean_score(records, task, ours, ratio)
        cand = [(b, mean_score(records, task, b, ratio)) for b in baselines]
        cand = [(b, s) for b, s in cand if s is not None]
        if ours_score is None or not cand:
            rows.append({"task": task, "status": "missing"})
            continue

        best_baseline, best_score = max(cand, key=lambda bs: bs[1])
        ours_latency = mean_latency(records, task, ours, ratio)
        base_latency = mean_latency(records, task, best_baseline, ratio)
        quality_ok = ours_score >= best_score
        latency_ok = (ours_latency is None or base_latency is None
                      or ours_latency <= base_latency * LATENCY_TOLERANCE)
        won = bool(quality_ok and latency_ok)
        wins += won
        rows.append({
            "task": task, "status": "won" if won else "lost",
            "ours": ours_score, "best_baseline": best_baseline,
            "best_baseline_score": best_score,
            "ours_ms_per_token": ours_latency, "baseline_ms_per_token": base_latency,
            "quality_ok": quality_ok, "latency_ok": latency_ok,
        })

    complete = all(r["status"] != "missing" for r in rows)
    return {
        "ratio": ratio, "tasks": tasks, "wins": wins, "needed": 2,
        "complete": complete,
        "passed": bool(complete and wins >= 2),
        "rows": rows,
    }


def save(records, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
