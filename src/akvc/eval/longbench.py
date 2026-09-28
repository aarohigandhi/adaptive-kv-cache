"""The four LongBench subsets the spec asks for.

qasper and hotpotqa are question answering over long documents, scored with token
F1. gov_report and samsum are summarization, scored with ROUGE-L. The prompt
templates and the scoring choice are the ones LongBench ships, so the numbers sit
next to published ones instead of being their own private scale.

Contexts run past what a 1.5B model can hold, so long ones are truncated from the
middle. That is what the LongBench harness does too: the instruction sits at the
start and the question at the end, and cutting the middle leaves both intact.
"""

from .metrics import token_f1, rouge_l, best_over_references

DATASET = "THUDM/LongBench"
DATA_ARCHIVE = "data.zip"

TASKS = {
    "qasper": {
        "metric": "f1",
        "max_new_tokens": 32,
        "template": (
            "You are given a scientific article and a question. Answer the question "
            "as concisely as you can, using a single phrase or sentence if possible. "
            "If the question cannot be answered based on the information in the "
            "article, write \"unanswerable\".\n\n"
            "Article: {context}\n\n"
            "Question: {input}\n\nAnswer:"
        ),
    },
    "hotpotqa": {
        "metric": "f1",
        "max_new_tokens": 32,
        "template": (
            "Answer the question based on the given passages. Only give me the answer "
            "and do not output any other words.\n\n"
            "The following are given passages.\n{context}\n\n"
            "Answer the question based on the given passages. Only give me the answer "
            "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
        ),
    },
    "gov_report": {
        "metric": "rouge_l",
        "max_new_tokens": 128,
        "template": (
            "You are given a report by a government agency. Write a one page summary "
            "of the report.\n\nReport:\n{context}\n\nNow, write a one page summary of "
            "the report.\n\nSummary:"
        ),
    },
    "samsum": {
        "metric": "rouge_l",
        "max_new_tokens": 64,
        "template": (
            "Summarize the dialogue into a few short sentences. The following are some "
            "examples.\n\n{context}\n\n{input}\n"
        ),
    },
}

SCORERS = {"f1": token_f1, "rouge_l": rouge_l}


def load_samples(task, n_samples, seed=0):
    """Take n_samples from a LongBench subset, deterministically.

    Read out of the official data.zip on the Hub rather than through
    load_dataset. LongBench ships as a loading script, and datasets 5 dropped
    script datasets entirely, so load_dataset("THUDM/LongBench", "qasper") now
    raises. The zip is about 110 MB, downloaded once and cached by huggingface_hub.

    Sampling is a seeded shuffle of the row order, so every policy sees the same
    documents and a rerun picks the same ones.
    """
    import io
    import json
    import random
    import zipfile

    from huggingface_hub import hf_hub_download

    if task not in TASKS:
        raise ValueError(f"Unknown LongBench task: {task}. Choose from {sorted(TASKS)}.")

    archive = hf_hub_download(DATASET, DATA_ARCHIVE, repo_type="dataset")
    with zipfile.ZipFile(archive) as z:
        name = next((n for n in z.namelist()
                     if n.endswith(f"{task}.jsonl") and "_e" not in n.rsplit("/", 1)[-1]), None)
        if name is None:
            raise RuntimeError(f"{task}.jsonl is not in {DATA_ARCHIVE}: {z.namelist()[:10]}")
        with z.open(name) as f:
            rows = [json.loads(line) for line in io.TextIOWrapper(f, encoding="utf-8")
                    if line.strip()]

    random.Random(seed).shuffle(rows)
    if n_samples:
        rows = rows[:n_samples]
    return [{"context": r["context"], "input": r["input"], "answers": r["answers"]}
            for r in rows]


def truncate_middle(tokenizer, text, max_tokens):
    """Keep the head and the tail of a long context and drop the middle."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    half = max_tokens // 2
    return tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(
        ids[-(max_tokens - half):], skip_special_tokens=True
    )


def make_prompt(tokenizer, task, sample, max_context_tokens):
    """Fill the task template, truncating the context to fit."""
    spec = TASKS[task]
    context = truncate_middle(tokenizer, sample["context"], max_context_tokens)
    return spec["template"].format(context=context, input=sample["input"])


def score(task, prediction, answers):
    """Score one prediction, taking the best over the reference answers."""
    return best_over_references(SCORERS[TASKS[task]["metric"]], prediction, answers)
