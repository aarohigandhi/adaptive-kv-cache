"""Scoring functions for the LongBench subsets.

LongBench scores qasper and hotpotqa with token level F1 against the reference
answers, and gov_report and samsum with ROUGE-L. Both are short enough to write
here, which keeps the dependency list to datasets and nothing else. Normalization
follows the SQuAD convention the LongBench harness uses: lowercase, strip
articles and punctuation, collapse whitespace.
"""

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text):
    text = text.lower()
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def token_f1(prediction, reference):
    """Harmonic mean of token precision and recall after normalization."""
    pred = normalize(prediction).split()
    ref = normalize(reference).split()
    if not pred or not ref:
        return float(pred == ref)

    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _lcs(a, b):
    """Length of the longest common subsequence, the row at a time version."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction, reference):
    """F measure over the longest common subsequence of tokens."""
    pred = normalize(prediction).split()
    ref = normalize(reference).split()
    if not pred or not ref:
        return float(pred == ref)

    match = _lcs(pred, ref)
    if match == 0:
        return 0.0
    precision = match / len(pred)
    recall = match / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_over_references(scorer, prediction, references):
    """LongBench takes the best score over the reference answers for a sample."""
    if isinstance(references, str):
        references = [references]
    return max((scorer(prediction, r) for r in references), default=0.0)
