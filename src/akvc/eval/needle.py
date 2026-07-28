"""Needle-in-a-haystack test.

Hide a specific fact ("the needle") at a chosen depth inside a long filler
context, then ask for it. A policy that evicts the region holding the needle
will fail to answer -- which is exactly where heavy-hitter/recency policies
famously break, and where our skeleton-keeping method aims to do better.
"""

from typing import List

FILLER = ("The sea was calm and the sky was clear over the northern coast. "
          "The keeper walked the shore and noted the quiet tide. ")

QUESTION = " Question: what is the vault passcode? Answer with just the number."


def make_prompt(needle: str, depth: float, n_filler: int = 16) -> str:
    """Build a prompt with `needle` inserted at ~`depth` fraction of the context.

    depth 0.0 -> needle near the start, 1.0 -> needle near the end.
    """
    before = int(round(depth * n_filler))
    after = n_filler - before
    note = f"Important administrative note: the vault passcode is {needle}. "
    return FILLER * before + note + FILLER * after + QUESTION


def found(answer: str, needle: str) -> bool:
    """Did the model's answer contain the needle?"""
    return needle in answer


# A few distinct needles so each depth can be tested with more than one code.
NEEDLES: List[str] = ["7391", "2648", "5127"]
DEPTHS: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
