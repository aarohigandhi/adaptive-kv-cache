"""The needle in a haystack test.

Hide a code somewhere in a long filler prompt, then ask for it. A policy that
evicts the region holding the code cannot answer, which is where the recency and
heavy hitter policies tend to fail.
"""

FILLER = ("The sea was calm and the sky was clear over the northern coast. "
          "The keeper walked the shore and noted the quiet tide. ")

QUESTION = " Question: what is the vault passcode? Answer with just the number."

NEEDLES = ["7391", "2648", "5127"]
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]


def make_prompt(needle, depth, n_filler=16):
    """Build a prompt with the needle placed at roughly depth through the filler."""
    before = int(round(depth * n_filler))
    note = f"Important administrative note: the vault passcode is {needle}. "
    return FILLER * before + note + FILLER * (n_filler - before) + QUESTION


def found(answer, needle):
    return needle in answer
