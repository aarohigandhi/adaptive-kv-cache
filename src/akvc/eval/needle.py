"""Needle-in-a-haystack test.

Hide a specific fact ("the needle") inside a long filler context and check
whether the model can still retrieve it. This is where heavy-hitter policies
famously break, so it is a key stress test for our method.

TODO (Phase 4): implement needle placement + retrieval scoring sweep.
"""
