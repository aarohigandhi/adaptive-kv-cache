"""Model + tokenizer loading, and our own token-by-token decode loop.

Why our own loop instead of model.generate(): the whole project is about
controlling the KV cache during decoding. generate() hides the cache from us,
so in Phase 0 we rewrite the loop ourselves and verify it matches generate()
token for token.

TODO (Phase 0): load_model(), build_inputs(), and manual_decode().
"""
