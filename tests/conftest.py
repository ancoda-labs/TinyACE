"""Shared fixtures.

Deliberately almost empty. It used to hold `tiny_model` and `tiny_tokenizer`,
which imported `edge_slm_ace.models.model_manager` -- a module deleted with the
old pipeline. Nothing used the fixtures, so nothing failed; they would have
raised ImportError the moment a test asked for one.

The checkpoint they loaded is exercised where it belongs now: the CI smoke test
runs a real arm end to end on `sshleifer/tiny-gpt2`.
"""
