"""Shared fixtures.

The model-manager tests need a real checkpoint. They loaded
`sshleifer/tiny-gpt2` from the Hub seven separate times, with no skip if the
Hub was unreachable, so CI depended on network availability and paid for the
download repeatedly. It is fetched once per session here, and the tests skip
rather than fail when it cannot be reached.
"""

import pytest


@pytest.fixture(scope="session")
def tiny_model():
    """
    The tiny checkpoint used by the model-manager tests, loaded once.

    Returns:
        (model, tokenizer). Skips the test when the Hub is unreachable.
    """
    from edge_slm_ace.models.model_manager import load_model_and_tokenizer

    try:
        return load_model_and_tokenizer("sshleifer/tiny-gpt2")
    except Exception as e:  # offline, rate-limited, corrupted cache
        pytest.skip(f"tiny-gpt2 unavailable ({type(e).__name__}: {e})")


@pytest.fixture
def tiny_tokenizer(tiny_model):
    """
    The tokenizer, with its chat template restored afterwards.

    Several tests set `chat_template` to exercise the rendering branches; the
    session-scoped model is shared, so the mutation has to be undone.
    """
    _, tokenizer = tiny_model
    original = getattr(tokenizer, "chat_template", None)
    yield tokenizer
    tokenizer.chat_template = original
