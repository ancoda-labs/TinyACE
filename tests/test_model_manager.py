"""Tests for model manager functionality."""

from edge_slm_ace.models.model_manager import (
    _FALLBACK_CONTEXT_TOKENS,
    generate,
    max_prompt_tokens,
    render_prompt,
)


def test_load_model_and_tokenizer(tiny_model):
    """Test loading a tiny model."""
    model, tokenizer = tiny_model

    assert model is not None
    assert tokenizer is not None
    assert tokenizer.pad_token is not None  # Should be set

    # tiny-gpt2 is always forced to CPU on Torch >= 2.6 due to security restrictions
    assert next(model.parameters()).device.type == "cpu"


def test_generate(tiny_model):
    """Generation returns decoded text.

    It does not assert the text is non-empty: tiny-gpt2 has random weights, so
    a whitespace-only sample is a legitimate outcome and stripping it leaves
    "". That assertion made the test fail on the model's own noise.
    """
    model, tokenizer = tiny_model

    output = generate(model, tokenizer, "Hello", max_new_tokens=10, temperature=0.7, top_p=0.95)

    assert isinstance(output, str)


def test_render_prompt_without_chat_template(tiny_tokenizer):
    """A base model with no template must fall through to raw completion."""
    tiny_tokenizer.chat_template = None

    text, used = render_prompt(tiny_tokenizer, "Hello")
    assert used is False
    assert text == "Hello"


def test_render_prompt_applies_chat_template(tiny_tokenizer):
    """An instruct-tuned checkpoint must be wrapped in its turn markers."""
    tiny_tokenizer.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )

    text, used = render_prompt(tiny_tokenizer, "Hello")
    assert used is True
    assert "<|user|>Hello" in text
    assert text.endswith("<|assistant|>")


def test_render_prompt_can_be_disabled(tiny_tokenizer):
    tiny_tokenizer.chat_template = "{{ 'SHOULD NOT APPEAR' }}"

    text, used = render_prompt(tiny_tokenizer, "Hello", use_chat_template=False)
    assert used is False
    assert text == "Hello"


def test_max_prompt_tokens_reserves_room_for_completion(tiny_model):
    """The prompt budget must leave space for max_new_tokens."""
    model, tokenizer = tiny_model
    context = model.config.max_position_embeddings

    assert max_prompt_tokens(model, tokenizer, 256) == context - 256
    assert max_prompt_tokens(model, tokenizer, 0) == context


def test_max_prompt_tokens_rejects_sentinel_model_max_length():
    """tokenizer.model_max_length is 1e30 on some checkpoints; never trust it."""

    class _NoPositions:
        config = type("C", (), {})()

    class _Sentinel:
        model_max_length = 1000000000000000019884624838656

    assert max_prompt_tokens(_NoPositions(), _Sentinel(), 256) == (_FALLBACK_CONTEXT_TOKENS - 256)


def test_generate_reports_metadata(tiny_model):
    """Truncation and template state must travel with the generation."""
    model, tokenizer = tiny_model

    text, meta = generate(
        model, tokenizer, "Hello", max_new_tokens=5, temperature=0.0, return_meta=True
    )

    assert isinstance(text, str)
    assert meta["prompt_truncated"] is False
    assert meta["greedy"] is True
    assert meta["prompt_tokens"] >= 1
