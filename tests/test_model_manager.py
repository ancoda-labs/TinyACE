"""Tests for model manager functionality."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from edge_slm_ace.models.model_manager import load_model_and_tokenizer, generate
from edge_slm_ace.utils.device_utils import get_device


def test_load_model_and_tokenizer():
    """Test loading a tiny model."""
    # Use tiny-gpt2 for fast testing
    model_id = "sshleifer/tiny-gpt2"
    
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(model_id, device=device)
    
    assert model is not None
    assert tokenizer is not None
    assert tokenizer.pad_token is not None  # Should be set
    
    # Check model is on correct device
    # Note: tiny-gpt2 is always forced to CPU on Torch >= 2.6 due to security restrictions
    model_device = next(model.parameters()).device
    expected_device = "cpu"  # tiny-gpt2 is forced to CPU
    assert model_device.type == expected_device


def test_generate():
    """Test text generation."""
    model_id = "sshleifer/tiny-gpt2"
    
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(model_id, device=device)
    
    prompt = "Hello"
    output = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=10,
        temperature=0.7,
        top_p=0.95,
    )
    
    assert isinstance(output, str)
    assert len(output) > 0  # Should produce some output



def test_render_prompt_without_chat_template():
    """A base model with no template must fall through to raw completion."""
    from edge_slm_ace.models.model_manager import render_prompt

    _, tokenizer = load_model_and_tokenizer("sshleifer/tiny-gpt2")
    tokenizer.chat_template = None

    text, used = render_prompt(tokenizer, "Hello")
    assert used is False
    assert text == "Hello"


def test_render_prompt_applies_chat_template():
    """An instruct-tuned checkpoint must be wrapped in its turn markers."""
    from edge_slm_ace.models.model_manager import render_prompt

    _, tokenizer = load_model_and_tokenizer("sshleifer/tiny-gpt2")
    tokenizer.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )

    text, used = render_prompt(tokenizer, "Hello")
    assert used is True
    assert "<|user|>Hello" in text
    assert text.endswith("<|assistant|>")


def test_render_prompt_can_be_disabled():
    from edge_slm_ace.models.model_manager import render_prompt

    _, tokenizer = load_model_and_tokenizer("sshleifer/tiny-gpt2")
    tokenizer.chat_template = "{{ 'SHOULD NOT APPEAR' }}"

    text, used = render_prompt(tokenizer, "Hello", use_chat_template=False)
    assert used is False
    assert text == "Hello"


def test_max_prompt_tokens_reserves_room_for_completion():
    """The prompt budget must leave space for max_new_tokens."""
    from edge_slm_ace.models.model_manager import max_prompt_tokens

    model, tokenizer = load_model_and_tokenizer("sshleifer/tiny-gpt2")
    context = model.config.max_position_embeddings

    assert max_prompt_tokens(model, tokenizer, 256) == context - 256
    assert max_prompt_tokens(model, tokenizer, 0) == context


def test_max_prompt_tokens_rejects_sentinel_model_max_length():
    """tokenizer.model_max_length is 1e30 on some checkpoints; never trust it."""
    from edge_slm_ace.models.model_manager import (
        _FALLBACK_CONTEXT_TOKENS,
        max_prompt_tokens,
    )

    class _NoPositions:
        config = type("C", (), {})()

    class _Sentinel:
        model_max_length = 1000000000000000019884624838656

    assert max_prompt_tokens(_NoPositions(), _Sentinel(), 256) == (
        _FALLBACK_CONTEXT_TOKENS - 256
    )


def test_generate_reports_metadata():
    """Truncation and template state must travel with the generation."""
    model, tokenizer = load_model_and_tokenizer("sshleifer/tiny-gpt2")

    text, meta = generate(
        model, tokenizer, "Hello", max_new_tokens=5, temperature=0.0, return_meta=True
    )

    assert isinstance(text, str)
    assert meta["prompt_truncated"] is False
    assert meta["greedy"] is True
    assert meta["prompt_tokens"] >= 1
