"""Model loading and generation utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache

if not hasattr(DynamicCache, "seen_tokens"):

    @property
    def seen_tokens(self):
        return 0

    DynamicCache.seen_tokens = seen_tokens
from typing import Optional

# Models permitted to execute their own code from the Hub. Empty by default,
# and never appended to automatically: the previous loader escalated to
# trust_remote_code=True on any load failure, which silently ran unreviewed
# code. Add an entry only after reading that code.
_MODELS_REQUIRING_TRUST_REMOTE_CODE: set = set()


def _load_tokenizer(model_id: str, trust_remote_code: bool):
    """
    Load a tokenizer, retrying with trust_remote_code only if asked to.

    Args:
        model_id: HuggingFace model identifier.
        trust_remote_code: Whether executing repo code is permitted.

    Returns:
        The loaded tokenizer.
    """
    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception:
        if trust_remote_code:
            raise
        print(
            f"Error: failed to load tokenizer for {model_id}. If this model "
            f"genuinely requires custom code, add it to "
            f"_MODELS_REQUIRING_TRUST_REMOTE_CODE after reviewing that code."
        )
        raise


def _load_model(model_id: str, trust_remote_code: bool, **kwargs):
    """
    Load a model, falling back from safetensors to legacy weights.

    A single fallback ladder shared by every device. The CUDA and CPU/MPS
    branches previously carried copy-pasted four-level nested versions of
    this, including two identical copies of the tiny-gpt2 error message.

    Note that failures are NOT retried with trust_remote_code=True. The old
    code did that automatically, which silently executed arbitrary code from
    the Hub whenever a load failed for any reason.

    Args:
        model_id: HuggingFace model identifier.
        trust_remote_code: Whether executing repo code is permitted.
        **kwargs: Passed through to from_pretrained (dtype, device_map, ...).

    Returns:
        The loaded model.
    """
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_safetensors=True,
            **kwargs,
        )
    except Exception as safetensors_error:
        print(
            f"Warning: failed to load {model_id} with safetensors=True "
            f"({safetensors_error}); retrying with legacy weights"
        )
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                use_safetensors=False,
                **kwargs,
            )
        except Exception as legacy_error:
            raise RuntimeError(
                f"Failed to load model {model_id}.\n"
                f"  with safetensors: {safetensors_error}\n"
                f"  with legacy weights: {legacy_error}\n"
                f"Torch {torch.__version__} refuses pickled weights "
                f"(CVE-2025-32434); a checkpoint without safetensors cannot be "
                f"loaded. Pick a model that publishes safetensors."
            ) from legacy_error


def load_model_and_tokenizer(
    model_id: str,
    device: Optional[torch.device] = None,
    device_override: Optional[str] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a HuggingFace model and tokenizer for inference.

    Args:
        model_id: HuggingFace model identifier (e.g., "microsoft/Phi-3-mini-4k-instruct").
        device: Target device (deprecated, use device_override instead).
        device_override: Device string ("cuda", "mps", "cpu", or None to
            auto-detect).

    Returns:
        Tuple of (model, tokenizer).
    """
    from edge_slm_ace.utils.device_utils import resolve_device_override

    device, forced = resolve_device_override(
        device_override if device_override is not None else None,
        model_id=model_id,
    )

    if forced == "forced" and device.type == "cpu":
        print(
            "[model_manager] tiny-gpt2 detected -> forcing CPU (CUDA load blocked by torch >=2.6)"
        )

    # trust_remote_code is opt-in via the allowlist and is never escalated on
    # failure, since that would execute unreviewed code from the Hub.
    trust_remote_code = model_id in _MODELS_REQUIRING_TRUST_REMOTE_CODE

    tokenizer = _load_tokenizer(model_id, trust_remote_code)

    # Set pad_token if not present (some models don't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # fp16 on CUDA; fp32 elsewhere. Recorded in run metadata, since numerics
    # differ between the two and that shows up in results.
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if device.type == "cuda":
        model = _load_model(model_id, trust_remote_code, dtype=dtype, device_map="auto")
    else:
        model = _load_model(model_id, trust_remote_code, dtype=dtype)
        model = model.to(device)

        # Some models do not run on MPS; check with a dummy forward pass
        # rather than failing mid-evaluation.
        if device.type == "mps":
            try:
                probe = tokenizer("test", return_tensors="pt")
                probe = {k: v.to(device) for k, v in probe.items()}
                with torch.no_grad():
                    _ = model(**probe)
            except Exception:
                print(f"Warning: MPS device not compatible with {model_id}, falling back to CPU")
                device = torch.device("cpu")
                model = model.to(device)

    model.eval()  # Set to evaluation mode

    print(f"[model_manager] Loaded {model_id} on device={device} (dtype={dtype})")

    return model, tokenizer


# Fallback context window when a checkpoint does not advertise one. Some
# tokenizers report model_max_length as a sentinel like 1e30, which is why we
# never trust that value on its own.
_FALLBACK_CONTEXT_TOKENS = 4096
_MIN_PROMPT_TOKENS = 128


def render_prompt(
    tokenizer: AutoTokenizer,
    prompt: str,
    use_chat_template: bool = True,
) -> tuple[str, bool]:
    """
    Render a prompt into the form the checkpoint was trained to receive.

    Every model in MODEL_CONFIGS is instruction-tuned (Phi-3-*-instruct,
    Mistral-*-Instruct, Qwen2.5-*-Instruct, TinyLlama-*-Chat). Those models
    are trained to respond inside their own turn markers; fed a raw
    completion they continue the text instead of answering it, which shows up
    as rambling, question echoing and self-dialogue. Prompting them raw
    benchmarks a misuse of the model rather than the model.

    Args:
        tokenizer: Tokenizer, consulted for its chat template.
        prompt: The user-turn content.
        use_chat_template: Set False to force raw completion formatting.

    Returns:
        Tuple of (rendered_text, template_was_applied).
    """
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return text, True
        except Exception as e:  # malformed template, unsupported role, ...
            print(f"Warning: chat template failed ({e}); falling back to raw prompt")
    return prompt, False


def max_prompt_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> int:
    """
    Largest prompt length that still leaves room to generate.

    Args:
        model: Loaded model, consulted for its positional limit.
        tokenizer: Tokenizer, used only as a fallback source.
        max_new_tokens: Tokens reserved for the completion.

    Returns:
        Maximum prompt length in tokens.
    """
    limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)

    if not limit:
        candidate = getattr(tokenizer, "model_max_length", None)
        # Reject the "unbounded" sentinel some tokenizers ship.
        if candidate and candidate < 1_000_000:
            limit = candidate

    if not limit:
        limit = _FALLBACK_CONTEXT_TOKENS

    return max(_MIN_PROMPT_TOKENS, int(limit) - int(max_new_tokens))


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    use_chat_template: bool = True,
    return_meta: bool = False,
):
    """
    Generate text from a prompt using the model.

    Args:
        model: The loaded language model.
        tokenizer: The tokenizer for the model.
        prompt: Input text prompt (user-turn content).
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. 0.0 selects greedy decoding.
        top_p: Nucleus sampling parameter (ignored when greedy).
        use_chat_template: Apply the checkpoint's chat template if it has one.
        return_meta: If True, return (text, meta) where meta records whether
            the chat template was applied and whether the prompt was
            truncated.

    Returns:
        Generated text, or (text, meta) when return_meta is True.
    """
    text, used_chat_template = render_prompt(tokenizer, prompt, use_chat_template)

    # Bound truncation explicitly. `truncation=True` with no max_length falls
    # back to tokenizer.model_max_length, which is 512 on some checkpoints and
    # a 1e30 sentinel on others. Where it is small the tail of the prompt is
    # silently dropped -- and the tail is the question. ACE prompts are much
    # longer than baseline prompts, so that would hit one arm preferentially
    # and look exactly like "the playbook confuses small models".
    limit = max_prompt_tokens(model, tokenizer, max_new_tokens)

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=limit,
            # The chat template already inserted BOS/turn markers; adding
            # special tokens again would duplicate them.
            add_special_tokens=not used_chat_template,
        )
    except Exception as e:
        raise ValueError(f"Failed to tokenize prompt: {e}") from e

    prompt_len = int(inputs["input_ids"].shape[1])
    truncated = prompt_len >= limit
    if truncated:
        print(
            f"Warning: prompt truncated to {limit} tokens "
            f"(context={limit + max_new_tokens}); this result is suspect"
        )

    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Build generation kwargs. Passing temperature/top_p alongside
    # do_sample=False makes transformers warn and obscures which decoding
    # actually ran, so branch explicitly instead.
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature and temperature > 0.0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        gen_kwargs["do_sample"] = False

    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}") from e

    # Decode output (skip the input tokens)
    try:
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    except Exception as e:
        raise ValueError(f"Failed to decode output: {e}") from e

    generated_text = generated_text.strip()

    if return_meta:
        return generated_text, {
            "used_chat_template": used_chat_template,
            "prompt_truncated": truncated,
            "prompt_tokens": prompt_len,
            "prompt_token_limit": limit,
            "greedy": not (temperature and temperature > 0.0),
        }
    return generated_text


def count_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    """
    Count the number of tokens in a text string.

    Args:
        tokenizer: The tokenizer to use.
        text: Text to count tokens for.

    Returns:
        Number of tokens.
    """
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception:
        # Fallback: approximate via word count
        return int(len(text.split()) * 1.3)
