"""Base model inference using Hugging Face Transformers.

Provides deterministic generation with comprehensive metrics tracking.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelSpec:
    """Model specification for loading."""

    id: str
    revision: str | None = None


@dataclass
class RuntimeSpec:
    """Runtime configuration for model execution."""

    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class GenSpec:
    """Generation configuration."""

    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


def load_model_and_tokenizer(
    model_spec: ModelSpec,
    runtime_spec: RuntimeSpec,
) -> tuple[Any, Any]:
    """Load model and tokenizer from Hugging Face.

    Args:
        model_spec: Model identification (id, revision)
        runtime_spec: Runtime settings (device, dtype)

    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if offline mode is enabled
    offline = os.environ.get("POCKETGUIDE_OFFLINE", "0") == "1"
    local_files_only = offline

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec.id,
        revision=model_spec.revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(runtime_spec.dtype, torch.float32)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_spec.id,
        revision=model_spec.revision,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )

    # Move to device and set to eval mode
    model = model.to(runtime_spec.device)
    model.eval()

    return model, tokenizer


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    gen_spec: GenSpec,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate text from a single prompt with metrics.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Input prompt string
        gen_spec: Generation configuration
        seed: Random seed for reproducibility

    Returns:
        Dictionary with keys: text, usage, timing
    """
    # Set seeds for determinism
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    prompt_tokens = input_ids.shape[1]

    # Start timing
    start_time = time.perf_counter()

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=gen_spec.max_new_tokens,
            do_sample=gen_spec.do_sample,
            temperature=gen_spec.temperature if gen_spec.do_sample else 1.0,
            top_p=gen_spec.top_p,
            repetition_penalty=gen_spec.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )

    # End timing
    end_time = time.perf_counter()
    latency_s = end_time - start_time

    # Extract generated sequence
    generated_ids = outputs.sequences[0]
    total_tokens = generated_ids.shape[0]
    completion_tokens = total_tokens - prompt_tokens

    # Decode full sequence and completion-only (for validation of model output)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    completion_text = tokenizer.decode(
        generated_ids[prompt_tokens:], skip_special_tokens=True
    )

    # Calculate tokens per second
    tokens_per_s = completion_tokens / latency_s if latency_s > 0 else 0.0

    return {
        "text": generated_text,
        "completion_text": completion_text,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "timing": {
            "latency_s": latency_s,
            "tokens_per_s": tokens_per_s,
        },
    }
