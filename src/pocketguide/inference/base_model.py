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
    eos_token_id: int | None = None
    stop_sequences: list[str] | None = None


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
    # Explicit attention_mask when pad_token_id == eos_token_id (avoids Transformers warning)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    else:
        attention_mask = attention_mask.to(model.device)
    prompt_tokens = input_ids.shape[1]

    # Start timing
    start_time = time.perf_counter()

    # Use eos_token_id from spec or fall back to tokenizer default
    eos_token_id = gen_spec.eos_token_id if gen_spec.eos_token_id is not None else tokenizer.eos_token_id
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_spec.max_new_tokens,
            do_sample=gen_spec.do_sample,
            temperature=gen_spec.temperature if gen_spec.do_sample else 1.0,
            top_p=gen_spec.top_p,
            repetition_penalty=gen_spec.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=True,
            return_legacy_cache=True,
        )
    
    # Post-process: truncate completion at stop sequences if found (handles multiple JSON generation)
    # Note: JSON extraction in run_samples.py will handle this more robustly, but this helps too
    if gen_spec.stop_sequences:
        generated_ids = outputs.sequences[0]
        completion_ids = generated_ids[prompt_tokens:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
        for stop_seq in gen_spec.stop_sequences:
            if stop_seq in completion_text:
                stop_idx = completion_text.find(stop_seq)
                # Truncate at stop sequence (keep the stop sequence)
                truncated_text = completion_text[:stop_idx + len(stop_seq)]
                # Re-encode and reconstruct
                truncated_ids = tokenizer.encode(truncated_text, add_special_tokens=False, return_tensors="pt")
                if truncated_ids.shape[1] > 0:
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    new_seq = torch.cat([input_ids[0], truncated_ids[0].to(model.device)], dim=0)
                    # Pad or truncate to match original length
                    if new_seq.shape[0] < generated_ids.shape[0]:
                        padding = torch.full((generated_ids.shape[0] - new_seq.shape[0],), pad_id, device=model.device, dtype=new_seq.dtype)
                        new_seq = torch.cat([new_seq, padding], dim=0)
                    elif new_seq.shape[0] > generated_ids.shape[0]:
                        new_seq = new_seq[:generated_ids.shape[0]]
                    outputs.sequences[0] = new_seq
                break

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


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    gen_spec: GenSpec,
    seed: int = 42,
    pad_token_id: int | None = None,
) -> list[dict[str, Any]]:
    """Generate text for multiple prompts in one batch (uses GPU better, faster eval).

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer (must have pad_token_id set)
        prompts: List of prompt strings
        gen_spec: Generation configuration
        seed: Random seed
        pad_token_id: Override tokenizer pad_token_id if needed

    Returns:
        List of result dicts (same shape as generate_one): completion_text, usage, timing.
        Latency is total wall time for the batch; tokens_per_s is aggregate.
    """
    if not prompts:
        return []

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pad_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # Tokenize with padding (right pad so attention_mask is correct for generation)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
        return_attention_mask=True,
        pad_to_multiple_of=8,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_lengths = attention_mask.sum(dim=1).tolist()

    # Use eos_token_id from spec or fall back to tokenizer default
    eos_token_id = gen_spec.eos_token_id if gen_spec.eos_token_id is not None else tokenizer.eos_token_id
    
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_spec.max_new_tokens,
            do_sample=gen_spec.do_sample,
            temperature=gen_spec.temperature if gen_spec.do_sample else 1.0,
            top_p=gen_spec.top_p,
            repetition_penalty=gen_spec.repetition_penalty,
            pad_token_id=pad_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=True,
            return_legacy_cache=True,
        )
    
    # Post-process batch: truncate completion at stop sequences if found
    # Note: JSON extraction in run_samples.py will handle this more robustly, but this helps too
    if gen_spec.stop_sequences:
        sequences = outputs.sequences.clone()
        for i in range(sequences.shape[0]):
            start_len = prompt_lengths[i]
            completion_ids = sequences[i, start_len:]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
            for stop_seq in gen_spec.stop_sequences:
                if stop_seq in completion_text:
                    stop_idx = completion_text.find(stop_seq)
                    # Truncate at stop sequence (keep the stop sequence)
                    truncated_text = completion_text[:stop_idx + len(stop_seq)]
                    truncated_ids = tokenizer.encode(truncated_text, add_special_tokens=False, return_tensors="pt")
                    if truncated_ids.shape[1] > 0:
                        new_completion = truncated_ids[0].to(model.device)
                        # Reconstruct full sequence
                        new_seq = torch.cat([sequences[i, :start_len], new_completion], dim=0)
                        # Pad or truncate to match original length
                        if new_seq.shape[0] < sequences.shape[1]:
                            padding = torch.full((sequences.shape[1] - new_seq.shape[0],), pad_id, device=model.device, dtype=new_seq.dtype)
                            new_seq = torch.cat([new_seq, padding], dim=0)
                        elif new_seq.shape[0] > sequences.shape[1]:
                            new_seq = new_seq[:sequences.shape[1]]
                        sequences[i] = new_seq
                    break
        outputs.sequences = sequences
    end_time = time.perf_counter()
    latency_s = end_time - start_time

    sequences = outputs.sequences
    batch_size = sequences.shape[0]
    total_completion_tokens = 0
    results = []
    for i in range(batch_size):
        start_len = prompt_lengths[i]
        completion_ids = sequences[i, start_len:]
        completion_tokens = completion_ids.shape[0]
        total_completion_tokens += completion_tokens
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        prompt_tokens = input_ids.shape[1]  # padded length; actual is prompt_lengths[i]
        results.append({
            "text": tokenizer.decode(sequences[i], skip_special_tokens=True),
            "completion_text": completion_text,
            "usage": {
                "prompt_tokens": prompt_lengths[i],
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_lengths[i] + completion_tokens,
            },
            "timing": {
                "latency_s": latency_s / batch_size,
                "tokens_per_s": total_completion_tokens / latency_s if latency_s > 0 else 0.0,
            },
        })
    return results
