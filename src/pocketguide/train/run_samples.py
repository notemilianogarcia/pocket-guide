"""
Lesson 5.4: Run qualitative evaluation + sample generations.

Generates outputs from base and fine-tuned LoRA models on a fixed prompt suite,
validates with existing parsing/schema code, and writes comparison metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from pocketguide.eval.metrics_v0 import check_required_fields, detect_uncertainty_markers
from pocketguide.eval.parsing import parse_and_validate

# Envelope required fields for required_field_presence_rate
ENVELOPE_REQUIRED_FIELDS = [
    "summary",
    "assumptions",
    "uncertainty_notes",
    "next_steps",
    "verification_steps",
    "payload_type",
    "payload",
]


def _load_config(run_dir: Path) -> dict[str, Any]:
    """Load training config snapshot from run_dir/config.yaml."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config snapshot not found: {config_path}. Run training first."
        )
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_prompts(prompts_path: Path) -> list[dict[str, Any]]:
    """Load fixed prompts JSONL; require id and prompt per line."""
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    records = []
    with open(prompts_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {prompts_path} at line {line_num}: {e}"
                ) from e
            if "id" not in item or "prompt" not in item:
                raise ValueError(
                    f"Missing 'id' or 'prompt' in {prompts_path} at line {line_num}"
                )
            records.append(item)
    return records


def _ensure_adapter(run_dir: Path) -> None:
    """Fail loudly if adapter directory is missing."""
    adapter_dir = run_dir / "adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}. Run training (Lesson 5.3) first."
        )
    # Require adapter_config.json (PEFT standard) or config.json (legacy)
    if not (adapter_dir / "adapter_config.json").exists() and not (
        adapter_dir / "config.json"
    ).exists():
        raise FileNotFoundError(
            f"Adapter config not found in {adapter_dir}. Incomplete or invalid adapter."
        )


def _build_sample_record(
    prompt_row: dict[str, Any],
    raw_text_output: str,
    parse_result: Any,
    latency_ms: float,
    tokens_generated: int | None,
) -> dict[str, Any]:
    """Build one sample output record for JSONL."""
    parse_success = parse_result.success
    schema_valid = parse_success
    parsed_json = parse_result.data if parse_success else None
    error_code = None
    missing_fields: list[str] = []

    if not parse_success and parse_result.error:
        error_code = getattr(parse_result.error, "code", None) or str(
            parse_result.error
        )
    elif parse_success and parsed_json:
        ok, missing = check_required_fields(parsed_json, ENVELOPE_REQUIRED_FIELDS)
        if not ok:
            missing_fields = missing

    uncertainty = detect_uncertainty_markers(raw_text_output)
    uncertainty_present = (
        uncertainty.get("has_assumptions_marker", False)
        or uncertainty.get("has_verification_marker", False)
    )

    record: dict[str, Any] = {
        "prompt_id": prompt_row.get("id"),
        "prompt": prompt_row.get("prompt", ""),
        "payload_type": prompt_row.get("payload_type"),
        "raw_text_output": raw_text_output,
        "parsed_json": parsed_json,
        "parse_success": parse_success,
        "schema_valid": schema_valid,
        "missing_fields": missing_fields,
        "error_code": error_code,
        "latency_ms": round(latency_ms, 2),
        "tokens_generated": tokens_generated,
        "uncertainty_marker_present": uncertainty_present,
    }
    return record


def _run_inference_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    gen_spec: Any,
    seed: int,
) -> dict[str, Any]:
    """Run inference for one prompt; delegate to inference.base_model."""
    from pocketguide.inference.base_model import generate_one

    result = generate_one(model, tokenizer, prompt, gen_spec, seed=seed)
    return result


def _compute_aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute lightweight aggregate metrics for a set of sample records."""
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "parse_success_rate": 0.0,
            "schema_valid_rate": 0.0,
            "required_field_presence_rate": 0.0,
            "uncertainty_marker_presence_rate": 0.0,
            "avg_latency_ms": None,
        }

    parse_ok = sum(1 for r in records if r.get("parse_success", False))
    schema_ok = sum(1 for r in records if r.get("schema_valid", False))
    required_ok = sum(
        1 for r in records if r.get("parse_success") and not r.get("missing_fields")
    )
    required_denom = sum(1 for r in records if r.get("parse_success", False))
    required_field_presence_rate = (
        required_ok / required_denom if required_denom else 0.0
    )
    uncertainty_ok = sum(
        1 for r in records if r.get("uncertainty_marker_present", False)
    )
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]

    return {
        "n": n,
        "parse_success_rate": parse_ok / n,
        "schema_valid_rate": schema_ok / n,
        "required_field_presence_rate": required_field_presence_rate,
        "uncertainty_marker_presence_rate": uncertainty_ok / n,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }


def _compute_delta(base_metrics: dict, finetuned_metrics: dict) -> dict[str, Any]:
    """Compute delta (finetuned - base) for numeric metrics."""
    delta = {}
    for key in (
        "parse_success_rate",
        "schema_valid_rate",
        "required_field_presence_rate",
        "uncertainty_marker_presence_rate",
    ):
        b = base_metrics.get(key)
        f = finetuned_metrics.get(key)
        if b is not None and f is not None:
            delta[key] = round(f - b, 4)
    if base_metrics.get("avg_latency_ms") is not None and finetuned_metrics.get(
        "avg_latency_ms"
    ) is not None:
        delta["avg_latency_ms"] = round(
            finetuned_metrics["avg_latency_ms"] - base_metrics["avg_latency_ms"], 2
        )
    return delta


def run_samples(
    run_dir: Path,
    prompts_path: Path,
    project_root: Path,
    seed: int = 42,
) -> None:
    """Load config and prompts, run base and finetuned inference, write outputs and metrics."""
    cfg = _load_config(run_dir)
    prompts = _load_prompts(prompts_path)
    _ensure_adapter(run_dir)

    # Resolve device/precision (same as train)
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for run_samples") from None

    runtime = cfg.get("runtime", {})
    device_cfg = runtime.get("device", "auto")
    precision_cfg = runtime.get("precision", "auto")

    if device_cfg and str(device_cfg).lower() != "auto":
        device = str(device_cfg).strip().lower()
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if precision_cfg and str(precision_cfg).lower() != "auto":
        precision = str(precision_cfg).strip().lower()
    else:
        if device == "cuda" and torch.cuda.is_bf16_supported():
            precision = "bf16"
        elif device == "cuda":
            precision = "fp16"
        elif device == "mps":
            # fp16 keeps 7B ~14GB so it can fit in 24GB unified memory (fp32 would OOM)
            precision = "fp16"
        else:
            precision = "fp32"

    # Generation params (fixed for both runs)
    data_cfg = cfg.get("data", {})
    max_new_tokens = int(data_cfg.get("max_seq_len", 1024))
    train_cfg = cfg.get("training", {})
    gen_seed = int(train_cfg.get("seed", 42)) if seed is None else seed

    from pocketguide.inference.base_model import (
        GenSpec,
        ModelSpec,
        RuntimeSpec,
        load_model_and_tokenizer,
    )

    dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    dtype = dtype_map.get(precision, "float32")
    gen_spec = GenSpec(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    model_spec = ModelSpec(id=cfg.get("base_model_id", ""))
    runtime_spec = RuntimeSpec(device=device, dtype=dtype)

    # Load base model (no adapter)
    base_model, tokenizer = load_model_and_tokenizer(model_spec, runtime_spec)

    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    base_records = []
    for row in prompts:
        prompt_text = row["prompt"]
        result = _run_inference_one(
            base_model, tokenizer, prompt_text, gen_spec, gen_seed
        )
        raw_output = result.get("completion_text", result.get("text", ""))
        parse_result = parse_and_validate(raw_output, strict_json=False)
        latency_s = result.get("timing", {}).get("latency_s", 0)
        usage = result.get("usage", {})
        tokens_gen = usage.get("completion_tokens")

        rec = _build_sample_record(
            row,
            raw_output,
            parse_result,
            latency_ms=latency_s * 1000,
            tokens_generated=tokens_gen,
        )
        base_records.append(rec)

    base_out = samples_dir / "base_outputs.jsonl"
    with open(base_out, "w", encoding="utf-8") as f:
        for rec in base_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Free base model before loading finetuned (save memory)
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Load base + LoRA adapter
    from peft import PeftModel

    base_model, tokenizer = load_model_and_tokenizer(model_spec, runtime_spec)
    adapter_path = run_dir / "adapter"
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    finetuned_records = []
    for row in prompts:
        prompt_text = row["prompt"]
        result = _run_inference_one(model, tokenizer, prompt_text, gen_spec, gen_seed)
        raw_output = result.get("completion_text", result.get("text", ""))
        parse_result = parse_and_validate(raw_output, strict_json=False)
        latency_s = result.get("timing", {}).get("latency_s", 0)
        usage = result.get("usage", {})
        tokens_gen = usage.get("completion_tokens")

        rec = _build_sample_record(
            row,
            raw_output,
            parse_result,
            latency_ms=latency_s * 1000,
            tokens_generated=tokens_gen,
        )
        finetuned_records.append(rec)

    finetuned_out = samples_dir / "finetuned_outputs.jsonl"
    with open(finetuned_out, "w", encoding="utf-8") as f:
        for rec in finetuned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    base_metrics = _compute_aggregate_metrics(base_records)
    finetuned_metrics = _compute_aggregate_metrics(finetuned_records)
    delta = _compute_delta(base_metrics, finetuned_metrics)

    comparison = {
        "num_prompts": len(prompts),
        "base": base_metrics,
        "finetuned": finetuned_metrics,
        "delta": delta,
    }
    metrics_path = samples_dir / "comparison_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"Wrote {base_out} ({len(base_records)} records)")
    print(f"Wrote {finetuned_out} ({len(finetuned_records)} records)")
    print(f"Wrote {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run base and fine-tuned sample generation (Lesson 5.4)."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Training run directory (e.g. runs/train/<run_id>)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to fixed prompts JSONL (e.g. eval/suites/fixed20_v1.jsonl)",
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=None,
        help="Project root (default: cwd)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation (default: 42)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    prompts_path = Path(args.prompts).resolve()
    project_root = Path(args.project_root or ".").resolve()

    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        run_samples(run_dir, prompts_path, project_root, seed=args.seed)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
