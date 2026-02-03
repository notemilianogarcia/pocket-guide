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
    # parse_success = we got valid JSON (strict or lenient); schema_valid = full envelope+payload pass
    parsed_json = getattr(parse_result, "data", None)
    parse_success = parsed_json is not None
    schema_valid = parse_result.success
    error_code = None
    error_message: str | None = None
    missing_fields: list[str] = []

    if not parse_result.success and parse_result.error:
        err = parse_result.error
        error_code = getattr(err, "code", None) or str(err)
        error_message = getattr(err, "message", None) or str(err)
    if parsed_json:
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
    if error_message is not None:
        record["schema_error_message"] = (error_message[:500] if len(error_message) > 500 else error_message)
    if parsed_json and isinstance(parsed_json, dict):
        record["parsed_top_level_keys"] = list(parsed_json.keys())
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


def _run_inference_batch(
    model: Any,
    tokenizer: Any,
    prompt_rows: list[dict[str, Any]],
    gen_spec: Any,
    seed: int,
) -> list[dict[str, Any]]:
    """Run inference for a batch of prompts; returns list of sample records."""
    from pocketguide.inference.base_model import generate_batch

    if not prompt_rows:
        return []
    prompts = [r["prompt"] for r in prompt_rows]
    results = generate_batch(
        model, tokenizer, prompts, gen_spec, seed=seed,
        pad_token_id=tokenizer.pad_token_id,
    )
    records = []
    for row, result in zip(prompt_rows, results):
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
        records.append(rec)
    return records


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
    verbose: bool = True,
    batch_size: int = 8,
) -> None:
    """Load config and prompts, run base and finetuned inference, write outputs and metrics.
    Uses batched generation when batch_size > 1 for faster eval (better GPU utilization).
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log("=" * 60)
    log("RUN SAMPLES (base + finetuned inference)")
    log("=" * 60)
    log(f"  Run dir:   {run_dir}")
    log(f"  Prompts:   {prompts_path}")
    log(f"  Batch size: {batch_size} (batched generation for faster eval)")

    cfg = _load_config(run_dir)
    prompts = _load_prompts(prompts_path)
    n_prompts = len(prompts)
    log(f"  Prompts loaded: {n_prompts}")
    _ensure_adapter(run_dir)
    log("  Adapter:   OK")
    log("")

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

    # Generation params (fixed for both runs). Use at least 2048 tokens so full envelope+payload
    # is not truncated (truncation at 1024 produces incomplete JSON and 0% schema_valid).
    data_cfg = cfg.get("data", {})
    max_new_tokens = max(int(data_cfg.get("max_seq_len", 1024)), 2048)
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

    log(f"  Device:    {device}, precision: {precision}")
    log(f"  Max new tokens: {max_new_tokens} (ensures full envelope+payload not truncated)")
    log("")
    log("Loading base model (no adapter)...")
    base_model, tokenizer = load_model_and_tokenizer(model_spec, runtime_spec)
    log("  Base model loaded.")
    log("")

    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    base_records = []
    for start in range(0, n_prompts, batch_size):
        batch_rows = prompts[start : start + batch_size]
        batch_ids = [r.get("id", "") for r in batch_rows]
        log(f"  Base batch {start // batch_size + 1}/{(n_prompts + batch_size - 1) // batch_size}  ids={batch_ids[:3]}{'...' if len(batch_ids) > 3 else ''}")
        batch_records = _run_inference_batch(
            base_model, tokenizer, batch_rows, gen_spec, gen_seed
        )
        base_records.extend(batch_records)
        for rec in batch_records:
            log(f"             id={rec.get('prompt_id')} -> {rec.get('latency_ms', 0):.0f} ms  parse_ok={rec.get('parse_success')}  schema_ok={rec.get('schema_valid')}")

    base_out = samples_dir / "base_outputs.jsonl"
    with open(base_out, "w", encoding="utf-8") as f:
        for rec in base_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"  Base outputs written: {base_out} ({len(base_records)} records)")
    log("")

    # Free base model before loading finetuned (save memory)
    log("Freeing base model...")
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()
    log("Loading base model + LoRA adapter (finetuned)...")
    from peft import PeftModel

    base_model, tokenizer = load_model_and_tokenizer(model_spec, runtime_spec)
    adapter_path = run_dir / "adapter"
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    log("  Finetuned model loaded.")
    log("")

    debug_path = samples_dir / "schema_failures_debug.jsonl"
    debug_file = open(debug_path, "w", encoding="utf-8")
    try:
        finetuned_records = []
        for start in range(0, n_prompts, batch_size):
            batch_rows = prompts[start : start + batch_size]
            batch_ids = [r.get("id", "") for r in batch_rows]
            log(f"  Finetuned batch {start // batch_size + 1}/{(n_prompts + batch_size - 1) // batch_size}  ids={batch_ids[:3]}{'...' if len(batch_ids) > 3 else ''}")
            batch_records = _run_inference_batch(
                model, tokenizer, batch_rows, gen_spec, gen_seed
            )
            finetuned_records.extend(batch_records)
            for rec in batch_records:
                log(f"             id={rec.get('prompt_id')} -> {rec.get('latency_ms', 0):.0f} ms  parse_ok={rec.get('parse_success')}  schema_ok={rec.get('schema_valid')}")
                if not rec.get("schema_valid") and rec.get("parse_success"):
                    err_code = rec.get("error_code") or "?"
                    missing = rec.get("missing_fields") or []
                    err_msg = (rec.get("schema_error_message") or "")[:120]
                    keys = rec.get("parsed_top_level_keys") or []
                    log(f"             [schema fail] error_code={err_code}  missing_fields={missing}  parsed_keys={keys}")
                    if err_msg:
                        log(f"             [schema fail] message: {err_msg}...")
                    entry = {
                        "prompt_id": rec.get("prompt_id"),
                        "error_code": rec.get("error_code"),
                        "missing_fields": rec.get("missing_fields"),
                        "schema_error_message": (rec.get("schema_error_message") or "")[:300],
                        "parsed_top_level_keys": rec.get("parsed_top_level_keys"),
                        "raw_output_len": len(rec.get("raw_text_output") or ""),
                    }
                    debug_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    debug_file.flush()
        n_failures = sum(1 for r in finetuned_records if r.get("parse_success") and not r.get("schema_valid"))
        if n_failures:
            log(f"  Schema failures debug: {debug_path} ({n_failures} entries)")
    finally:
        debug_file.close()

    finetuned_out = samples_dir / "finetuned_outputs.jsonl"
    with open(finetuned_out, "w", encoding="utf-8") as f:
        for rec in finetuned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"  Finetuned outputs written: {finetuned_out} ({len(finetuned_records)} records)")
    log("")

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

    log("=" * 60)
    log("COMPLETE")
    log("=" * 60)
    log(f"  Base:       {base_out}")
    log(f"  Finetuned:  {finetuned_out}")
    log(f"  Metrics:    {metrics_path}")
    log("")
    log("  Finetuned aggregate:")
    log(f"    parse_success_rate:   {finetuned_metrics.get('parse_success_rate', 0):.2%}")
    log(f"    schema_valid_rate:    {finetuned_metrics.get('schema_valid_rate', 0):.2%}")
    log(f"    avg_latency_ms:       {finetuned_metrics.get('avg_latency_ms')}")
    log("=" * 60)


def recompute_sample_metrics(run_dir: Path) -> None:
    """Re-parse raw_text_output from existing sample JSONL and recompute parse_success/schema_valid and comparison_metrics. No model load."""
    samples_dir = run_dir / "samples"
    base_path = samples_dir / "base_outputs.jsonl"
    finetuned_path = samples_dir / "finetuned_outputs.jsonl"
    if not base_path.exists() or not finetuned_path.exists():
        raise FileNotFoundError(
            f"Missing {base_path} or {finetuned_path}. Run sample generation first."
        )

    def _load_jsonl(p: Path) -> list[dict[str, Any]]:
        records = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _reparse_and_rebuild(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for r in records:
            raw = r.get("raw_text_output", "")
            parse_result = parse_and_validate(raw, strict_json=False)
            prompt_row = {
                "id": r.get("prompt_id"),
                "prompt": r.get("prompt", ""),
                "payload_type": r.get("payload_type"),
            }
            rec = _build_sample_record(
                prompt_row,
                raw,
                parse_result,
                latency_ms=r.get("latency_ms") or 0,
                tokens_generated=r.get("tokens_generated"),
            )
            out.append(rec)
        return out

    base_records = _reparse_and_rebuild(_load_jsonl(base_path))
    finetuned_records = _reparse_and_rebuild(_load_jsonl(finetuned_path))

    with open(base_path, "w", encoding="utf-8") as f:
        for rec in base_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(finetuned_path, "w", encoding="utf-8") as f:
        for rec in finetuned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    base_metrics = _compute_aggregate_metrics(base_records)
    finetuned_metrics = _compute_aggregate_metrics(finetuned_records)
    delta = _compute_delta(base_metrics, finetuned_metrics)
    comparison = {
        "num_prompts": len(base_records),
        "base": base_metrics,
        "finetuned": finetuned_metrics,
        "delta": delta,
    }
    metrics_path = samples_dir / "comparison_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Recomputed {base_path.name}, {finetuned_path.name}, {metrics_path.name}")


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
        default=None,
        help="Path to fixed prompts JSONL (e.g. eval/suites/fixed20_v1.jsonl); required unless --recompute_metrics",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8). Larger = faster eval, more VRAM.",
    )
    parser.add_argument(
        "--recompute_metrics",
        action="store_true",
        help="Re-parse existing sample JSONL and recompute metrics (no model load)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    if args.recompute_metrics:
        try:
            recompute_sample_metrics(run_dir)
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        return

    if not args.prompts:
        print("--prompts required unless --recompute_metrics", file=sys.stderr)
        sys.exit(1)
    prompts_path = Path(args.prompts).resolve()
    project_root = Path(args.project_root or ".").resolve()

    try:
        run_samples(
            run_dir,
            prompts_path,
            project_root,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
