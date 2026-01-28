"""Benchmark evaluation script for PocketGuide.

Runs inference on evaluation suites and saves structured outputs with metadata.
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from pocketguide.eval.metrics_v0 import (
    aggregate_metrics,
    check_required_fields,
    detect_uncertainty_markers,
    lenient_json_extract_and_parse,
    strict_json_parse,
)
from pocketguide.inference.cli import format_response, generate_stub_response
from pocketguide.utils.run_id import make_run_id


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load JSONL file with error handling.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of parsed JSON objects

    Raises:
        ValueError: If JSON parsing fails or required fields missing
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation suite not found: {file_path}")

    examples = []
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip blank lines
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {file_path} at line {line_num}: {e}"
                ) from e

            # Validate required fields
            required = ["id", "prompt"]
            missing = [k for k in required if k not in item]
            if missing:
                raise ValueError(
                    f"Missing required fields in {file_path} at line {line_num}: {missing}"
                )

            examples.append(item)

    return examples


def save_jsonl(data: list[dict[str, Any]], file_path: Path) -> None:
    """Save data as JSONL file.

    Args:
        data: List of dictionaries to save
        file_path: Output file path
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def save_json(data: dict[str, Any], file_path: Path) -> None:
    """Save data as formatted JSON file.

    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_git_commit() -> str | None:
    """Get current git commit hash if available.

    Returns:
        Git commit hash or None if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_git_is_dirty() -> bool | None:
    """Check if git repo has uncommitted changes.

    Returns:
        True if dirty, False if clean, None if git unavailable
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_package_version(package_name: str) -> str:
    """Get installed package version.

    Args:
        package_name: Name of the package

    Returns:
        Version string or "unknown"
    """
    try:
        import importlib.metadata

        return importlib.metadata.version(package_name)
    except Exception:
        return "unknown"


def run_smoke_inference(
    config_path: Path,
    prompt: str,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> None:
    """Run single-prompt smoke test with real model inference.

    Args:
        config_path: Path to eval.yaml config file
        prompt: Input prompt for generation
        out_dir: Override output directory from config
        verbose: Print progress messages
    """
    # Load eval config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Determine output directory
    if out_dir is None:
        out_dir = Path(cfg.get("out_root", "runs/eval"))

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Running smoke inference...")
        print(f"Output directory: {run_dir}")
        print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}")

    # Import base_model (delay import to avoid dependency if not needed)
    try:
        from pocketguide.inference.base_model import (
            GenSpec,
            ModelSpec,
            RuntimeSpec,
            generate_one,
            load_model_and_tokenizer,
        )
    except ImportError as e:
        raise ImportError("Transformers/torch not installed. Run: pip install -e .[dev]") from e

    # Build specs from config
    model_cfg = cfg.get("model", {})
    model_spec = ModelSpec(id=model_cfg.get("id", "REPLACE_ME"), revision=model_cfg.get("revision"))

    runtime_spec = RuntimeSpec(device=cfg.get("device", "cpu"), dtype=cfg.get("dtype", "float32"))

    gen_cfg = cfg.get("gen", {})
    gen_spec = GenSpec(
        max_new_tokens=gen_cfg.get("max_new_tokens", 256),
        do_sample=gen_cfg.get("do_sample", False),
        temperature=gen_cfg.get("temperature", 0.0),
        top_p=gen_cfg.get("top_p", 1.0),
        repetition_penalty=gen_cfg.get("repetition_penalty", 1.0),
    )

    seed = cfg.get("seed", 42)

    # Load model
    if verbose:
        print(f"\nLoading model: {model_spec.id}")
    model, tokenizer = load_model_and_tokenizer(model_spec, runtime_spec)

    # Generate
    if verbose:
        print("Generating response...")
    result = generate_one(model, tokenizer, prompt, gen_spec, seed)

    # Save result
    output_file = run_dir / "smoke_infer.json"
    output_data = {
        "timestamp": timestamp,
        "prompt": prompt,
        "model": {"id": model_spec.id, "revision": model_spec.revision},
        "seed": seed,
        "generation_config": {
            "max_new_tokens": gen_spec.max_new_tokens,
            "do_sample": gen_spec.do_sample,
            "temperature": gen_spec.temperature,
            "top_p": gen_spec.top_p,
            "repetition_penalty": gen_spec.repetition_penalty,
        },
        "result": result,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"\nGenerated text:\n{result['text'][:200]}...")
        print(f"\nUsage: {result['usage']}")
        print(f"Timing: {result['timing']}")
        print(f"\nSaved to: {output_file}")

    # Also print JSON to stdout for easy parsing
    print(json.dumps(output_data))


def write_meta(
    out_dir: Path,
    cfg: dict[str, Any],
    suite_paths: list[Path],
    run_id: str,
    config_paths: dict[str, Path] | None = None,
) -> None:
    """Write comprehensive metadata file for reproducible evaluation.

    Args:
        out_dir: Output directory for meta.json
        cfg: Evaluation configuration dictionary
        suite_paths: List of paths to evaluation suites
        run_id: Run ID string (YYYYMMDD_HHMMSS)
        config_paths: Optional dict of config file paths for snapshot
    """
    # Gather suite information with per-suite counts
    suites_info = []
    suite_counts = {}
    for suite_path in suite_paths:
        suite_name = suite_path.stem
        suite_info = {
            "name": suite_name,
            "path": str(suite_path),
            "exists": suite_path.exists(),
        }
        if suite_path.exists():
            try:
                examples = load_jsonl(suite_path)
                suite_info["num_examples"] = len(examples)
                suite_counts[suite_name] = len(examples)
            except Exception as e:
                suite_info["error"] = str(e)
                suite_info["num_examples"] = 0
        else:
            suite_info["num_examples"] = 0
        suites_info.append(suite_info)

    # Snapshot config file contents
    configs_raw = {}
    if config_paths:
        for _name, path in config_paths.items():
            if path and path.exists():
                try:
                    configs_raw[str(path)] = path.read_text(encoding="utf-8")
                except Exception:
                    configs_raw[str(path)] = "<error reading file>"

    # Build enhanced metadata
    model_cfg = cfg.get("model", {})
    model_id = model_cfg.get("id", "REPLACE_ME")

    # Check for placeholder model ID
    if model_id == "REPLACE_ME":
        import warnings

        warnings.warn(
            "Model ID is still 'REPLACE_ME'. Update configs/eval.yaml with a real model ID "
            "or HuggingFace model path (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')",
            UserWarning,
            stacklevel=2,
        )

    meta = {
        # Run identification
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "timezone": datetime.now().astimezone().tzname(),

        # Model identity
        "model_id": model_id,
        "revision": model_cfg.get("revision"),

        # Runtime config (resolved values)
        "seed": cfg.get("seed", 42),
        "device": cfg.get("device", "cpu"),
        "dtype": cfg.get("dtype", "float32"),
        "generation_config": cfg.get("gen", {}),

        # Suite manifest
        "suite_dir": str(suite_paths[0].parent) if suite_paths else None,
        "suite_files": [str(p) for p in suite_paths],
        "suite_counts": suite_counts,
        "suites": suites_info,

        # Config snapshots
        "eval_config_resolved": cfg,
        "configs_raw": configs_raw,

        # Environment snapshot
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "sys_platform": sys.platform,
            "machine": platform.machine(),
        },

        # Package versions
        "package_versions": {
            "pocketguide": get_package_version("pocketguide"),
            "pyyaml": get_package_version("pyyaml"),
            "pytest": get_package_version("pytest"),
            "ruff": get_package_version("ruff"),
            "torch": get_package_version("torch"),
            "transformers": get_package_version("transformers"),
            "jsonschema": get_package_version("jsonschema"),
        },

        # Git snapshot
        "git_commit": get_git_commit(),
        "git_is_dirty": get_git_is_dirty(),

        # Output location
        "run_dir": str(out_dir),
    }

    meta_path = out_dir / "meta.json"
    save_json(meta, meta_path)


def run_suite_directory(
    config_path: Path | None = None,
    suite_dir: Path | None = None,
    out_dir: Path | None = None,
    run_id: str | None = None,
    verbose: bool = True,
) -> None:
    """Run benchmark evaluation on all suites in a directory.

    Args:
        config_path: Path to eval.yaml config file (default: configs/eval.yaml)
        suite_dir: Path to directory containing JSONL suite files
        out_dir: Override output directory from config
        run_id: Optional run ID (YYYYMMDD_HHMMSS). If None, generated from current time.
        verbose: Print progress messages
    """
    # Load evaluation config
    if config_path is None:
        config_path = Path("configs/eval.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Determine suite directory
    if suite_dir is None:
        suite_dir = Path(cfg.get("suite_dir", "data/benchmarks/v0"))

    if not suite_dir.exists():
        raise FileNotFoundError(f"Suite directory not found: {suite_dir}")

    # Get all JSONL files in directory, sorted for determinism
    suite_paths = sorted(suite_dir.glob("*.jsonl"))
    if not suite_paths:
        raise ValueError(f"No JSONL files found in: {suite_dir}")

    # Determine output directory
    if out_dir is None:
        out_dir = Path(cfg.get("out_root", "runs/eval"))

    # Generate or use provided run_id
    if run_id is None:
        run_id = make_run_id()

    # Create run directory
    run_dir = out_dir / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. "
            "Use a different --run_id or remove the existing directory."
        )
    run_dir.mkdir(parents=True, exist_ok=False)

    if verbose:
        print(f"Run ID: {run_id}")
        print(f"Output directory: {run_dir}")
        print(f"Suite directory: {suite_dir}")
        print(f"Found {len(suite_paths)} suite(s)")

    # Snapshot config files for provenance
    config_paths = {"eval": config_path}

    # Write comprehensive metadata
    write_meta(run_dir, cfg, suite_paths, run_id, config_paths)

    if verbose:
        print(f"Saved metadata to: {run_dir / 'meta.json'}")

    # Run inference on all suites
    all_outputs = []
    total_examples = 0

    for suite_path in suite_paths:
        suite_name = suite_path.stem  # filename without .jsonl

        if verbose:
            print(f"\nLoading suite: {suite_name}")

        examples = load_jsonl(suite_path)

        if verbose:
            print(f"Loaded {len(examples)} examples from {suite_name}")

        # Run inference on each example
        for idx, example in enumerate(examples, 1):
            example_id = example.get("id", f"{suite_name}_{idx}")
            prompt = example.get("prompt", "")
            required_fields = example.get("required_fields", [])

            if verbose and idx % max(1, len(examples) // 5) == 0:
                print(f"  Processing {idx}/{len(examples)}...")

            # Generate response (stub for now)
            response = generate_stub_response(prompt)
            output_text = format_response(response)

            # Compute checks
            strict_ok, strict_parsed, _ = strict_json_parse(output_text)
            lenient_ok, lenient_parsed, _ = lenient_json_extract_and_parse(output_text)

            # Required fields check (only if required_fields specified)
            required_fields_ok = None
            missing_fields = []
            if required_fields and lenient_ok and lenient_parsed:
                required_fields_ok, missing_fields = check_required_fields(lenient_parsed, required_fields)

            # Uncertainty markers
            uncertainty = detect_uncertainty_markers(output_text)

            # Build output record
            output_record = {
                "id": example_id,
                "suite": suite_name,
                "prompt": prompt,
                "output_text": output_text,
                "usage": response.get("usage", {}),
                "timing": response.get("timing", {}),
                "model": cfg.get("model", {}).get("id", "REPLACE_ME"),
                "seed": cfg.get("seed", 42),
                "gen": cfg.get("gen", {}),
                "checks": {
                    "strict_json_ok": strict_ok,
                    "lenient_json_ok": lenient_ok,
                    "required_fields_ok": required_fields_ok,
                    "missing_fields": missing_fields,
                    "uncertainty": uncertainty,
                },
            }

            # Add parsed JSON preview if available (top-level keys only for dicts)
            if lenient_ok and lenient_parsed:
                if isinstance(lenient_parsed, dict):
                    output_record["parsed_json_preview"] = {"keys": list(lenient_parsed.keys())}
                elif isinstance(lenient_parsed, list):
                    output_record["parsed_json_preview"] = {"type": "array", "length": len(lenient_parsed)}

            all_outputs.append(output_record)

        total_examples += len(examples)

    # Save outputs
    if all_outputs:
        outputs_path = run_dir / "base_model_outputs.jsonl"
        save_jsonl(all_outputs, outputs_path)

        if verbose:
            print(f"\nSaved {len(all_outputs)} outputs to: {outputs_path}")

        # Compute and save metrics
        metrics = aggregate_metrics(all_outputs)
        metrics_path = run_dir / "metrics.json"
        save_json(metrics, metrics_path)

        if verbose:
            print(f"Saved metrics to: {metrics_path}")
            print("\nOverall metrics:")
            print(
                f"  Strict JSON parse rate: {metrics['overall'].get('strict_json_parse_rate', 0):.1%}"
            )
            print(
                f"  Lenient JSON parse rate: {metrics['overall'].get('lenient_json_parse_rate', 0):.1%}"
            )
            if metrics["overall"].get("required_fields_rate") is not None:
                print(
                    f"  Required fields rate: {metrics['overall'].get('required_fields_rate', 0):.1%}"
                )
            print(
                f"  Assumptions marker rate: {metrics['overall'].get('assumptions_marker_rate', 0):.1%}"
            )

    if verbose:
        print(
            f"Evaluation complete! Processed {total_examples} examples from {len(suite_paths)} suite(s)"
        )


def run_benchmark(
    config_path: Path | None = None,
    suite_path: Path | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> None:
    """Run benchmark evaluation on a suite.

    Args:
        config_path: Path to eval.yaml config file (default: configs/eval.yaml)
        suite_path: Override suite path from config
        out_dir: Override output directory from config
        verbose: Print progress messages
    """
    # Load evaluation config
    if config_path is None:
        config_path = Path("configs/eval.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Determine output directory
    if out_dir is None:
        out_dir = Path(cfg.get("out_root", "runs/eval"))

    # Determine which suites to run
    if suite_path is not None:
        # Single suite override
        suite_paths = [suite_path]
    else:
        # Use suites from config
        suites_cfg = cfg.get("suites", [])
        if not suites_cfg:
            raise ValueError("No suites specified in config")
        suite_paths = [Path(s["path"]) for s in suites_cfg]

    # Create timestamped output directory
    run_id = make_run_id()
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Output directory: {run_dir}")

    # Snapshot config files for provenance
    config_paths = {"eval": config_path}

    # Write comprehensive metadata
    write_meta(run_dir, cfg, suite_paths, run_id, config_paths)

    if verbose:
        print(f"Saved metadata to: {run_dir / 'meta.json'}")

    # Run inference on all suites
    all_predictions = []
    for suite_path_current in suite_paths:
        if not suite_path_current.exists():
            if verbose:
                print(f"Warning: Suite not found: {suite_path_current}")
            continue

        if verbose:
            print(f"\nLoading evaluation suite: {suite_path_current}")

        examples = load_jsonl(suite_path_current)

        if verbose:
            print(f"Loaded {len(examples)} examples")

        # Run inference on each example
        for idx, example in enumerate(examples, 1):
            if verbose:
                print(f"Processing example {idx}/{len(examples)}: {example.get('id', 'unknown')}")

            example_id = example.get("id", f"example_{idx}")
            prompt = example.get("prompt", "")

            # Generate response (stub for now)
            response = generate_stub_response(prompt)
            response_text = format_response(response)

            all_predictions.append(
                {
                    "id": example_id,
                    "prompt": prompt,
                    "response": response,
                    "response_text": response_text,
                    "suite": str(suite_path_current),
                }
            )

    # Save predictions if any
    if all_predictions and cfg.get("save_predictions", True):
        predictions_path = run_dir / "predictions.jsonl"
        save_jsonl(all_predictions, predictions_path)

        if verbose:
            print(f"\nSaved predictions to: {predictions_path}")

    if verbose:
        print("\nEvaluation complete!")


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(description="Run PocketGuide evaluation benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Path to eval config file (default: configs/eval.yaml)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Override suite path from config (single file)",
    )
    parser.add_argument(
        "--suite_dir",
        type=str,
        default=None,
        help="Override suite directory from config (multiple JSONL files)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Explicit run ID (YYYYMMDD_HHMMSS). If not provided, generated from current time.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    parser.add_argument(
        "--smoke_infer",
        action="store_true",
        help="Run smoke inference on a single prompt (requires --prompt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What are visa requirements for Japan?",
        help="Prompt for smoke inference (default: travel question)",
    )

    args = parser.parse_args()

    # Convert to Path objects
    config_path = Path(args.config)
    suite_path = Path(args.suite) if args.suite else None
    suite_dir = Path(args.suite_dir) if args.suite_dir else None
    out_dir = Path(args.out_dir) if args.out_dir else None

    # Run appropriate mode
    try:
        if args.smoke_infer:
            # Smoke inference mode
            run_smoke_inference(
                config_path=config_path,
                prompt=args.prompt,
                out_dir=out_dir,
                verbose=not args.quiet,
            )
        elif args.suite_dir or suite_dir:
            # Multi-suite directory mode
            run_suite_directory(
                config_path=config_path,
                suite_dir=suite_dir,
                out_dir=out_dir,
                run_id=args.run_id,
                verbose=not args.quiet,
            )
        else:
            # Normal benchmark mode
            run_benchmark(
                config_path=config_path,
                suite_path=suite_path,
                out_dir=out_dir,
                verbose=not args.quiet,
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
