"""
Thin runner around the unified CLI that supports model selection via registry.

Usage:

    python -m pocketguide.inference.run \
      --prompt "plan a 2-day itinerary for Montreal" \
      --local \
      --model adapted \
      --registry configs/model_registry.yaml

This avoids parsing YAML in the Makefile and lets `make run LOCAL=1 MODEL=...`
select a GGUF model by name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pocketguide.artifacts.registry import resolve_model_gguf
from pocketguide.eval.parsing import parse_and_validate
from pocketguide.inference import cli as unified_cli


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PocketGuide inference runner with model registry support.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The travel-related prompt/question to process",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local runtime (llama.cpp or stub) instead of HF stub.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="adapted",
        help="Logical model name from model registry (e.g. base, adapted). Default: adapted.",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="configs/model_registry.yaml",
        help="Path to model registry YAML (default: configs/model_registry.yaml).",
    )
    parser.add_argument(
        "--runtime_config",
        type=str,
        default="configs/runtime_local.yaml",
        help="Path to local runtime config (used only when --local).",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[3]  # src/pocketguide/inference -> repo root

    runtime = "local" if args.local else "hf"

    runtime_config_path = Path(args.runtime_config)
    if not runtime_config_path.is_absolute():
        runtime_config_path = (project_root / runtime_config_path).resolve()

    gguf_path: Path | None = None
    if runtime == "local":
        gguf_path = resolve_model_gguf(
            model_name=args.model,
            registry_path=args.registry,
            project_root=project_root,
        )

    try:
        raw_text, metadata = unified_cli.get_raw_text(
            runtime,
            args.prompt,
            runtime_config_path if runtime == "local" else None,
            project_root,
            gguf_path=gguf_path,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except (KeyError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)

    if metadata.get("latency_ms") is not None:
        print(f"latency_ms: {metadata['latency_ms']:.2f}", file=sys.stderr)
    if metadata.get("command_used"):
        cmd = metadata["command_used"]
        print(f"command_used: {cmd[:200]}...", file=sys.stderr) if len(cmd) > 200 else print(
            f"command_used: {cmd}", file=sys.stderr
        )

    # Parse and validate, mirroring unified CLI behavior
    parse_result = parse_and_validate(raw_text, strict_json=True)
    if not parse_result.success and parse_result.error is not None:
        parse_result = parse_and_validate(raw_text, strict_json=False)
    if not parse_result.success:
        # Delegate to unified CLI's error envelope builder
        envelope = unified_cli._build_error_envelope(parse_result, raw_excerpt=raw_text[:1000])  # type: ignore[attr-defined]
        print(json.dumps(envelope, indent=2))
        sys.exit(0)

    assert parse_result.data is not None
    print(json.dumps(parse_result.data, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

