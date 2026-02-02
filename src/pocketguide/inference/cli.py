"""CLI for PocketGuide inference.

Unified entrypoint: --runtime hf | local. Outputs are validated against the
PocketGuide envelope; parse failures return a structured error envelope (JSON).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pocketguide.eval.parsing import ParseResult, ParseValidationError, parse_and_validate


# --- Backward compatibility for benchmark.py (legacy stub + format) ---

def generate_stub_response(prompt: str) -> dict[str, str]:
    """Legacy stub: returns dict with summary, assumptions, next_steps (for benchmark)."""
    return {
        "summary": (
            "This is a stub response for travel-related query. "
            "The actual model will provide detailed travel guidance."
        ),
        "assumptions": (
            "No external data sources available (offline mode). "
            "Response based on prompt content only. "
            "Model inference not yet implemented."
        ),
        "next_steps": (
            "1. Review the response structure\n"
            "2. Verify deterministic output\n"
            "3. Proceed with evaluation pipeline"
        ),
    }


def format_response(response: dict[str, str]) -> str:
    """Legacy: format response dict into sectioned text (for benchmark)."""
    return (
        f"Summary:\n{response['summary']}\n\n"
        f"Assumptions:\n{response['assumptions']}\n\n"
        f"Next steps:\n{response['next_steps']}"
    )


# --- HF runtime (stub: returns envelope-shaped JSON so it validates) ---

def _hf_stub_envelope(prompt: str) -> dict[str, Any]:
    """Deterministic stub envelope for --runtime hf (no model load)."""
    return {
        "summary": (
            "This is a stub response for travel-related query. "
            "The actual model will provide detailed travel guidance. "
            "Use --runtime local with a GGUF model for real local inference."
        ),
        "assumptions": [
            "No external data sources available (offline mode).",
            "Response based on prompt content only.",
            "Model inference not yet loaded; stub only.",
        ],
        "uncertainty_notes": "Stub mode. Load a model or use local runtime for real inference.",
        "next_steps": [
            "Review the response structure.",
            "Use --runtime local with configs/runtime_local.yaml for local GGUF inference.",
        ],
        "verification_steps": [
            "Inspect raw output in logs/run artifacts.",
        ],
        "payload_type": "procedure",
        "payload": {
            "title": "Stub procedure",
            "steps": [
                {"step": 1, "instruction": "Review the response structure"},
                {"step": 2, "instruction": "Verify deterministic output"},
                {"step": 3, "instruction": "Proceed with evaluation or use --runtime local"},
            ],
        },
    }


def generate_raw_text_hf(prompt: str) -> tuple[str, dict[str, Any]]:
    """Generate raw text for HF runtime (stub). Returns (raw_text, metadata)."""
    envelope = _hf_stub_envelope(prompt)
    return json.dumps(envelope, indent=2), {"latency_ms": 0.0, "tokens_generated": None, "command_used": "[hf stub]"}


# --- Error envelope (parse/validation failed) ---

def _build_error_envelope(parse_result: ParseResult, raw_excerpt: str = "") -> dict[str, Any]:
    """Build a valid PocketGuide envelope describing the failure (payload_type=procedure)."""
    err = parse_result.error
    if err is None:
        err = ParseValidationError(
            code="UNKNOWN",
            error_type="unknown",
            message="Parse/validation failed",
            guidance=[],
        )
    guidance = getattr(err, "guidance", None) or []
    if isinstance(guidance, str):
        guidance = [guidance]
    next_steps = [
        "Try again with lower temperature",
        "Use a different runtime/model",
        "Check that the model was trained for PocketGuide envelope output",
    ]
    verification_steps = ["Inspect raw output in logs/run artifacts"]
    steps = [{"step": 1, "instruction": f"Error: {err.message}"}]
    for i, g in enumerate(guidance[:5], start=2):
        steps.append({"step": i, "instruction": g})
    return {
        "summary": "Parsing/validation failed",
        "assumptions": [],
        "uncertainty_notes": "Model output did not match required JSON envelope.",
        "next_steps": next_steps,
        "verification_steps": verification_steps,
        "payload_type": "procedure",
        "payload": {
            "title": "Parsing/validation failed",
            "steps": steps,
            "extra": {
                "error_type": err.error_type,
                "message": err.message,
                "guidance": guidance,
                "raw_excerpt": raw_excerpt[:500] if raw_excerpt else "",
            },
        },
    }


# --- Unified flow ---

def get_raw_text(runtime: str, prompt: str, runtime_config_path: Path | None, project_root: Path | None) -> tuple[str, dict[str, Any]]:
    """Get raw model output from the selected runtime. Returns (raw_text, metadata)."""
    if runtime == "hf":
        return generate_raw_text_hf(prompt)
    if runtime == "local":
        if runtime_config_path is None or not runtime_config_path.exists():
            raise FileNotFoundError(f"Local runtime config not found: {runtime_config_path}")
        from pocketguide.inference.local_llamacpp import generate_raw_text_local
        return generate_raw_text_local(prompt, runtime_config_path, project_root=project_root)
    raise ValueError(f"Unknown runtime: {runtime}. Use 'hf' or 'local'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PocketGuide CLI - Offline travel assistant inference (unified runtime).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The travel-related prompt/question to process",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        choices=["hf", "local"],
        default="hf",
        help="Inference backend: hf (stub) or local (llama.cpp / stub)",
    )
    parser.add_argument(
        "--runtime_config",
        type=str,
        default="configs/runtime_local.yaml",
        help="Path to runtime config (used only for --runtime local)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="json",
        help="Output format (default: json); always JSON envelope when using --runtime local",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]  # src/pocketguide/inference -> repo root
    runtime_config_path = Path(args.runtime_config)
    if not runtime_config_path.is_absolute():
        runtime_config_path = (project_root / runtime_config_path).resolve()

    try:
        raw_text, metadata = get_raw_text(args.runtime, args.prompt, runtime_config_path, project_root)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)

    if metadata.get("latency_ms") is not None:
        print(f"latency_ms: {metadata['latency_ms']:.2f}", file=sys.stderr)
    if metadata.get("command_used"):
        print(f"command_used: {metadata['command_used'][:200]}...", file=sys.stderr) if len(metadata.get("command_used", "")) > 200 else print(f"command_used: {metadata['command_used']}", file=sys.stderr)

    # Parse and validate
    parse_result = parse_and_validate(raw_text, strict_json=True)
    if not parse_result.success and parse_result.error is not None:
        parse_result = parse_and_validate(raw_text, strict_json=False)
    if not parse_result.success:
        envelope = _build_error_envelope(parse_result, raw_excerpt=raw_text[:1000])
        print(json.dumps(envelope, indent=2))
        sys.exit(0)  # Structured error is valid output
    assert parse_result.data is not None
    print(json.dumps(parse_result.data, indent=2))


if __name__ == "__main__":
    main()
