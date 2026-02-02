"""
Local runtime CLI (Milestone 6 — Lesson 6.1).

Config-driven entrypoint for llama.cpp / GGUF. In stub mode returns a
deterministic PocketGuide response envelope (no model required).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load runtime_local.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _stub_envelope(prompt: str) -> dict[str, Any]:
    """Return a deterministic PocketGuide envelope (no timestamps)."""
    # Deterministic payload: procedure stub
    return {
        "summary": (
            "Stub response for local runtime (llama.cpp). "
            "Set runtime.stub=false and provide a GGUF model to run real inference."
        ),
        "assumptions": [
            "No model loaded; stub mode.",
            "Prompt was received but not executed by a language model.",
        ],
        "uncertainty_notes": "This is placeholder output. Enable real inference in configs/runtime_local.yaml.",
        "next_steps": [
            "Place a GGUF model in models/gguf/ and set model.gguf_path in config.",
            "Set runtime.stub: false to call llama.cpp.",
        ],
        "verification_steps": [
            "Check configs/runtime_local.yaml for runtime.stub and model.gguf_path.",
        ],
        "payload_type": "procedure",
        "payload": {
            "title": "Local runtime setup",
            "steps": [
                {"step": 1, "instruction": "Add a GGUF model to models/gguf/"},
                {"step": 2, "instruction": "Set model.gguf_path in configs/runtime_local.yaml"},
                {"step": 3, "instruction": "Set runtime.stub to false"},
            ],
        },
    }


def run(config_path: Path, prompt: str) -> str:
    """Load config; if stub, return JSON envelope; else raise. Returns JSON string for stdout."""
    cfg = _load_config(config_path)
    runtime = cfg.get("runtime") or {}
    stub = runtime.get("stub", True)
    if stub:
        envelope = _stub_envelope(prompt)
        return json.dumps(envelope, indent=2)
    raise NotImplementedError(
        "llama.cpp execution not implemented yet; set runtime.stub=true"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PocketGuide local runtime CLI (llama.cpp stub)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to runtime config (e.g. configs/runtime_local.yaml)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()

    try:
        out = run(config_path, args.prompt)
        print(out)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
