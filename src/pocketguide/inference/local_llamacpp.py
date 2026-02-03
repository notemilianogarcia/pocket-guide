"""
Local inference backend via llama.cpp (Milestone 6 — Lesson 6.3).

Loads runtime_local.yaml; supports stub mode (deterministic envelope) and real
execution via llama.cpp binary. Returns raw text + metadata for unified CLI.
"""

import json
import os
import subprocess
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
    """Return a deterministic PocketGuide envelope (no model)."""
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


def _resolve_llamacpp_bin(cfg: dict[str, Any], project_root: Path) -> Path | None:
    """Resolve llama.cpp binary: config execution.llamacpp_bin or auto-detect."""
    execution = cfg.get("execution") or {}
    bin_path = execution.get("llamacpp_bin")
    if bin_path is not None and str(bin_path).strip():
        p = Path(bin_path)
        if p.is_absolute():
            return p if p.exists() else None
        return (project_root / p).resolve() if (project_root / p).exists() else None
    # Auto-detect: common names in PATH
    for name in ("llama-cli", "main", "llama"):
        try:
            result = subprocess.run(
                ["which", name],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return Path(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    return None


def run_llamacpp(
    prompt: str,
    runtime_cfg: dict[str, Any],
    project_root: Path | None = None,
) -> dict[str, Any]:
    """
    Run local inference: stub or real llama.cpp.

    Args:
        prompt: User prompt.
        runtime_cfg: Full runtime config (from runtime_local.yaml).
        project_root: Repo root for resolving relative paths; default cwd.

    Returns:
        Dict with: raw_text_output (str), latency_ms (float), tokens_generated (int | None), command_used (str).
    """
    if project_root is None:
        project_root = Path.cwd()

    runtime = runtime_cfg.get("runtime") or {}
    stub = runtime.get("stub", True)

    if stub:
        envelope = _stub_envelope(prompt)
        raw_text = json.dumps(envelope, indent=2)
        return {
            "raw_text_output": raw_text,
            "latency_ms": 0.0,
            "tokens_generated": None,
            "command_used": "[stub] no binary",
        }

    # Real execution: validate config
    model_cfg = runtime_cfg.get("model") or {}
    gguf_path_raw = model_cfg.get("gguf_path")
    if not gguf_path_raw or not str(gguf_path_raw).strip():
        raise ValueError("model.gguf_path is required when runtime.stub is false")
    gguf_path = Path(gguf_path_raw)
    if not gguf_path.is_absolute():
        gguf_path = (project_root / gguf_path).resolve()
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF model not found: {gguf_path}")

    llamacpp_bin = _resolve_llamacpp_bin(runtime_cfg, project_root)
    if llamacpp_bin is None:
        raise FileNotFoundError(
            "llama.cpp binary not found. Set execution.llamacpp_bin in config or ensure llama-cli/main is in PATH."
        )

    gen = runtime_cfg.get("generation") or {}
    context_length = int(gen.get("context_length", 4096))
    temperature = float(gen.get("temperature", 0.2))
    top_p = float(gen.get("top_p", 0.9))
    max_tokens = int(gen.get("max_tokens", 512))

    extra_args = runtime_cfg.get("execution") or {}
    extra_list = extra_args.get("extra_args") or []

    # Build command: -m model -p prompt -c ctx -n n_predict --temp T --top-p P
    cmd = [
        str(llamacpp_bin),
        "-m",
        str(gguf_path),
        "-p",
        prompt,
        "-c",
        str(context_length),
        "-n",
        str(max_tokens),
        "--temp",
        str(temperature),
        "--top-p",
        str(top_p),
    ] + list(extra_list)

    import time
    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(project_root),
        )
    except subprocess.TimeoutExpired as e:
        print(f"llama.cpp timed out: {e}", file=sys.stderr)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"llama.cpp exited with code {result.returncode}. See stderr for details."
        )

    raw_text = result.stdout.strip() if result.stdout else ""
    # tokens_generated: best-effort; llama.cpp may print token count to stderr
    tokens_generated = None

    return {
        "raw_text_output": raw_text,
        "latency_ms": elapsed_ms,
        "tokens_generated": tokens_generated,
        "command_used": " ".join(cmd),
    }


def generate_raw_text_local(
    prompt: str,
    config_path: Path,
    project_root: Path | None = None,
    gguf_path_override: str | Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Generate raw text using local backend (stub or llama.cpp).

    Returns:
        (raw_text, metadata) where metadata has latency_ms, tokens_generated, command_used.
    """
    cfg = _load_config(config_path)
    # Optional override for model.gguf_path (used by CLI/registry and local eval)
    if gguf_path_override is not None:
        override_path = Path(gguf_path_override)
        if not override_path.is_absolute():
            override_path = (project_root or Path.cwd()) / override_path
            override_path = override_path.resolve()
        if "model" not in cfg:
            cfg["model"] = {}
        cfg["model"]["gguf_path"] = str(override_path)

    out = run_llamacpp(prompt, cfg, project_root=project_root)
    return out["raw_text_output"], {
        "latency_ms": out["latency_ms"],
        "tokens_generated": out.get("tokens_generated"),
        "command_used": out.get("command_used", ""),
    }
