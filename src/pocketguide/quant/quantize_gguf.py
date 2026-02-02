"""
Quantization pipeline v1 (GGUF export) — Lesson 6.2.

Takes base HF model + LoRA adapter, optionally merges, exports to GGUF via
llama.cpp, quantizes to configurable formats (e.g. Q4_K_M, Q5_K_M).
Writes artifacts under runs/quant/<run_id>/ with config snapshot and meta.json.
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from pocketguide.utils.run_id import make_run_id
from pocketguide.utils.run_logging import (
    compute_file_hash,
    get_git_commit,
    get_package_version,
    write_json,
    write_yaml,
)


def load_config(path: Path) -> dict[str, Any]:
    """Load quant config YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML object (dict).")
    return cfg


def resolve_llamacpp_dir(cfg: dict[str, Any]) -> Path:
    """Resolve llama.cpp repo directory from config or LLAMACPP_DIR env."""
    repo_dir = cfg.get("llamacpp", {}).get("repo_dir")
    if repo_dir is not None and str(repo_dir).strip():
        path = Path(repo_dir).resolve()
    else:
        env_dir = os.environ.get("LLAMACPP_DIR")
        if not env_dir or not env_dir.strip():
            raise ValueError(
                "llama.cpp directory not set. Set config input.llamacpp.repo_dir "
                "or environment variable LLAMACPP_DIR."
            )
        path = Path(env_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"llama.cpp directory does not exist: {path}")
    return path


def get_repo_git_commit(repo_dir: Path) -> str | None:
    """Return HEAD commit hash of a git repo, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
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


def detect_quantize_binary(llamacpp_dir: Path, config_override: str | None) -> Path:
    """Detect quantize binary: config override, else build/bin/llama-quantize, else build/bin/quantize."""
    if config_override and str(config_override).strip():
        p = Path(config_override)
        if p.is_absolute():
            return p
        return llamacpp_dir / p
    build_bin = llamacpp_dir / "build" / "bin"
    for name in ("llama-quantize", "quantize"):
        candidate = build_bin / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Quantize binary not found. Looked in {build_bin} for 'llama-quantize' or 'quantize'. "
        "Build llama.cpp first (cmake .. && cmake --build .)."
    )


def create_run_dir(cfg: dict[str, Any], run_id: str, project_root: Path) -> Path:
    """Create runs/quant/<run_id> and return path."""
    runs_dir = project_root / (cfg.get("output", {}).get("runs_dir", "runs/quant")).strip()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_initial_meta(
    cfg: dict[str, Any],
    run_id: str,
    run_dir: Path,
    project_root: Path,
    llamacpp_dir: Path,
    planned_commands: list[str] | None = None,
) -> dict[str, Any]:
    """Build initial meta.json (provenance; commands filled later)."""
    import datetime

    meta: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "package_versions": {
            "torch": get_package_version("torch"),
            "transformers": get_package_version("transformers"),
            "peft": get_package_version("peft"),
        },
        "llamacpp": {
            "repo_dir": str(llamacpp_dir),
            "git_commit": get_repo_git_commit(llamacpp_dir),
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "run_dir": str(run_dir),
        "commands": planned_commands if planned_commands is not None else [],
        "outputs": {},
    }
    return meta


def run_merge(
    base_model_id: str,
    adapter_dir: Path,
    merged_dir: Path,
) -> None:
    """Load base model + PEFT adapter, merge, save to merged_dir."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise ImportError("Merge step requires transformers and peft. Install with pip install transformers peft.") from e

    print("Loading base model and tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    print("Loading PEFT adapter...", file=sys.stderr)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    print("Merging and unloading adapter...", file=sys.stderr)
    model = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    print("Saving merged model...", file=sys.stderr)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print("Merge complete.", file=sys.stderr)


def run_subprocess_logged(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run command, writing stdout and stderr to log_path. Return completed process."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
            text=True,
        )
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PocketGuide quantization pipeline: merge adapter → GGUF → quantize.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/quantize_gguf.yaml"), help="Quant config YAML")
    parser.add_argument("--dry_run", action="store_true", help="Print planned commands and write run dir + meta only")
    parser.add_argument("--base_model_id", type=str, default=None, help="Override config input.base_model_id")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Override config input.adapter_dir")
    parser.add_argument("--llamacpp_dir", type=str, default=None, help="Override config / LLAMACPP_DIR")
    parser.add_argument("--run_id", type=str, default=None, help="Use this run ID instead of generating one")
    args = parser.parse_args()

    # Repo root: src/pocketguide/quant/quantize_gguf.py -> parents[3]
    project_root = Path(__file__).resolve().parents[3]
    if not args.config.is_absolute():
        args.config = project_root / args.config

    cfg = load_config(args.config)

    # Overrides
    inp = cfg.setdefault("input", {})
    if args.base_model_id is not None:
        inp["base_model_id"] = args.base_model_id
    if args.adapter_dir is not None:
        inp["adapter_dir"] = args.adapter_dir
    if args.llamacpp_dir is not None:
        cfg.setdefault("llamacpp", {})["repo_dir"] = args.llamacpp_dir

    base_model_id = inp.get("base_model_id") or ""
    adapter_dir_raw = inp.get("adapter_dir") or ""
    adapter_dir = (project_root / adapter_dir_raw.strip()).resolve() if adapter_dir_raw else Path()

    if not base_model_id:
        print("Missing input.base_model_id in config or --base_model_id.", file=sys.stderr)
        sys.exit(1)

    # Resolve llama.cpp dir (required even for dry_run so we can record paths)
    try:
        llamacpp_dir = resolve_llamacpp_dir(cfg)
    except (ValueError, FileNotFoundError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    run_id = args.run_id or make_run_id()
    try:
        run_dir = create_run_dir(cfg, run_id, project_root)
    except FileExistsError as e:
        print(f"Run directory already exists: {e}", file=sys.stderr)
        sys.exit(1)

    # Directories under run_dir
    out = cfg.get("output", {})
    merged_dirname = cfg.get("merge", {}).get("merged_output_dirname", "merged_hf")
    merged_dir = run_dir / merged_dirname
    gguf_dir = run_dir / "gguf"
    logs_dir = run_dir / "logs"

    # Config snapshot
    write_yaml(run_dir / "config.yaml", cfg)

    # Paths for convert/quant
    convert_script = cfg.get("llamacpp", {}).get("convert_script", "convert_hf_to_gguf.py")
    convert_script_path = llamacpp_dir / convert_script
    f16_filename = cfg.get("gguf", {}).get("f16_filename", "model.f16.gguf")
    f16_gguf = gguf_dir / f16_filename
    quant_formats = cfg.get("quant", {}).get("formats", ["Q4_K_M", "Q5_K_M"])
    output_template = cfg.get("quant", {}).get("output_template", "model.{quant}.gguf")

    merge_enabled = cfg.get("merge", {}).get("enabled", True)
    # For convert script: merged HF dir if we merge, else base model id/path as-is
    hf_source_for_convert = str(merged_dir) if merge_enabled else base_model_id

    # Planned / executed commands
    planned: list[str] = []

    # 1) Merge command (conceptual; we run Python, not a shell command)
    if merge_enabled:
        planned.append(f"[merge] Python: load base {base_model_id} + adapter {adapter_dir} -> save {merged_dir}")

    # 2) Convert command
    convert_cmd = [
        sys.executable,
        str(convert_script_path),
        hf_source_for_convert,
        "--outfile",
        str(f16_gguf),
    ]
    planned.append(" ".join(convert_cmd))

    # 3) Quantize commands (need quantize binary path; may fail in dry_run if not built)
    try:
        quantize_bin = detect_quantize_binary(
            llamacpp_dir,
            cfg.get("llamacpp", {}).get("quantize_bin"),
        )
    except FileNotFoundError as e:
        if args.dry_run:
            planned.append(f"[quantize] binary not found: {e}")
        else:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        quantize_bin = None
    else:
        for q in quant_formats:
            out_name = output_template.format(quant=q)
            out_path = gguf_dir / out_name
            q_cmd = [str(quantize_bin), str(f16_gguf), str(out_path), q]
            planned.append(" ".join(q_cmd))

    meta = build_initial_meta(cfg, run_id, run_dir, project_root, llamacpp_dir, planned_commands=planned)
    write_json(run_dir / "meta.json", meta)

    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    if args.dry_run:
        print("Dry run: planned commands:")
        for c in planned:
            print(f"  {c}")
        return

    # --- Real execution ---

    # Merge
    if merge_enabled:
        if not adapter_dir.is_dir():
            print(f"Adapter directory not found: {adapter_dir}", file=sys.stderr)
            sys.exit(1)
        run_merge(base_model_id, adapter_dir, merged_dir)
        hf_source_for_convert = merged_dir
    else:
        hf_source_for_convert = base_model_id

    gguf_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Convert
    if not convert_script_path.exists():
        print(f"Convert script not found: {convert_script_path}. Clone llama.cpp and ensure {convert_script} exists.", file=sys.stderr)
        sys.exit(1)
    convert_log = logs_dir / "convert.log"
    proc = run_subprocess_logged(convert_cmd, convert_log)
    if proc.returncode != 0:
        print(f"Convert failed (see {convert_log}).", file=sys.stderr)
        sys.exit(1)

    # Quantize
    quantize_bin = detect_quantize_binary(llamacpp_dir, cfg.get("llamacpp", {}).get("quantize_bin"))
    for q in quant_formats:
        out_name = output_template.format(quant=q)
        out_path = gguf_dir / out_name
        q_cmd = [str(quantize_bin), str(f16_gguf), str(out_path), q]
        q_log = logs_dir / f"quantize_{q}.log"
        proc = run_subprocess_logged(q_cmd, q_log)
        if proc.returncode != 0:
            print(f"Quantize {q} failed (see {q_log}).", file=sys.stderr)
            sys.exit(1)

    # Finalize meta: commands + output hashes/sizes
    meta["commands"] = planned
    outputs: dict[str, Any] = {}
    for f in [f16_gguf] + [gguf_dir / output_template.format(quant=q) for q in quant_formats]:
        if f.exists():
            try:
                outputs[str(f.name)] = {"sha256": compute_file_hash(f), "size_bytes": f.stat().st_size}
            except Exception:
                outputs[str(f.name)] = {"size_bytes": f.stat().st_size}
    meta["outputs"] = outputs
    write_json(run_dir / "meta.json", meta)
    print("Quantization complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
