"""
Measure local runtime performance for a packaged model.

Runs a small prompt suite (e.g. fixed10) through the local runtime (llama.cpp or stub)
and records:
- approximate load time (warm-up prompt latency)
- latency distribution over the suite
- approximate RAM usage (best-effort)

Writes results to:
    artifacts/models/<model_name>/perf.json

CLI:
    python -m pocketguide.artifacts.measure_perf \
      --model_name <name> \
      --registry configs/model_registry.yaml \
      --suite eval/suites/fixed10_v1.jsonl \
      --runtime_config configs/runtime_local.yaml \
      --out artifacts/models/<name>/perf.json \
      [--allow_stub]

In tests / dev, --allow_stub lets the script run even when runtime.stub=true, so
no llama.cpp or real GGUF is required.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import psutil  # type: ignore[import]
except Exception:  # pragma: no cover - psutil may not be installed
    psutil = None  # type: ignore[assignment]

import yaml

from pocketguide.artifacts.registry import resolve_model_gguf
from pocketguide.eval.benchmark import load_jsonl
from pocketguide.inference.local_llamacpp import _load_config, run_llamacpp
from pocketguide.utils.run_logging import compute_file_hash, get_git_commit


@dataclass
class LatencyStats:
    avg: float
    p50: float
    p90: float


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    rank = (p / 100.0) * (n - 1)
    lower = int(rank)
    if lower >= n - 1:
        return values[-1]
    upper = lower + 1
    frac = rank - lower
    return values[lower] + frac * (values[upper] - values[lower])


def _compute_latency_stats(latencies: Iterable[float]) -> LatencyStats:
    vals = list(latencies)
    if not vals:
        return LatencyStats(avg=0.0, p50=0.0, p90=0.0)
    avg = sum(vals) / len(vals)
    p50 = _percentile(vals, 50.0)
    p90 = _percentile(vals, 90.0)
    return LatencyStats(avg=avg, p50=p50, p90=p90)


def _infer_quant_from_gguf_path(gguf_path: Path) -> str | None:
    name = gguf_path.name
    if not name.endswith(".gguf"):
        return None
    parts = name.rsplit(".", 2)
    if len(parts) < 2:
        return None
    quant = parts[-2]
    return quant or None


def _approx_tokens_and_tps(latencies_ms: list[float], texts: list[str]) -> tuple[LatencyStats, str]:
    """Approximate tokens/sec from output length; no llama.cpp introspection."""
    if not latencies_ms or not texts:
        return LatencyStats(0.0, 0.0, 0.0), "unavailable"
    approx_tokens: list[float] = []
    tps: list[float] = []
    for lat_ms, txt in zip(latencies_ms, texts):
        # Rough heuristic: 1 token ~ 4 characters
        tokens = max(1.0, len(txt) / 4.0)
        approx_tokens.append(tokens)
        lat_s = max(lat_ms / 1000.0, 1e-6)
        tps.append(tokens / lat_s)
    stats = _compute_latency_stats(tps)
    return stats, "approx_output_len_div4"


def _measure_ram_usage_self() -> tuple[dict[str, Any], str]:
    """Best-effort RAM usage: current process RSS via psutil (if available)."""
    if psutil is None:
        return {"peak_rss_bytes": None, "notes": "psutil not available"}, "unavailable"
    proc = psutil.Process()
    rss = proc.memory_info().rss
    return {"peak_rss_bytes": int(rss), "notes": "single-sample self RSS"}, "psutil_self_rss"


def measure_perf_main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Measure local runtime performance for a packaged model.")
    parser.add_argument("--model_name", required=True, help="Logical model name (e.g. base, adapted)")
    parser.add_argument(
        "--registry",
        type=str,
        default="configs/model_registry.yaml",
        help="Path to model registry YAML (default: configs/model_registry.yaml)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="eval/suites/fixed10_v1.jsonl",
        help="Prompt suite for perf measurement (default: eval/suites/fixed10_v1.jsonl)",
    )
    parser.add_argument(
        "--runtime_config",
        type=str,
        default="configs/runtime_local.yaml",
        help="Path to runtime_local.yaml (must have runtime.stub=false unless --allow_stub)",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Output path for perf.json (default: artifacts/models/<model_name>/perf.json)",
    )
    parser.add_argument(
        "--allow_stub",
        action="store_true",
        help="Allow runtime.stub=true (for tests/dev). Real measurement should run with stub=false.",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[3]

    gguf_path = resolve_model_gguf(args.model_name, args.registry, project_root=project_root)
    quant = _infer_quant_from_gguf_path(gguf_path)

    # Resolve runtime config and suite paths
    runtime_config_path = Path(args.runtime_config)
    if not runtime_config_path.is_absolute():
        runtime_config_path = (project_root / runtime_config_path).resolve()
    suite_path = Path(args.suite)
    if not suite_path.is_absolute():
        suite_path = (project_root / suite_path).resolve()

    if not suite_path.exists():
        print(f"Prompt suite not found: {suite_path}", file=sys.stderr)
        sys.exit(1)

    # Load runtime config and enforce stub rule (unless allow_stub)
    runtime_cfg = _load_config(runtime_config_path)
    runtime = runtime_cfg.get("runtime") or {}
    stub = bool(runtime.get("stub", True))
    if stub and not args.allow_stub:
        print(
            "runtime.stub is true in runtime_local.yaml. Set stub=false for real perf measurement "
            "or pass --allow_stub for dev/testing.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Override GGUF path for measurement
    if "model" not in runtime_cfg:
        runtime_cfg["model"] = {}
    runtime_cfg["model"]["gguf_path"] = str(gguf_path)

    # Load prompts
    examples = load_jsonl(suite_path)
    prompts = [ex.get("prompt", "") for ex in examples]
    num_prompts = len(prompts)

    # Warm-up: use first prompt (or a simple default)
    warmup_prompt = prompts[0] if prompts else "Explain local runtime performance for PocketGuide."

    # Approximate load time as warm-up latency (best-effort)
    t0 = time.perf_counter()
    warmup_out = run_llamacpp(warmup_prompt, runtime_cfg, project_root=project_root)
    t1 = time.perf_counter()
    warmup_latency_ms = warmup_out.get("latency_ms", (t1 - t0) * 1000.0)
    load_time_ms = warmup_latency_ms
    warmup_total_ms = warmup_latency_ms

    # Measure suite latencies
    latencies_ms: list[float] = []
    outputs: list[str] = []
    for prompt in prompts:
        out = run_llamacpp(prompt, runtime_cfg, project_root=project_root)
        lat_ms = float(out.get("latency_ms") or 0.0)
        latencies_ms.append(lat_ms)
        outputs.append(out.get("raw_text_output", ""))

    latency_stats = _compute_latency_stats(latencies_ms)
    tps_stats, tps_method = _approx_tokens_and_tps(latencies_ms, outputs)

    ram_info, ram_method = _measure_ram_usage_self()
    ram_info["method"] = ram_method

    # Hardware info
    hw = {
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": None,
        "python": platform.python_version(),
    }
    if psutil is not None:
        try:
            hw["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)  # type: ignore[union-attr]
        except Exception:
            hw["ram_gb"] = None

    # Provenance
    model_dir = project_root / "artifacts" / "models" / args.model_name
    meta_path = model_dir / "meta.json"
    meta_hash = compute_file_hash(meta_path) if meta_path.exists() else None

    perf = {
        "model_name": args.model_name,
        "gguf_path": str(gguf_path),
        "quant": quant,
        "hardware": hw,
        "measurement": {
            "suite": str(suite_path.relative_to(project_root)),
            "num_prompts": num_prompts,
            "load_time_ms": load_time_ms,
            "warmup_total_latency_ms": warmup_total_ms,
            "latency_ms": asdict(latency_stats),
            "tokens_per_sec": {
                "avg": tps_stats.avg,
                "p50": tps_stats.p50,
                "p90": tps_stats.p90,
                "tokens_estimate_method": tps_method,
            },
            "ram_usage": ram_info,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": get_git_commit(project_root) or None,
            "artifacts_meta_sha256": meta_hash,
        },
    }

    # Determine output path
    out_path = Path(args.out) if args.out else model_dir / "perf.json"
    if not out_path.is_absolute():
        out_path = (project_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(perf, indent=2), encoding="utf-8")

    print(f"Wrote performance metrics to {out_path}", file=sys.stderr)


def main() -> None:  # pragma: no cover - thin wrapper
    measure_perf_main()


if __name__ == "__main__":  # pragma: no cover
    main()

