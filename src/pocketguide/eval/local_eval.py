"""
Local eval: latency + schema compliance regression (Milestone 6 — Lesson 6.4).

Runs local runtime inference over prompt suites (fixed20 + local regression),
records latency, approx tokens/sec, parse/schema rates. Writes runs/eval/<run_id>/local_metrics.json.
"""

import json
import sys
from pathlib import Path
from typing import Any

from pocketguide.eval.benchmark import load_jsonl
from pocketguide.eval.parsing import ParseResult, parse_and_validate
from pocketguide.inference.local_llamacpp import _load_config, run_llamacpp
from pocketguide.utils.run_id import make_run_id


def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear interpolation percentile."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    rank = (p / 100.0) * (n - 1)
    lower_idx = int(rank)
    if lower_idx >= n - 1:
        return sorted_values[-1]
    upper_idx = lower_idx + 1
    fraction = rank - lower_idx
    return sorted_values[lower_idx] + fraction * (sorted_values[upper_idx] - sorted_values[lower_idx])


def _contract_from_parse(strict_result: ParseResult, lenient_result: ParseResult | None) -> dict[str, Any]:
    """Build contract booleans and error summary from parse results."""
    strict_ok = strict_result.success
    if strict_ok:
        return {
            "strict_parse_ok": True,
            "lenient_parse_ok": True,
            "envelope_valid": True,
            "payload_valid": True,
            "error_type": None,
            "failed_at": None,
            "guidance_summary": None,
        }
    lenient_ok = lenient_result.success if lenient_result else False
    err = strict_result.error or (lenient_result.error if lenient_result else None)
    envelope_ok = False
    payload_ok = False
    if lenient_result and lenient_result.success:
        envelope_ok = True
        payload_ok = True
    elif err:
        if err.code == "ENVELOPE_SCHEMA_FAILED":
            envelope_ok = False
            payload_ok = False
        elif err.code == "PAYLOAD_SCHEMA_FAILED":
            envelope_ok = True
            payload_ok = False
    guidance_summary = (err.guidance[:2] if getattr(err, "guidance", None) else None) or []
    if isinstance(guidance_summary, str):
        guidance_summary = [guidance_summary]
    return {
        "strict_parse_ok": strict_ok,
        "lenient_parse_ok": lenient_ok,
        "envelope_valid": envelope_ok,
        "payload_valid": payload_ok,
        "error_type": err.error_type if err else None,
        "failed_at": err.failed_at if err else None,
        "guidance_summary": guidance_summary[:3] if guidance_summary else None,
    }


def _run_one_prompt(
    prompt: str,
    suite_name: str,
    example_id: str,
    runtime_cfg: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """Run local inference for one prompt; return per-sample record."""
    try:
        out = run_llamacpp(prompt, runtime_cfg, project_root=project_root)
    except Exception as e:
        return {
            "suite": suite_name,
            "id": example_id,
            "latency_ms": None,
            "approx_tokens": None,
            "tokens_per_sec": None,
            "strict_parse_ok": False,
            "lenient_parse_ok": False,
            "envelope_valid": False,
            "payload_valid": False,
            "error_type": "runtime_error",
            "failed_at": None,
            "guidance_summary": [str(e)],
            "raw_output_excerpt": None,
        }
    raw_text = out["raw_text_output"]
    latency_ms = out["latency_ms"]
    tokens_generated = out.get("tokens_generated")
    approx_tokens = int(tokens_generated) if tokens_generated is not None else max(1, len(raw_text) // 4)
    latency_s = (latency_ms or 0) / 1000.0
    tokens_per_sec = approx_tokens / latency_s if latency_s > 0 else 0.0

    strict_result = parse_and_validate(raw_text, strict_json=True)
    lenient_result = None
    if not strict_result.success:
        lenient_result = parse_and_validate(raw_text, strict_json=False)
    contract = _contract_from_parse(strict_result, lenient_result)

    return {
        "suite": suite_name,
        "id": example_id,
        "latency_ms": latency_ms,
        "approx_tokens": approx_tokens,
        "tokens_per_sec": tokens_per_sec,
        "strict_parse_ok": contract["strict_parse_ok"],
        "lenient_parse_ok": contract["lenient_parse_ok"],
        "envelope_valid": contract["envelope_valid"],
        "payload_valid": contract["payload_valid"],
        "error_type": contract["error_type"],
        "failed_at": contract["failed_at"],
        "guidance_summary": contract["guidance_summary"],
        "raw_output_excerpt": raw_text[:500] if raw_text else None,
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-suite style metrics (latency ms, tokens/sec approx, parse/schema rates)."""
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "avg_latency_ms": None,
            "p50_latency_ms": None,
            "p90_latency_ms": None,
            "avg_tokens_per_sec_approx": None,
            "p50_tokens_per_sec_approx": None,
            "p90_tokens_per_sec_approx": None,
            "strict_parse_rate": 0.0,
            "lenient_parse_rate": 0.0,
            "envelope_valid_rate": 0.0,
            "payload_valid_rate": 0.0,
        }
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    tps_list = [r["tokens_per_sec"] for r in records if r.get("tokens_per_sec") is not None]
    latencies = sorted(latencies)
    tps_list = sorted(tps_list)
    return {
        "n": n,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "p50_latency_ms": _percentile(latencies, 50) if latencies else None,
        "p90_latency_ms": _percentile(latencies, 90) if latencies else None,
        "avg_tokens_per_sec_approx": sum(tps_list) / len(tps_list) if tps_list else None,
        "p50_tokens_per_sec_approx": _percentile(tps_list, 50) if tps_list else None,
        "p90_tokens_per_sec_approx": _percentile(tps_list, 90) if tps_list else None,
        "strict_parse_rate": sum(1 for r in records if r.get("strict_parse_ok")) / n,
        "lenient_parse_rate": sum(1 for r in records if r.get("lenient_parse_ok")) / n,
        "envelope_valid_rate": sum(1 for r in records if r.get("envelope_valid")) / n,
        "payload_valid_rate": sum(1 for r in records if r.get("payload_valid")) / n,
    }


def run_local_eval(
    runtime_config_path: Path,
    suite_paths: list[Path],
    out_dir: Path,
    run_id: str | None = None,
    project_root: Path | None = None,
    write_outputs_jsonl: bool = True,
) -> Path:
    """
    Run local eval over suites; write local_metrics.json (and optionally local_outputs.jsonl).

    Returns:
        Path to run directory (out_dir / run_id).
    """
    if project_root is None:
        project_root = Path.cwd()
    if run_id is None:
        run_id = make_run_id()
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    runtime_cfg = _load_config(runtime_config_path)
    all_records: list[dict[str, Any]] = []

    for suite_path in suite_paths:
        if not suite_path.exists():
            print(f"Suite not found: {suite_path}", file=sys.stderr)
            continue
        suite_name = suite_path.stem
        examples = load_jsonl(suite_path)
        for ex in examples:
            example_id = ex.get("id", "unknown")
            prompt = ex.get("prompt", "")
            record = _run_one_prompt(prompt, suite_name, example_id, runtime_cfg, project_root)
            all_records.append(record)

    by_suite: dict[str, list[dict]] = {}
    for r in all_records:
        s = r["suite"]
        if s not in by_suite:
            by_suite[s] = []
        by_suite[s].append(r)

    per_suite = {name: _aggregate(recs) for name, recs in by_suite.items()}
    overall = _aggregate(all_records)

    stub = (runtime_cfg.get("runtime") or {}).get("stub", True)
    metrics = {
        "run_id": run_id,
        "runtime_stub": stub,
        "suites": [str(p) for p in suite_paths],
        "per_suite": per_suite,
        "overall": overall,
        "tokens_per_sec_note": "approximate (len/4 when not from model); labeled _approx in keys",
    }
    if stub:
        metrics["_note"] = "runtime_stub=true: no real model; latency 0 and parse rates reflect stub output only."
    metrics_path = run_dir / "local_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if write_outputs_jsonl and all_records:
        outputs_path = run_dir / "local_outputs.jsonl"
        with open(outputs_path, "w", encoding="utf-8") as f:
            for r in all_records:
                # Redact long raw excerpt for readability
                out = {k: v for k, v in r.items() if k != "raw_output_excerpt"}
                if r.get("raw_output_excerpt"):
                    out["raw_output_excerpt_len"] = len(r["raw_output_excerpt"])
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return run_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local eval: latency + schema compliance over prompt suites.")
    parser.add_argument(
        "--runtime_config",
        type=Path,
        default=Path("configs/runtime_local.yaml"),
        help="Path to runtime config (runtime_local.yaml)",
    )
    parser.add_argument(
        "--suites",
        type=str,
        default="eval/suites/fixed20_v1.jsonl,eval/suites/local_regression_v1.jsonl",
        help="Comma-separated paths to JSONL suite files",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/eval"),
        help="Output directory for runs/eval/<run_id>",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run ID (default: generated timestamp)",
    )
    parser.add_argument(
        "--no_outputs_jsonl",
        action="store_true",
        help="Skip writing local_outputs.jsonl",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    runtime_config_path = args.runtime_config if args.runtime_config.is_absolute() else (project_root / args.runtime_config)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (project_root / args.out_dir)
    suite_paths = [project_root / p.strip() for p in args.suites.split(",") if p.strip()]

    try:
        run_dir = run_local_eval(
            runtime_config_path=runtime_config_path,
            suite_paths=suite_paths,
            out_dir=out_dir,
            run_id=args.run_id,
            project_root=project_root,
            write_outputs_jsonl=not args.no_outputs_jsonl,
        )
    except FileExistsError as e:
        print(f"Run directory already exists: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print(f"Run ID: {run_dir.name}")
    print(f"Run directory: {run_dir}")
    print(f"Metrics: {run_dir / 'local_metrics.json'}")
    overall = json.loads((run_dir / "local_metrics.json").read_text())["overall"]
    print(f"  n: {overall['n']}, strict_parse_rate: {overall.get('strict_parse_rate', 0):.2%}, envelope_valid_rate: {overall.get('envelope_valid_rate', 0):.2%}")


if __name__ == "__main__":
    main()
