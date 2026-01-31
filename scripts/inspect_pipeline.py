#!/usr/bin/env python3
"""Inspect pipeline artifacts for debugging: drafts, rejected samples, and stats.

Usage:
  # Use sample dirs (data/interim/sample, data/processed/sample)
  python scripts/inspect_pipeline.py --sample

  # Or specify paths explicitly
  python scripts/inspect_pipeline.py \\
    --drafts data/interim/sample/drafts_v1.jsonl \\
    --rejected data/processed/sample/dataset_v1_rejected.jsonl \\
    --stats data/processed/sample/dataset_v1_stats.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file; return list of records."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def load_json(path: Path) -> dict | None:
    """Load JSON file."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def trunc(s: str, n: int = 400) -> str:
    """Truncate string for display."""
    if not s:
        return ""
    return s[:n] + ("..." if len(s) > n else "")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect pipeline artifacts (drafts, rejected, stats) for debugging"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample dirs: data/interim/sample, data/processed/sample",
    )
    parser.add_argument(
        "--drafts",
        type=Path,
        help="Path to drafts_v1.jsonl",
    )
    parser.add_argument(
        "--rejected",
        type=Path,
        help="Path to dataset_v1_rejected.jsonl",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        help="Path to dataset_v1_stats.json",
    )
    parser.add_argument(
        "--max_output",
        type=int,
        default=400,
        help="Max chars of output_text/raw_refined_text to show (default 400)",
    )
    args = parser.parse_args()

    if args.sample:
        base_interim = Path("data/interim/sample")
        base_processed = Path("data/processed/sample")
        drafts_path = base_interim / "drafts_v1.jsonl"
        rejected_path = base_processed / "dataset_v1_rejected.jsonl"
        stats_path = base_processed / "dataset_v1_stats.json"
    else:
        drafts_path = args.drafts
        rejected_path = args.rejected
        stats_path = args.stats
        if not drafts_path and not rejected_path and not stats_path:
            print("Provide --sample or at least one of --drafts, --rejected, --stats", file=sys.stderr)
            return 1

    n = args.max_output

    # --- Drafts ---
    if drafts_path and drafts_path.exists():
        drafts = load_jsonl(drafts_path)
        print("\n" + "=" * 60)
        print("DRAFTS")
        print("=" * 60)
        print(f"File: {drafts_path} | Records: {len(drafts)}")
        for i, d in enumerate(drafts):
            c = d.get("contract", {})
            ok = c.get("overall_ok", False)
            err = c.get("error", {})
            out = d.get("output_text") or ""
            print(f"\n  [{i+1}] id={d.get('id', '?')} | overall_ok={ok}")
            if not ok and err:
                print(f"      error: code={err.get('code')} | message={err.get('message', '')[:200]!r} | failed_at={err.get('failed_at')}")
            print(f"      output_text prefix: {repr(trunc(out, n))}")
    elif drafts_path:
        print(f"\n[DRAFTS] File not found: {drafts_path}")

    # --- Rejected (non–missing_inputs only) ---
    if rejected_path and rejected_path.exists():
        rejected = load_jsonl(rejected_path)
        non_missing = [r for r in rejected if r.get("reason") != "missing_inputs"]
        print("\n" + "=" * 60)
        print("REJECTED (refinement/gates; excluding missing_inputs)")
        print("=" * 60)
        print(f"File: {rejected_path} | Total rejected: {len(rejected)} | Refinement/gates: {len(non_missing)}")
        for i, r in enumerate(non_missing):
            reason_codes = r.get("reason_codes", [])
            contract = r.get("contract", {})
            errors = contract.get("errors", [])
            raw = r.get("raw_refined_text", "") or ""
            print(f"\n  [{i+1}] id={r.get('id', '?')} | reason_codes={reason_codes}")
            print(f"      contract.errors: {errors!r}")
            print(f"      raw_refined_text prefix: {repr(trunc(raw, n))}")
    elif rejected_path:
        print(f"\n[REJECTED] File not found: {rejected_path}")

    # --- Stats ---
    if stats_path and stats_path.exists():
        stats = load_json(stats_path)
        print("\n" + "=" * 60)
        print("STATS")
        print("=" * 60)
        print(f"File: {stats_path}")
        if stats:
            summary = stats.get("summary", stats)
            print(f"  accepted: {summary.get('accepted', '?')}")
            print(f"  rejected: {summary.get('rejected', '?')}")
            print(f"  total_attempted: {summary.get('total_attempted', '?')}")
            print(f"  rejection_reasons (gate_failures): {stats.get('rejection_reasons', '?')}")
            print(f"  skipped: {stats.get('skipped', stats)}")
    elif stats_path:
        print(f"\n[STATS] File not found: {stats_path}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
