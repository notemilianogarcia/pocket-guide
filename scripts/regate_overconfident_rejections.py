#!/usr/bin/env python3
"""Re-apply quality gates to rejections that failed only on overconfident_time_sensitive.

After fixing the overconfidence gate to allow "always verify" style phrases, run this
script to re-gate those rejected samples and append newly accepted ones to the dataset.

Usage (from repo root):
  uv run python scripts/regate_overconfident_rejections.py [--dry-run]
  # or
  python scripts/regate_overconfident_rejections.py [--dry-run]

Options:
  --dry-run   Only report what would be appended; do not write dataset or manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path so we can import pocketguide
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pocketguide.data_generation.generate_dataset_v1 import (
    JoinedSample,
    build_accepted_record,
    load_jsonl,
    parse_refined_output,
)
from pocketguide.data_generation.quality_gates import (
    apply_all_gates,
    compute_overall_quality,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-gate overconfident_time_sensitive rejections and merge accepted into dataset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be appended; do not write files",
    )
    parser.add_argument(
        "--rejected",
        type=Path,
        default=Path("data/processed/dataset_v1_rejected.jsonl"),
        help="Path to rejected JSONL",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/dataset_v1.jsonl"),
        help="Path to accepted dataset JSONL to append to",
    )
    parser.add_argument(
        "--plans",
        type=Path,
        default=Path("data/interim/prompt_plan_v1.jsonl"),
        help="Path to prompt plan JSONL",
    )
    parser.add_argument(
        "--drafts",
        type=Path,
        default=Path("data/interim/drafts_v1.jsonl"),
        help="Path to drafts JSONL",
    )
    parser.add_argument(
        "--critiques",
        type=Path,
        default=Path("data/interim/critiques_v1.jsonl"),
        help="Path to critiques JSONL",
    )
    args = parser.parse_args()

    # Resolve paths from cwd (repo root)
    repo = Path.cwd()
    rejected_path = repo / args.rejected
    dataset_path = repo / args.dataset
    plans_path = repo / args.plans
    drafts_path = repo / args.drafts
    critiques_path = repo / args.critiques

    for p in [rejected_path, plans_path, drafts_path, critiques_path]:
        if not p.exists():
            print(f"[ERROR] Missing: {p}", file=sys.stderr)
            return 1

    # Load inputs
    plans = load_jsonl(plans_path)
    drafts = load_jsonl(drafts_path)
    critiques = load_jsonl(critiques_path)
    plan_by_id = {p["id"]: p for p in plans}
    draft_by_id = {d["id"]: d for d in drafts}
    critique_by_id = {c["draft_id"]: c for c in critiques}

    rejected = load_jsonl(rejected_path)
    overconfident_only = [
        r
        for r in rejected
        if r.get("reason_codes") == ["overconfident_time_sensitive"]
    ]

    if not overconfident_only:
        print("[INFO] No rejections with reason_codes == ['overconfident_time_sensitive']; nothing to re-gate.")
        return 0

    print(f"[INFO] Found {len(overconfident_only)} rejection(s) to re-gate (overconfident_time_sensitive only).")

    gating_mode = "lenient"
    now_accept = []

    for r in overconfident_only:
        sample_id = r.get("id")
        if not sample_id:
            continue
        plan = plan_by_id.get(sample_id)
        draft = draft_by_id.get(sample_id)
        critique = critique_by_id.get(sample_id)
        if not plan or not draft:
            print(f"[WARN] Skipping {sample_id}: missing plan or draft", file=sys.stderr)
            continue

        raw_text = r.get("raw_refined_text") or ""
        if not raw_text.strip():
            print(f"[WARN] Skipping {sample_id}: empty raw_refined_text", file=sys.stderr)
            continue

        parsed_envelope, contract = parse_refined_output(raw_text, gating_mode)
        if not parsed_envelope:
            print(f"[WARN] Skipping {sample_id}: re-parse failed (contract errors: {contract.get('errors')})", file=sys.stderr)
            continue

        envelope_ok = contract.get("envelope_ok", False)
        payload_ok = contract.get("payload_ok", False)
        parse_mode = contract.get("parse_mode", "lenient")

        gates = apply_all_gates(
            envelope_ok=envelope_ok,
            payload_ok=payload_ok,
            parse_mode=parse_mode,
            parsed_envelope=parsed_envelope,
            full_response_text=raw_text,
        )
        overall_ok = compute_overall_quality(gates)

        if not overall_ok:
            print(f"[INFO] {sample_id}: still fails gates after re-gate (e.g. {[k for k, v in gates.items() if not v.passed]})")
            continue

        sample = JoinedSample(
            id=sample_id,
            prompt_plan=plan,
            draft=draft,
            critique=critique_by_id.get(sample_id),
        )
        teacher_metadata = r.get("teacher") or {}
        accepted_record = build_accepted_record(
            sample=sample,
            refined_envelope=parsed_envelope,
            teacher_metadata=teacher_metadata,
            gates=gates,
            overall_ok=True,
        )
        now_accept.append(accepted_record)

    if not now_accept:
        print("[INFO] No samples passed re-gating; nothing to append.")
        return 0

    print(f"[INFO] Re-gate passed for {len(now_accept)} sample(s); appending to dataset.")

    if args.dry_run:
        for rec in now_accept:
            print(f"  Would append: {rec['id']}")
        print("[INFO] Dry run: no files written.")
        return 0

    # Append to dataset
    with open(dataset_path, "a", encoding="utf-8") as f:
        for rec in now_accept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] Appended {len(now_accept)} record(s) to {dataset_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
