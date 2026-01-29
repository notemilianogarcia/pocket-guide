"""
Quality assurance and diversity analysis for final training dataset.

This module provides tooling to:
1. Validate schema compliance across the dataset
2. Compute diversity statistics across multiple dimensions
3. Enforce quality gates (minimum schema validity rate)
4. Generate seeded spot-check samples for manual review

Design principles:
- Stream-safe JSONL processing (no memory explosion on large datasets)
- Deterministic seeded sampling for reproducibility
- Clear error messages when gates fail
- Human-friendly markdown output for manual review
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from pocketguide.eval.parsing import parse_and_validate


def load_dataset_stream(dataset_path: Path) -> List[Dict[str, Any]]:
    """
    Load dataset from JSONL file.
    
    Uses streaming for memory efficiency, but materializes to list
    for multiple passes (stats + sampling).
    
    Args:
        dataset_path: Path to dataset JSONL file
    
    Returns:
        List of dataset records
    """
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSON at line {line_num}: {e}", file=sys.stderr)
                continue
    
    return records


def validate_record_schema(record: Dict[str, Any]) -> Tuple[bool, str | None]:
    """
    Validate a dataset record's response against envelope + payload schemas.
    
    Args:
        record: Dataset record with 'response' field
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for required top-level fields
    if "response" not in record:
        return False, "Missing 'response' field"
    
    response = record["response"]
    
    # Serialize response back to JSON for parsing
    try:
        response_json = json.dumps(response)
    except (TypeError, ValueError) as e:
        return False, f"Response not JSON-serializable: {e}"
    
    # Use parse_and_validate to check schemas
    result = parse_and_validate(response_json, strict_json=True)
    
    if not result.success:
        error_msg = result.error.message if result.error else "Unknown validation error"
        return False, error_msg
    
    return True, None


def compute_distributions(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Compute distribution counts across multiple dimensions.
    
    Args:
        records: List of dataset records
    
    Returns:
        Dict mapping dimension name to counts
    """
    distributions = {
        "payload_type": Counter(),
        "category": Counter(),
        "difficulty": Counter(),
        "region_tag": Counter()  # Individual tags counted
    }
    
    for record in records:
        # Payload type
        payload_type = record.get("payload_type", "unknown")
        distributions["payload_type"][payload_type] += 1
        
        # Category
        category = record.get("category", "unknown")
        distributions["category"][category] += 1
        
        # Difficulty
        difficulty = record.get("difficulty", "unknown")
        distributions["difficulty"][difficulty] += 1
        
        # Region tags (each tag counted separately)
        region_tags = record.get("region_tags", [])
        for tag in region_tags:
            distributions["region_tag"][tag] += 1
    
    return distributions


def compute_percentages(distributions: Dict[str, Dict[str, int]], total: int) -> Dict[str, Dict[str, float]]:
    """
    Convert count distributions to percentages.
    
    Args:
        distributions: Count distributions from compute_distributions
        total: Total number of records
    
    Returns:
        Dict mapping dimension to percentages
    """
    percentages = {}
    
    for dimension, counts in distributions.items():
        percentages[dimension] = {
            key: round((count / total * 100), 2) if total > 0 else 0.0
            for key, count in counts.items()
        }
    
    return percentages


def compute_coverage_metrics(distributions: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """
    Compute coverage metrics: unique values, min/max shares.
    
    Args:
        distributions: Count distributions from compute_distributions
    
    Returns:
        Coverage metrics dict
    """
    coverage = {}
    
    for dimension, counts in distributions.items():
        if not counts:
            coverage[dimension] = {
                "unique_count": 0,
                "min_count": 0,
                "max_count": 0,
                "min_key": None,
                "max_key": None
            }
            continue
        
        min_count = min(counts.values())
        max_count = max(counts.values())
        min_key = min(counts, key=counts.get)
        max_key = max(counts, key=counts.get)
        
        coverage[dimension] = {
            "unique_count": len(counts),
            "min_count": min_count,
            "max_count": max_count,
            "min_key": min_key,
            "max_key": max_key
        }
    
    return coverage


def check_drift(
    distributions: Dict[str, Dict[str, int]],
    spec_path: Path | None
) -> Dict[str, Any] | None:
    """
    Compare observed distributions vs intended spec.
    
    Args:
        distributions: Observed count distributions
        spec_path: Path to dataset spec YAML (optional)
    
    Returns:
        Drift metrics dict or None if spec not available
    """
    if not spec_path or not spec_path.exists():
        return None
    
    try:
        with open(spec_path, "r") as f:
            spec = yaml.safe_load(f)
    except Exception as e:
        print(f"[WARN] Could not load spec for drift check: {e}", file=sys.stderr)
        return None
    
    drift = {}
    
    # Compare category distribution
    if "categories" in spec:
        intended_categories = {cat["name"]: cat["count"] for cat in spec["categories"]}
        observed_categories = distributions["category"]
        
        category_drift = {}
        for cat_name, intended_count in intended_categories.items():
            observed_count = observed_categories.get(cat_name, 0)
            delta = observed_count - intended_count
            category_drift[cat_name] = {
                "intended": intended_count,
                "observed": observed_count,
                "delta": delta
            }
        
        drift["category"] = category_drift
    
    return drift if drift else None


def sample_spotcheck(
    records: List[Dict[str, Any]],
    sample_n: int,
    seed: int
) -> List[Dict[str, Any]]:
    """
    Deterministically sample records for spot-checking.
    
    Args:
        records: All dataset records
        sample_n: Number of samples to draw
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled records
    """
    if sample_n >= len(records):
        # Return all if sample size >= dataset size
        return records[:]
    
    # Seed for deterministic sampling
    rng = random.Random(seed)
    
    # Sample without replacement
    sampled = rng.sample(records, sample_n)
    
    return sampled


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def write_spotcheck_markdown(
    samples: List[Dict[str, Any]],
    output_path: Path,
    seed: int
) -> None:
    """
    Write spot-check samples to human-friendly markdown.
    
    Args:
        samples: Sampled records
        output_path: Path to output markdown file
        seed: Random seed used for sampling
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Dataset Spot-Check Report\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).isoformat()}Z\n")
        f.write(f"**Seed:** {seed}\n")
        f.write(f"**Sample Size:** {len(samples)}\n\n")
        f.write("---\n\n")
        
        for i, record in enumerate(samples, 1):
            f.write(f"## Sample {i}: {record.get('id', 'unknown')}\n\n")
            
            # Metadata
            f.write(f"**Category:** {record.get('category', 'N/A')}  \n")
            f.write(f"**Difficulty:** {record.get('difficulty', 'N/A')}  \n")
            f.write(f"**Region Tags:** {', '.join(record.get('region_tags', []))}  \n")
            f.write(f"**Payload Type:** {record.get('payload_type', 'N/A')}  \n\n")
            
            # Prompt
            prompt = record.get("prompt", "N/A")
            f.write(f"**Prompt:**\n> {truncate_text(prompt, 300)}\n\n")
            
            # Response envelope
            response = record.get("response", {})
            
            f.write(f"**Summary:**\n> {response.get('summary', 'N/A')}\n\n")
            
            # Assumptions
            assumptions = response.get("assumptions", [])
            if assumptions:
                f.write(f"**Assumptions:**\n")
                for assumption in assumptions[:5]:  # Top 5
                    f.write(f"- {truncate_text(assumption, 150)}\n")
                f.write("\n")
            
            # Uncertainty
            uncertainty = response.get("uncertainty_notes", "")
            if uncertainty:
                f.write(f"**Uncertainty Notes:**\n> {truncate_text(uncertainty, 200)}\n\n")
            
            # Verification steps
            verification_steps = response.get("verification_steps", [])
            if verification_steps:
                f.write(f"**Verification Steps:**\n")
                for step in verification_steps[:5]:  # Top 5
                    f.write(f"- {truncate_text(step, 150)}\n")
                f.write("\n")
            
            # Payload preview
            payload = response.get("payload", {})
            if payload:
                payload_keys = list(payload.keys())[:10]  # Top 10 keys
                f.write(f"**Payload Preview:**\n")
                f.write(f"- Keys: {', '.join(payload_keys)}\n")
                
                # Show a small snippet of first value
                if payload_keys:
                    first_key = payload_keys[0]
                    first_value = payload[first_key]
                    value_str = str(first_value)
                    f.write(f"- `{first_key}`: {truncate_text(value_str, 100)}\n")
                f.write("\n")
            
            f.write("---\n\n")


def write_spotcheck_jsonl(
    samples: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Write spot-check samples to JSONL for programmatic analysis.
    
    Args:
        samples: Sampled records
        output_path: Path to output JSONL file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_qa_report(
    output_path: Path,
    total_samples: int,
    schema_valid_count: int,
    invalid_example_ids: List[str],
    distributions: Dict[str, Dict[str, int]],
    percentages: Dict[str, Dict[str, float]],
    coverage: Dict[str, Any],
    drift: Dict[str, Any] | None,
    sampled_ids: List[str],
    seed: int,
    gate_threshold: float,
    gate_passed: bool
) -> None:
    """
    Write comprehensive QA report to JSON.
    
    Args:
        output_path: Path to output JSON file
        total_samples: Total number of records
        schema_valid_count: Number of schema-valid records
        invalid_example_ids: List of invalid record IDs (capped)
        distributions: Count distributions
        percentages: Percentage distributions
        coverage: Coverage metrics
        drift: Drift analysis (optional)
        sampled_ids: List of sampled IDs for spot-check
        seed: Random seed used
        gate_threshold: Minimum schema validity threshold
        gate_passed: Whether gate passed
    """
    schema_valid_rate = schema_valid_count / total_samples if total_samples > 0 else 0.0
    
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_samples": total_samples,
        "schema_validation": {
            "valid_count": schema_valid_count,
            "invalid_count": total_samples - schema_valid_count,
            "valid_rate": round(schema_valid_rate, 4),
            "invalid_example_ids": invalid_example_ids
        },
        "quality_gate": {
            "threshold": gate_threshold,
            "passed": gate_passed,
            "message": f"Schema validity rate {schema_valid_rate:.2%} {'≥' if gate_passed else '<'} threshold {gate_threshold:.2%}"
        },
        "distributions": {
            "counts": distributions,
            "percentages": percentages
        },
        "coverage": coverage,
        "drift": drift,
        "spotcheck": {
            "sample_size": len(sampled_ids),
            "sampled_ids": sampled_ids
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main():
    """CLI entry point for dataset QA."""
    parser = argparse.ArgumentParser(
        description="Quality assurance and diversity analysis for training dataset"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/dataset_v1.jsonl"),
        help="Path to dataset JSONL file"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for QA report"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for spot-check sampling"
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=30,
        help="Number of samples for spot-check"
    )
    parser.add_argument(
        "--min_schema_valid_rate",
        type=float,
        default=0.85,
        help="Minimum schema validity rate (0.0-1.0)"
    )
    parser.add_argument(
        "--spec",
        type=Path,
        help="Path to dataset spec YAML for drift analysis (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.dataset.exists():
        print(f"[ERROR] Dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loading dataset: {args.dataset}")
    records = load_dataset_stream(args.dataset)
    total_samples = len(records)
    
    if total_samples == 0:
        print("[ERROR] Dataset is empty", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loaded {total_samples} records")
    
    # Validate schema compliance
    print(f"[INFO] Validating schema compliance...")
    schema_valid_count = 0
    invalid_example_ids = []
    
    for record in records:
        is_valid, error_msg = validate_record_schema(record)
        if is_valid:
            schema_valid_count += 1
        else:
            record_id = record.get("id", "unknown")
            if len(invalid_example_ids) < 50:  # Cap at 50
                invalid_example_ids.append(f"{record_id}: {error_msg}")
    
    schema_valid_rate = schema_valid_count / total_samples
    print(f"[INFO] Schema validity: {schema_valid_count}/{total_samples} ({schema_valid_rate:.2%})")
    
    # Compute distributions
    print(f"[INFO] Computing diversity statistics...")
    distributions = compute_distributions(records)
    percentages = compute_percentages(distributions, total_samples)
    coverage = compute_coverage_metrics(distributions)
    
    # Check drift (optional)
    drift = None
    if args.spec:
        print(f"[INFO] Checking drift against spec: {args.spec}")
        drift = check_drift(distributions, args.spec)
    
    # Generate spot-check samples
    print(f"[INFO] Sampling {args.sample_n} records for spot-check (seed={args.seed})...")
    samples = sample_spotcheck(records, args.sample_n, args.seed)
    sampled_ids = [s.get("id", "unknown") for s in samples]
    
    # Write outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    qa_report_path = args.out_dir / "dataset_v1_qa.json"
    print(f"[INFO] Writing QA report: {qa_report_path}")
    
    gate_passed = schema_valid_rate >= args.min_schema_valid_rate
    
    write_qa_report(
        output_path=qa_report_path,
        total_samples=total_samples,
        schema_valid_count=schema_valid_count,
        invalid_example_ids=invalid_example_ids,
        distributions=distributions,
        percentages=percentages,
        coverage=coverage,
        drift=drift,
        sampled_ids=sampled_ids,
        seed=args.seed,
        gate_threshold=args.min_schema_valid_rate,
        gate_passed=gate_passed
    )
    
    # Write spot-check outputs
    spotcheck_md_path = args.out_dir / f"spotcheck_v1_seed_{args.seed}.md"
    spotcheck_jsonl_path = args.out_dir / f"spotcheck_v1_seed_{args.seed}.jsonl"
    
    print(f"[INFO] Writing spot-check markdown: {spotcheck_md_path}")
    write_spotcheck_markdown(samples, spotcheck_md_path, args.seed)
    
    print(f"[INFO] Writing spot-check JSONL: {spotcheck_jsonl_path}")
    write_spotcheck_jsonl(samples, spotcheck_jsonl_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"QA SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Schema valid: {schema_valid_count} ({schema_valid_rate:.2%})")
    print(f"Gate threshold: {args.min_schema_valid_rate:.2%}")
    print(f"Gate status: {'✅ PASS' if gate_passed else '❌ FAIL'}")
    
    if not gate_passed:
        print(f"\n[ERROR] Schema validity gate FAILED!")
        print(f"[ERROR] Expected >= {args.min_schema_valid_rate:.2%}, got {schema_valid_rate:.2%}")
        print(f"[ERROR] Inspect rejected samples and validation errors.")
        print(f"[ERROR] Check {qa_report_path} for invalid_example_ids.")
        sys.exit(1)
    
    print(f"\n✅ QA complete! All gates passed.")
    print(f"📊 Report: {qa_report_path}")
    print(f"🔍 Spot-check: {spotcheck_md_path}")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
