"""
Split dataset v1 with leakage guardrails.

This module creates deterministic train/val/test splits while preventing
exact duplicate and near-duplicate leakage across splits using group-based splitting.

Key concepts:
- Leakage groups: records that are exact or near-duplicates must stay together
- Union-find (DSU): efficiently merge records into connected component groups
- Group-based splitting: assign entire groups to splits, not individual records
- Held-out benchmark: extract prompts from test split for evaluation

Usage:
    python -m pocketguide.data_generation.split_dataset_v1 \
        --in_path data/processed/dataset_v1_clean.jsonl \
        --out_dir data/processed/splits/v1 \
        --seed 42 \
        --train_frac 0.8 --val_frac 0.1 --test_frac 0.1 \
        --sim_threshold 0.75
"""

import argparse
import hashlib
import json
import random
import re
import string
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional


# Reuse constants from qa_pipeline_v1
SHINGLE_SIZE = 3
BUCKET_MODULO = 1000


class UnionFind:
    """Union-Find (Disjoint Set Union) data structure for grouping records."""
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x: str) -> str:
        """Find root of element x with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        
        return self.parent[x]
    
    def union(self, x: str, y: str) -> None:
        """Union two sets by rank."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def get_groups(self) -> Dict[str, List[str]]:
        """Get all groups as dict mapping group_id -> list of member ids."""
        groups = defaultdict(list)
        for element in list(self.parent.keys()):
            root = self.find(element)
            groups[root].append(element)
        return dict(groups)


def normalize_text(text: str) -> str:
    """Normalize text for fingerprinting (reused from qa_pipeline_v1)."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def compute_fingerprint(record: Dict[str, Any]) -> str:
    """
    Compute SHA256 fingerprint for exact duplicate detection.
    
    Reused from qa_pipeline_v1.py - combines payload_type, prompt, and response.summary.
    """
    payload_type = record.get("payload_type", "unknown")
    prompt = record.get("prompt", "")
    response = record.get("response", {})
    
    if isinstance(response, dict):
        summary = response.get("summary", "")
    else:
        summary = ""
    
    # Normalize and concatenate
    canonical = f"{payload_type}||{normalize_text(prompt)}||{normalize_text(summary)}"
    
    # Hash
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words (reused from qa_pipeline_v1)."""
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split on whitespace and filter empty
    tokens = [tok.strip().lower() for tok in text.split() if tok.strip()]
    return tokens


def generate_shingles(tokens: List[str], shingle_size: int = 3) -> Set[str]:
    """Generate k-token shingles (reused from qa_pipeline_v1)."""
    if len(tokens) < shingle_size:
        return set()
    
    shingles = set()
    for i in range(len(tokens) - shingle_size + 1):
        shingle = ' '.join(tokens[i:i + shingle_size])
        shingles.add(shingle)
    
    return shingles


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity (reused from qa_pipeline_v1)."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def get_bucket_id(shingles: Set[str], modulo: int = 1000) -> int:
    """Get bucket ID for candidate filtering (reused from qa_pipeline_v1)."""
    if not shingles:
        return 0
    
    # Hash first shingle and mod
    first_shingle = sorted(shingles)[0]
    hash_val = int(hashlib.sha256(first_shingle.encode()).hexdigest(), 16)
    return hash_val % modulo


def build_near_dup_text(record: Dict[str, Any]) -> str:
    """Build text for near-duplicate detection (reused from qa_pipeline_v1)."""
    payload_type = record.get("payload_type", "")
    prompt = record.get("prompt", "")
    response = record.get("response", {})
    
    if isinstance(response, dict):
        summary = response.get("summary", "")
    else:
        summary = ""
    
    return f"{payload_type} {prompt} {summary}"


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
    return records


def build_leakage_groups(
    records: List[Dict[str, Any]],
    sim_threshold: float = 0.75,
    shingle_size: int = SHINGLE_SIZE,
    bucket_modulo: int = BUCKET_MODULO
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Build leakage groups using exact and near-duplicate detection.
    
    Returns:
        Tuple of (record_id -> group_id, group_id -> list of record_ids)
    """
    dsu = UnionFind()
    
    # Step 1: Group by exact fingerprint
    fingerprint_to_ids = defaultdict(list)
    for record in records:
        rec_id = record["id"]
        fp = compute_fingerprint(record)
        fingerprint_to_ids[fp].append(rec_id)
        
        # Initialize in DSU
        dsu.find(rec_id)
    
    # Union all records with same fingerprint
    for fp, ids in fingerprint_to_ids.items():
        if len(ids) > 1:
            for i in range(1, len(ids)):
                dsu.union(ids[0], ids[i])
    
    # Step 2: Near-duplicate detection via shingling
    # Build shingles and bucket for each record
    record_shingles = {}
    bucket_to_ids = defaultdict(list)
    
    for record in records:
        rec_id = record["id"]
        text = build_near_dup_text(record)
        tokens = tokenize_text(text)
        shingles = generate_shingles(tokens, shingle_size)
        
        record_shingles[rec_id] = shingles
        
        bucket = get_bucket_id(shingles, bucket_modulo)
        bucket_to_ids[bucket].append(rec_id)
    
    # Step 3: Check similarity within buckets
    for bucket, ids in bucket_to_ids.items():
        # Compare all pairs in bucket
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                
                # Skip if already in same group
                if dsu.find(id1) == dsu.find(id2):
                    continue
                
                # Compute similarity
                shingles1 = record_shingles[id1]
                shingles2 = record_shingles[id2]
                sim = jaccard_similarity(shingles1, shingles2)
                
                # Union if similar
                if sim >= sim_threshold:
                    dsu.union(id1, id2)
    
    # Step 4: Extract final groups
    groups = dsu.get_groups()
    
    # Build record_id -> group_id mapping
    record_to_group = {}
    for group_id, member_ids in groups.items():
        for member_id in member_ids:
            record_to_group[member_id] = group_id
    
    return record_to_group, groups


def assign_groups_to_splits(
    groups: Dict[str, List[str]],
    records: List[Dict[str, Any]],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42
) -> Dict[str, str]:
    """
    Assign groups to splits to match target fractions.
    
    Uses greedy assignment to balance split sizes by record count.
    
    Returns:
        Dict mapping record_id -> split_name ('train', 'val', 'test')
    """
    # Validate fractions
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 0.01:
        raise ValueError(f"Split fractions must sum to ~1.0, got {total_frac}")
    
    # Build list of (group_id, group_size, member_ids)
    group_list = []
    for group_id, member_ids in groups.items():
        group_list.append((group_id, len(member_ids), member_ids))
    
    # Sort by group_id for stability, then shuffle with seed
    group_list.sort(key=lambda x: x[0])
    rng = random.Random(seed)
    rng.shuffle(group_list)
    
    # Target counts
    total_records = len(records)
    target_train = int(total_records * train_frac)
    target_val = int(total_records * val_frac)
    target_test = int(total_records * test_frac)
    
    # Greedy assignment
    splits = {'train': [], 'val': [], 'test': []}
    counts = {'train': 0, 'val': 0, 'test': 0}
    targets = {'train': target_train, 'val': target_val, 'test': target_test}
    
    for group_id, group_size, member_ids in group_list:
        # Find split that's furthest from its target (as fraction)
        best_split = None
        best_deficit = -1
        
        for split_name in ['train', 'val', 'test']:
            current = counts[split_name]
            target = targets[split_name]
            if target == 0:
                deficit = 0
            else:
                deficit = (target - current) / target
            
            if deficit > best_deficit:
                best_deficit = deficit
                best_split = split_name
        
        # Assign group to best split
        splits[best_split].extend(member_ids)
        counts[best_split] += group_size
    
    # Build record_id -> split mapping
    record_to_split = {}
    for split_name, rec_ids in splits.items():
        for rec_id in rec_ids:
            record_to_split[rec_id] = split_name
    
    return record_to_split


def write_splits(
    records: List[Dict[str, Any]],
    record_to_split: Dict[str, str],
    out_dir: Path
) -> Dict[str, int]:
    """
    Write train/val/test JSONL files.
    
    Returns:
        Dict of split_name -> count
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Group records by split
    splits = defaultdict(list)
    for record in records:
        rec_id = record["id"]
        split_name = record_to_split.get(rec_id)
        if split_name:
            splits[split_name].append(record)
    
    # Write each split
    counts = {}
    for split_name in ['train', 'val', 'test']:
        split_records = splits[split_name]
        counts[split_name] = len(split_records)
        
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, 'w', encoding='utf-8') as f:
            for record in split_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    return counts


def write_benchmark_prompts(
    records: List[Dict[str, Any]],
    record_to_split: Dict[str, str],
    benchmark_dir: Path
) -> int:
    """
    Write held-out benchmark prompts from test split.
    
    Returns:
        Number of prompts written
    """
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract test records
    test_records = [r for r in records if record_to_split.get(r["id"]) == "test"]
    
    # Write prompts (no responses)
    prompt_path = benchmark_dir / "prompts_test.jsonl"
    with open(prompt_path, 'w', encoding='utf-8') as f:
        for record in test_records:
            prompt_record = {
                "id": record["id"],
                "payload_type": record.get("payload_type"),
                "category": record.get("category"),
                "difficulty": record.get("difficulty"),
                "region_tags": record.get("region_tags", []),
                "prompt": record.get("prompt"),
            }
            f.write(json.dumps(prompt_record, ensure_ascii=False) + '\n')
    
    return len(test_records)


def write_benchmark_readme(
    benchmark_dir: Path,
    params: Dict[str, Any],
    prompt_count: int
) -> None:
    """Write README for benchmark suite."""
    readme_path = benchmark_dir / "README.md"
    
    lines = [
        "# Held-Out Benchmark v1",
        "",
        "This directory contains held-out test prompts for evaluating travel guide generation models.",
        "",
        "## Overview",
        "",
        f"- **Total prompts:** {prompt_count}",
        f"- **Generation date:** {params['timestamp']}",
        f"- **Random seed:** {params['seed']}",
        f"- **Train/Val/Test fractions:** {params['train_frac']:.1%} / {params['val_frac']:.1%} / {params['test_frac']:.1%}",
        f"- **Similarity threshold:** {params['sim_threshold']}",
        "",
        "## Files",
        "",
        "- **prompts_test.jsonl**: Test prompts without responses (for evaluation)",
        "",
        "## Leakage Guardrails",
        "",
        "All prompts in this benchmark are:",
        "- Held out from training data (test split only)",
        "- Protected from near-duplicate leakage via group-based splitting",
        "- Validated to have no exact duplicates in train/val splits",
        "",
        "## Usage",
        "",
        "To evaluate a model on this benchmark:",
        "",
        "```bash",
        "python -m pocketguide.eval.run_benchmark \\",
        "  --prompts data/benchmarks/v1/prompts_test.jsonl \\",
        "  --output results/benchmark_v1_results.jsonl",
        "```",
        "",
        "Each prompt includes:",
        "- `id`: Unique identifier",
        "- `payload_type`: Expected response format (checklist, guide, etc.)",
        "- `category`: Topic category (visa, passport, etc.)",
        "- `difficulty`: Complexity level (easy, medium, hard)",
        "- `region_tags`: Relevant geographic regions",
        "- `prompt`: The question to answer",
        "",
        "Responses should be generated in the format specified by `payload_type`.",
    ]
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def validate_splits(
    records: List[Dict[str, Any]],
    record_to_split: Dict[str, str],
    record_to_group: Dict[str, str]
) -> Dict[str, Any]:
    """
    Validate split quality and check for leakage.
    
    Returns:
        Dict with validation results
    """
    # Check 1: No group spans multiple splits
    group_to_splits = defaultdict(set)
    for rec_id, group_id in record_to_group.items():
        split_name = record_to_split.get(rec_id)
        if split_name:
            group_to_splits[group_id].add(split_name)
    
    groups_spanning_splits = [
        group_id for group_id, splits in group_to_splits.items()
        if len(splits) > 1
    ]
    
    # Check 2: No exact fingerprint spans multiple splits
    fingerprint_to_splits = defaultdict(set)
    for record in records:
        rec_id = record["id"]
        split_name = record_to_split.get(rec_id)
        if split_name:
            fp = compute_fingerprint(record)
            fingerprint_to_splits[fp].add(split_name)
    
    fingerprints_spanning_splits = [
        fp for fp, splits in fingerprint_to_splits.items()
        if len(splits) > 1
    ]
    
    return {
        "groups_spanning_splits": len(groups_spanning_splits),
        "fingerprints_spanning_splits": len(fingerprints_spanning_splits),
        "leakage_free": len(groups_spanning_splits) == 0 and len(fingerprints_spanning_splits) == 0
    }


def compute_split_distributions(
    records: List[Dict[str, Any]],
    record_to_split: Dict[str, str]
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Compute distributions per split."""
    distributions = {
        'train': defaultdict(lambda: defaultdict(int)),
        'val': defaultdict(lambda: defaultdict(int)),
        'test': defaultdict(lambda: defaultdict(int))
    }
    
    for record in records:
        rec_id = record["id"]
        split_name = record_to_split.get(rec_id)
        if not split_name:
            continue
        
        # Category
        category = record.get("category", "unknown")
        distributions[split_name]['category'][category] += 1
        
        # Difficulty
        difficulty = record.get("difficulty", "medium")
        distributions[split_name]['difficulty'][difficulty] += 1
        
        # Payload type
        payload_type = record.get("payload_type", "unknown")
        distributions[split_name]['payload_type'][payload_type] += 1
    
    # Convert to regular dicts
    result = {}
    for split_name in ['train', 'val', 'test']:
        result[split_name] = {
            'category': dict(distributions[split_name]['category']),
            'difficulty': dict(distributions[split_name]['difficulty']),
            'payload_type': dict(distributions[split_name]['payload_type'])
        }
    
    return result


def compute_group_stats(groups: Dict[str, List[str]]) -> Dict[str, Any]:
    """Compute statistics about group sizes."""
    sizes = [len(members) for members in groups.values()]
    sizes.sort()
    
    if not sizes:
        return {
            "num_groups": 0,
            "min_size": 0,
            "median_size": 0,
            "max_size": 0,
            "singleton_groups": 0
        }
    
    return {
        "num_groups": len(sizes),
        "min_size": sizes[0],
        "median_size": sizes[len(sizes) // 2],
        "max_size": sizes[-1],
        "singleton_groups": sum(1 for s in sizes if s == 1)
    }


def write_manifest(
    out_dir: Path,
    params: Dict[str, Any],
    split_counts: Dict[str, int],
    prompt_count: int
) -> None:
    """Write splits manifest JSON."""
    manifest = {
        "metadata": {
            "timestamp": params["timestamp"],
            "input_path": str(params["in_path"]),
            "seed": params["seed"]
        },
        "parameters": {
            "train_frac": params["train_frac"],
            "val_frac": params["val_frac"],
            "test_frac": params["test_frac"],
            "sim_threshold": params["sim_threshold"],
            "shingle_size": params["shingle_size"],
            "bucket_modulo": params["bucket_modulo"]
        },
        "outputs": {
            "train_path": str(out_dir / "train.jsonl"),
            "val_path": str(out_dir / "val.jsonl"),
            "test_path": str(out_dir / "test.jsonl"),
            "benchmark_prompts_path": "data/benchmarks/v1/prompts_test.jsonl"
        },
        "counts": {
            "train": split_counts["train"],
            "val": split_counts["val"],
            "test": split_counts["test"],
            "total": sum(split_counts.values()),
            "benchmark_prompts": prompt_count
        }
    }
    
    manifest_path = out_dir / "splits_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def write_stats(
    out_dir: Path,
    params: Dict[str, Any],
    split_counts: Dict[str, int],
    groups: Dict[str, List[str]],
    validation_results: Dict[str, Any],
    distributions: Dict[str, Dict[str, Dict[str, int]]],
    prompt_count: int
) -> None:
    """Write splits statistics JSON."""
    group_stats = compute_group_stats(groups)
    
    stats = {
        "summary": {
            "total_records": sum(split_counts.values()),
            "train_count": split_counts["train"],
            "val_count": split_counts["val"],
            "test_count": split_counts["test"],
            "benchmark_prompts": prompt_count
        },
        "groups": group_stats,
        "leakage_validation": validation_results,
        "distributions": distributions,
        "parameters": {
            "seed": params["seed"],
            "train_frac": params["train_frac"],
            "val_frac": params["val_frac"],
            "test_frac": params["test_frac"],
            "sim_threshold": params["sim_threshold"],
            "shingle_size": params["shingle_size"],
            "bucket_modulo": params["bucket_modulo"]
        }
    }
    
    stats_path = out_dir / "splits_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Split dataset v1 with leakage guardrails"
    )
    parser.add_argument(
        "--in_path",
        type=Path,
        required=True,
        help="Input dataset JSONL path"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/processed/splits/v1"),
        help="Output directory for splits"
    )
    parser.add_argument(
        "--benchmark_dir",
        type=Path,
        default=Path("data/benchmarks/v1"),
        help="Output directory for benchmark prompts"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism"
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Train split fraction (default 0.8)"
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Validation split fraction (default 0.1)"
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.1,
        help="Test split fraction (default 0.1)"
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.75,
        help="Jaccard similarity threshold for near-duplicates (default 0.75)"
    )
    parser.add_argument(
        "--shingle_size",
        type=int,
        default=SHINGLE_SIZE,
        help=f"Token shingle size (default {SHINGLE_SIZE})"
    )
    parser.add_argument(
        "--bucket_modulo",
        type=int,
        default=BUCKET_MODULO,
        help=f"Bucket modulo for candidate filtering (default {BUCKET_MODULO})"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Dataset Splitting with Leakage Guardrails")
    print("="*70)
    
    # Validate fractions
    total_frac = args.train_frac + args.val_frac + args.test_frac
    if abs(total_frac - 1.0) > 0.01:
        raise ValueError(f"Split fractions must sum to ~1.0, got {total_frac}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.in_path}...")
    records = load_dataset(args.in_path)
    print(f"  Loaded {len(records)} records")
    
    # Build leakage groups
    print(f"\nBuilding leakage groups (sim_threshold={args.sim_threshold})...")
    record_to_group, groups = build_leakage_groups(
        records,
        sim_threshold=args.sim_threshold,
        shingle_size=args.shingle_size,
        bucket_modulo=args.bucket_modulo
    )
    print(f"  Created {len(groups)} groups")
    
    group_stats = compute_group_stats(groups)
    print(f"  Group sizes: min={group_stats['min_size']}, "
          f"median={group_stats['median_size']}, max={group_stats['max_size']}")
    print(f"  Singleton groups: {group_stats['singleton_groups']}")
    
    # Assign groups to splits
    print(f"\nAssigning groups to splits "
          f"(train={args.train_frac:.1%}, val={args.val_frac:.1%}, test={args.test_frac:.1%})...")
    record_to_split = assign_groups_to_splits(
        groups,
        records,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed
    )
    
    # Write splits
    print(f"\nWriting splits to {args.out_dir}...")
    split_counts = write_splits(records, record_to_split, args.out_dir)
    print(f"  Train: {split_counts['train']} records")
    print(f"  Val: {split_counts['val']} records")
    print(f"  Test: {split_counts['test']} records")
    
    # Validate splits
    print("\nValidating splits for leakage...")
    validation_results = validate_splits(records, record_to_split, record_to_group)
    if validation_results["leakage_free"]:
        print("  ✓ No leakage detected")
    else:
        print(f"  ✗ WARNING: {validation_results['groups_spanning_splits']} groups span multiple splits")
        print(f"  ✗ WARNING: {validation_results['fingerprints_spanning_splits']} fingerprints span multiple splits")
    
    # Write benchmark prompts
    print(f"\nWriting benchmark prompts to {args.benchmark_dir}...")
    prompt_count = write_benchmark_prompts(records, record_to_split, args.benchmark_dir)
    print(f"  Wrote {prompt_count} test prompts")
    
    # Write benchmark README
    params = {
        "timestamp": datetime.now().isoformat(),
        "in_path": args.in_path,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "sim_threshold": args.sim_threshold,
        "shingle_size": args.shingle_size,
        "bucket_modulo": args.bucket_modulo
    }
    
    write_benchmark_readme(args.benchmark_dir, params, prompt_count)
    print(f"  Wrote README.md")
    
    # Compute distributions
    print("\nComputing split distributions...")
    distributions = compute_split_distributions(records, record_to_split)
    
    # Write manifest and stats
    print(f"\nWriting manifest and stats...")
    write_manifest(args.out_dir, params, split_counts, prompt_count)
    write_stats(args.out_dir, params, split_counts, groups, validation_results, distributions, prompt_count)
    
    print("\n" + "="*70)
    print("Splitting Complete")
    print("="*70)
    print(f"Output directory: {args.out_dir}")
    print(f"Benchmark directory: {args.benchmark_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
