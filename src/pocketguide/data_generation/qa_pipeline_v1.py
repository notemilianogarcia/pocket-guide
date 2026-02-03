"""QA Pipeline v1: Exact and near-duplicate detection with quality filters.

This module processes the generated dataset (dataset_v1.jsonl) and:
- Removes exact duplicates via content fingerprinting
- Removes near-duplicates via token shingles and Jaccard similarity
- Applies quality filters: length, vagueness, overclaim/time-sensitive
- Tracks before/after distributions
- Generates auditable QA report (markdown + JSON)

Usage:
    python -m pocketguide.data_generation.qa_pipeline_v1 \
        --in_path data/processed/dataset_v1.jsonl \
        --out_clean data/processed/dataset_v1_clean.jsonl \
        --out_report data/processed/dataset_v1_qa_report.md \
        --out_summary data/processed/dataset_v1_qa_summary.json \
        --seed 42 \
        --near_dup_threshold 0.85 \
        --min_words 100 \
        --max_words 800
"""

import argparse
import hashlib
import json
import re
import string
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pocketguide.data_generation.quality_filters import (
    apply_all_filters,
    check_length,
    check_vagueness,
    check_overclaim,
    MIN_WORDS_DEFAULT,
    MAX_WORDS_DEFAULT,
    VAGUE_PHRASE_THRESHOLD,
    CONCRETE_ACTION_THRESHOLD,
    OVERCLAIM_REJECTION_THRESHOLD,
)

from pocketguide.data_generation.balancing import (
    apply_balancing,
    compute_bucket_stats,
)

# Constants for near-duplicate detection
SHINGLE_SIZE = 3  # Token shingle size
BUCKET_MODULO = 1000  # Number of buckets for candidate filtering

# Constants for quality filters (re-exported for convenience)
QF_MIN_WORDS = MIN_WORDS_DEFAULT
QF_MAX_WORDS = MAX_WORDS_DEFAULT


def normalize_text(text: str) -> str:
    """
    Normalize text for fingerprinting.
    
    - Strip leading/trailing whitespace
    - Collapse multiple spaces/newlines to single space
    - Lowercase for case-insensitive comparison
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Strip outer whitespace
    text = text.strip()
    # Collapse all whitespace (spaces, newlines, tabs) to single space
    text = re.sub(r'\s+', ' ', text)
    # Lowercase for case-insensitive
    text = text.lower()
    return text


def compute_fingerprint(record: Dict[str, Any]) -> str:
    """
    Compute exact-dedup fingerprint for a record.
    
    Fingerprint components:
    - payload_type (from record or response)
    - normalized prompt
    - normalized response.summary
    
    Args:
        record: Dataset record
        
    Returns:
        SHA256 hex digest fingerprint
    """
    # Extract payload_type (may be at root or in response)
    payload_type = record.get("payload_type") or record.get("response", {}).get("payload_type", "unknown")
    
    # Extract prompt
    prompt = record.get("prompt", "")
    
    # Extract response.summary
    summary = record.get("response", {}).get("summary", "")
    
    # Normalize components
    payload_type_norm = payload_type.strip().lower()
    prompt_norm = normalize_text(prompt)
    summary_norm = normalize_text(summary)
    
    # Create stable string representation
    fingerprint_str = f"{payload_type_norm}|{prompt_norm}|{summary_norm}"
    
    # Hash to fixed-length fingerprint
    fingerprint = hashlib.sha256(fingerprint_str.encode('utf-8')).hexdigest()
    
    return fingerprint


def load_dataset_stream(path: Path) -> List[Dict[str, Any]]:
    """
    Load dataset JSONL stream.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of records
    """
    records = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            records.append(record)
    
    return records


def compute_distributions(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Compute distribution counts across dimensions.
    
    Dimensions:
    - category
    - difficulty
    - payload_type
    - region_tag (individual tags)
    
    Args:
        records: List of records
        
    Returns:
        Dict mapping dimension -> value -> count
    """
    distributions = {
        "category": defaultdict(int),
        "difficulty": defaultdict(int),
        "payload_type": defaultdict(int),
        "region_tag": defaultdict(int),
    }
    
    for record in records:
        # Category
        category = record.get("category", "unknown")
        distributions["category"][category] += 1
        
        # Difficulty
        difficulty = record.get("difficulty", "unknown")
        distributions["difficulty"][difficulty] += 1
        
        # Payload type
        payload_type = record.get("payload_type") or record.get("response", {}).get("payload_type", "unknown")
        distributions["payload_type"][payload_type] += 1
        
        # Region tags (individual)
        region_tags = record.get("region_tags", [])
        if isinstance(region_tags, list):
            for tag in region_tags:
                distributions["region_tag"][tag] += 1
    
    # Convert defaultdicts to regular dicts
    return {
        dim: dict(counts) for dim, counts in distributions.items()
    }


def apply_exact_dedup(
    records: List[Dict[str, Any]], 
    seed: int
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]:
    """
    Apply exact deduplication.
    
    Strategy:
    - Keep first occurrence of each fingerprint
    - Reject subsequent duplicates with reason "exact_duplicate"
    
    Args:
        records: Input records
        seed: Random seed (for determinism, though not used in exact dedup)
        
    Returns:
        Tuple of (accepted_records, rejected_records_with_reasons)
        where rejected entry is (record, reason, metadata)
    """
    seen_fingerprints = {}  # fingerprint -> first record ID
    accepted = []
    rejected = []
    
    for record in records:
        fingerprint = compute_fingerprint(record)
        
        if fingerprint not in seen_fingerprints:
            # First occurrence - accept
            seen_fingerprints[fingerprint] = record.get("id", "unknown")
            accepted.append(record)
        else:
            # Duplicate - reject
            reason = "exact_duplicate"
            metadata = {
                "anchor_id": seen_fingerprints[fingerprint],
                "similarity": 1.0,  # Exact duplicate
            }
            rejected.append((record, reason, metadata))
    
    return accepted, rejected


def build_near_dup_text(record: Dict[str, Any]) -> str:
    """
    Build text for near-duplicate detection.
    
    Combines: payload_type + prompt + response.summary
    Normalized and deduplicated for comparison.
    
    Args:
        record: Dataset record
        
    Returns:
        Combined normalized text
    """
    payload_type = record.get("payload_type") or record.get("response", {}).get("payload_type", "")
    prompt = record.get("prompt", "")
    summary = record.get("response", {}).get("summary", "")
    
    combined = f"{payload_type} {prompt} {summary}"
    return normalize_text(combined)


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Remove punctuation and split on whitespace.
    
    Args:
        text: Input text (should be normalized)
        
    Returns:
        List of tokens
    """
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Split and filter empty tokens
    tokens = [t for t in text.split() if t]
    return tokens


def generate_shingles(tokens: List[str], shingle_size: int = SHINGLE_SIZE) -> Set[str]:
    """
    Generate token shingles.
    
    Create k-token shingles from token sequence.
    
    Args:
        tokens: List of tokens
        shingle_size: Size of shingles (default 3)
        
    Returns:
        Set of shingle strings
    """
    shingles = set()
    
    for i in range(len(tokens) - shingle_size + 1):
        shingle = " ".join(tokens[i:i + shingle_size])
        shingles.add(shingle)
    
    return shingles


def jaccard_similarity(shingles_a: Set[str], shingles_b: Set[str]) -> float:
    """
    Compute Jaccard similarity between two shingle sets.
    
    J(A, B) = |A intersect B| / |A union B|
    
    Args:
        shingles_a: First set of shingles
        shingles_b: Second set of shingles
        
    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    if not shingles_a and not shingles_b:
        return 1.0
    
    if not shingles_a or not shingles_b:
        return 0.0
    
    intersection = len(shingles_a & shingles_b)
    union = len(shingles_a | shingles_b)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def get_bucket_id(shingles: Set[str], bucket_modulo: int = BUCKET_MODULO) -> int:
    """
    Get bucket ID for candidate filtering using first shingle hash.
    
    Args:
        shingles: Set of shingles
        bucket_modulo: Number of buckets
        
    Returns:
        Bucket ID (0 to bucket_modulo - 1)
    """
    if not shingles:
        return 0
    
    # Use first shingle for bucketing (deterministic)
    first_shingle = min(shingles)  # Deterministic ordering
    bucket = int(hashlib.md5(first_shingle.encode()).hexdigest(), 16) % bucket_modulo
    return bucket


def apply_near_dedup(
    records: List[Dict[str, Any]],
    threshold: float = 0.85,
    shingle_size: int = SHINGLE_SIZE,
    bucket_modulo: int = BUCKET_MODULO
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]:
    """
    Apply near-duplicate detection using token shingles and Jaccard similarity.
    
    Strategy:
    - Build shingles for each record
    - Bucket records by first shingle hash
    - Compare only within-bucket candidates
    - Keep first occurrence, reject near-duplicates
    
    Args:
        records: Input records (should be after exact dedup)
        threshold: Jaccard similarity threshold (default 0.85)
        shingle_size: Token shingle size (default 3)
        bucket_modulo: Number of buckets (default 1000)
        
    Returns:
        Tuple of (accepted_records, rejected_with_metadata)
        where rejected entry is (record, reason, metadata_dict)
    """
    # Build shingles for all records
    record_shingles = []
    for i, record in enumerate(records):
        text = build_near_dup_text(record)
        tokens = tokenize_text(text)
        shingles = generate_shingles(tokens, shingle_size)
        bucket = get_bucket_id(shingles, bucket_modulo)
        record_shingles.append((i, record, shingles, bucket))
    
    # Group by bucket
    bucket_to_indices = defaultdict(list)
    for i, record, shingles, bucket in record_shingles:
        bucket_to_indices[bucket].append(i)
    
    # Track which records are kept/rejected
    kept_indices = set()
    rejected = []
    anchor_map = {}  # anchor_index -> similarity -> rejected indices
    
    # Process each bucket
    for bucket, indices in bucket_to_indices.items():
        # Process records in order within bucket
        for current_idx in sorted(indices):
            _, current_record, current_shingles, _ = record_shingles[current_idx]
            
            if current_idx in kept_indices:
                continue  # Already rejected
            
            # Check against existing kept records in this bucket
            is_duplicate = False
            best_anchor = None
            best_similarity = 0.0
            
            for anchor_idx in sorted(indices):
                if anchor_idx >= current_idx or anchor_idx not in kept_indices:
                    continue
                
                _, _, anchor_shingles, _ = record_shingles[anchor_idx]
                similarity = jaccard_similarity(current_shingles, anchor_shingles)
                
                if similarity >= threshold:
                    is_duplicate = True
                    best_anchor = anchor_idx
                    best_similarity = similarity
                    break  # Use first match
            
            if is_duplicate:
                anchor_record = record_shingles[best_anchor][1]
                metadata = {
                    "anchor_id": anchor_record.get("id", "unknown"),
                    "similarity": round(best_similarity, 4),
                }
                rejected.append((current_record, "near_duplicate", metadata))
            else:
                kept_indices.add(current_idx)
    
    # Build accepted list preserving order
    accepted = [records[i] for i in range(len(records)) if i in kept_indices]
    
    return accepted, rejected


def apply_quality_filters(
    records: List[Dict[str, Any]],
    min_words: int = MIN_WORDS_DEFAULT,
    max_words: int = MAX_WORDS_DEFAULT,
    vague_phrase_threshold: int = VAGUE_PHRASE_THRESHOLD,
    concrete_action_threshold: int = CONCRETE_ACTION_THRESHOLD,
    overclaim_threshold: int = OVERCLAIM_REJECTION_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]:
    """
    Apply all quality filters sequentially to records.
    
    Filters applied in order (stops at first failure):
    1. Length (too_short / too_long)
    2. Vagueness (vague_low_specificity)
    3. Overclaim (overconfident_time_sensitive)
    
    Args:
        records: Input records
        min_words: Minimum words for length check
        max_words: Maximum words for length check
        vague_phrase_threshold: Vague phrase threshold
        concrete_action_threshold: Concrete action threshold
        overclaim_threshold: Overclaim rejection threshold
    
    Returns:
        Tuple of (accepted_records, rejected_records_with_metadata)
    """
    accepted = []
    rejected = []
    
    for record in records:
        # Apply all filters, returns first failure or None
        result = apply_all_filters(
            record,
            min_words=min_words,
            max_words=max_words,
            vague_phrase_threshold=vague_phrase_threshold,
            concrete_action_threshold=concrete_action_threshold,
            overclaim_threshold=overclaim_threshold,
        )
        
        if result is None:
            # All filters passed
            accepted.append(record)
        else:
            # At least one filter failed
            metadata = result.details.copy()
            metadata["filter_name"] = result.reason_code
            rejected.append((record, result.reason_code, metadata))
    
    return accepted, rejected


def write_clean_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """
    Write clean dataset JSONL.
    
    Args:
        records: Accepted records
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_duplicate_examples(
    records: List[Dict[str, Any]],
    rejected_with_reason: List[Tuple[Dict[str, Any], str, Dict[str, Any]]],
    max_pairs: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect example duplicate pairs organized by rejection reason.
    
    Args:
        records: All input records (not used here, for compatibility)
        rejected_with_reason: List of (record, reason, metadata) tuples
        max_pairs: Maximum pairs per reason
        
    Returns:
        Dict mapping reason -> list of example pair dicts
    """
    examples_by_reason = defaultdict(list)
    
    for record, reason, metadata in rejected_with_reason:
        if len(examples_by_reason[reason]) < max_pairs:
            example = {
                "duplicate_id": record.get("id", "unknown"),
                "anchor_id": metadata.get("anchor_id", "unknown"),
                "similarity": metadata.get("similarity"),
                "prompt_snippet": record.get("prompt", "")[:80],
                "summary_snippet": record.get("response", {}).get("summary", "")[:80],
            }
            examples_by_reason[reason].append(example)
    
    return dict(examples_by_reason)


def write_qa_report_markdown(
    path: Path,
    stats: Dict[str, Any],
    before_dist: Dict[str, Dict[str, int]],
    after_dist: Dict[str, Dict[str, int]],
    rejection_reasons: Dict[str, int],
    example_pairs: Dict[str, List[Dict[str, Any]]],
    near_dup_stats: Dict[str, Any],
    balancing_stats: Dict[str, Any] = None
) -> None:
    """
    Write QA report in markdown format.
    
    Args:
        path: Output path
        stats: Statistics dict
        before_dist: Distributions before filtering
        after_dist: Distributions after filtering
        rejection_reasons: Rejection reason counts
        example_pairs: Example duplicate pairs by reason
        near_dup_stats: Near-duplicate statistics
        balancing_stats: Balancing statistics (optional)
        near_dup_stats: Near-duplicate statistics
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    
    # Header
    lines.append("# Dataset QA Report v1")
    lines.append("")
    lines.append(f"**Generated:** {stats['timestamp']}")
    lines.append(f"**Input:** {stats['input_path']}")
    lines.append(f"**Seed:** {stats['seed']}")
    lines.append("")
    
    # Summary counts
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total input:** {stats['total_in']}")
    lines.append(f"- **Accepted:** {stats['accepted']}")
    lines.append(f"- **Rejected:** {stats['rejected']}")
    lines.append("")
    
    # Rejection reasons
    lines.append("## Rejection Reasons")
    lines.append("")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- **{reason}:** {count}")
    lines.append("")
    
    # Exact dedup details
    exact_dup_count = rejection_reasons.get("exact_duplicate", 0)
    if exact_dup_count > 0:
        lines.append("## Exact Duplicates")
        lines.append("")
        lines.append(f"**Count:** {exact_dup_count}")
        lines.append(f"**Rate:** {(exact_dup_count / stats['total_in'] * 100):.2f}%")
        lines.append("")
        
        if "exact_duplicate" in example_pairs:
            lines.append("**Example pairs:**")
            lines.append("")
            for i, pair in enumerate(example_pairs["exact_duplicate"][:3], 1):
                lines.append(f"{i}. {pair['duplicate_id']} → {pair['anchor_id']}")
            lines.append("")
    
    # Near-dedup analysis
    near_dup_count = rejection_reasons.get("near_duplicate", 0)
    if near_dup_count > 0:
        lines.append("## Near-Duplicate Analysis")
        lines.append("")
        lines.append(f"**Method:** Token shingles (size {near_dup_stats.get('shingle_size', SHINGLE_SIZE)}) " +
                     f"with Jaccard similarity threshold ≥ {near_dup_stats.get('threshold', 0.85)}")
        lines.append("")
        lines.append(f"**Count:** {near_dup_count}")
        lines.append(f"**Rate:** {(near_dup_count / stats['total_in'] * 100):.2f}%")
        lines.append(f"**Near-dup groups:** {near_dup_stats.get('near_dup_groups', 'N/A')}")
        lines.append(f"**Largest group size:** {near_dup_stats.get('max_group_size', 'N/A')}")
        lines.append("")
        
        if "near_duplicate" in example_pairs and example_pairs["near_duplicate"]:
            lines.append("**Example pairs (similarity score):**")
            lines.append("")
            for i, pair in enumerate(example_pairs["near_duplicate"][:3], 1):
                similarity = pair.get("similarity", 0)
                lines.append(f"{i}. {pair['duplicate_id']} → {pair['anchor_id']} " +
                           f"(similarity: {similarity:.3f})")
            lines.append("")
    
    # Quality filters analysis
    quality_filter_counts = {
        k: v for k, v in rejection_reasons.items()
        if k in ["too_short", "too_long", "vague_low_specificity", "overconfident_time_sensitive"]
    }
    
    if quality_filter_counts:
        lines.append("## Quality Filters")
        lines.append("")
        lines.append("Applied sequentially after deduplication to remove low-signal or unsafe samples:")
        lines.append("")
        
        # Filter counts
        lines.append("### Filter Results")
        lines.append("")
        for reason, count in sorted(quality_filter_counts.items()):
            pct = (count / stats['total_in'] * 100) if stats['total_in'] > 0 else 0
            lines.append(f"- **{reason}:** {count} ({pct:.2f}%)")
        lines.append("")
        
        # Filter descriptions
        lines.append("### Filter Descriptions")
        lines.append("")
        
        too_short_count = quality_filter_counts.get("too_short", 0)
        too_long_count = quality_filter_counts.get("too_long", 0)
        if too_short_count > 0 or too_long_count > 0:
            lines.append("**Length:** Rejects records with word count outside [100, 800] words")
            lines.append("")
        
        vague_count = quality_filter_counts.get("vague_low_specificity", 0)
        if vague_count > 0:
            lines.append("**Vagueness:** Rejects records with high vague phrase count and low concrete action verbs/structure")
            lines.append("")
        
        overclaim_count = quality_filter_counts.get("overconfident_time_sensitive", 0)
        if overclaim_count > 0:
            lines.append("**Overclaim/Time-Sensitive:** Rejects records with strong certainty phrases " +
                        "(always/never/guaranteed) on time-sensitive topics (visa/fees/border) " +
                        "without verification steps or uncertainty notes")
            lines.append("")
    
    # Balancing analysis
    if balancing_stats and balancing_stats.get("total_removed", 0) > 0:
        lines.append("## Balancing Analysis")
        lines.append("")
        lines.append("Reduces dataset skew by downsampling overrepresented buckets while preserving hard and rare cases.")
        lines.append("")
        lines.append("**Bucket definition:** (category, difficulty, payload_type)")
        lines.append("")
        
        cap_mult = balancing_stats.get("cap_multiplier", 1.5)
        cap_mult_hard = balancing_stats.get("cap_multiplier_hard", 3.0)
        median_size = balancing_stats.get("median_bucket_size", 0)
        
        lines.append("### Configuration")
        lines.append("")
        lines.append(f"- **Cap multiplier (normal):** {cap_mult}x median bucket size")
        lines.append(f"- **Cap multiplier (hard):** {cap_mult_hard}x median bucket size")
        lines.append(f"- **Median bucket size:** {median_size}")
        lines.append("")
        
        # Bucket stats table
        lines.append("### Top Overrepresented Buckets (Before → After)")
        lines.append("")
        lines.append("| Bucket (Category, Difficulty, Type) | Before | After | Target Cap |")
        lines.append("|---|---|---|---|")
        
        pre_stats = balancing_stats.get("pre_balance_bucket_stats", {})
        post_stats = balancing_stats.get("post_balance_bucket_stats", {})
        bucket_caps = balancing_stats.get("bucket_caps", {})
        
        # Show top 10 buckets by removal count
        removals = []
        for bucket, pre_count in pre_stats.items():
            post_count = post_stats.get(bucket, 0)
            removed = pre_count - post_count
            if removed > 0:
                cap = bucket_caps.get(bucket, pre_count)
                removals.append((bucket, pre_count, post_count, cap, removed))
        
        removals.sort(key=lambda x: x[4], reverse=True)
        
        for bucket, pre, post, cap, removed in removals[:10]:
            cat, diff, ptype = bucket
            bucket_str = f"{cat}, {diff}, {ptype}"
            lines.append(f"| {bucket_str} | {pre} | {post} | {cap} |")
        
        lines.append("")
        
        total_removed = balancing_stats.get("total_removed", 0)
        total_in = balancing_stats.get("total_in", 0)
        removal_rate = balancing_stats.get("removal_rate", 0.0)
        
        lines.append("### Summary")
        lines.append("")
        lines.append(f"- **Total records removed:** {total_removed} ({removal_rate*100:.2f}%)")
        lines.append(f"- **Total input to balancing:** {total_in}")
        lines.append(f"- **Total buckets:** {len(pre_stats)}")
        lines.append("")
    
    # Distributions - Before
    lines.append("## Distributions: Before Filtering")
    lines.append("")
    
    for dim in ["payload_type", "category", "difficulty"]:
        lines.append(f"### {dim.replace('_', ' ').title()}")
        lines.append("")
        dist = before_dist.get(dim, {})
        total = sum(dist.values())
        for value, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"- **{value}:** {count} ({pct:.1f}%)")
        lines.append("")
    
    # Distributions - After
    lines.append("## Distributions: After Filtering")
    lines.append("")
    
    for dim in ["payload_type", "category", "difficulty"]:
        lines.append(f"### {dim.replace('_', ' ').title()}")
        lines.append("")
        dist = after_dist.get(dim, {})
        total = sum(dist.values())
        for value, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"- **{value}:** {count} ({pct:.1f}%)")
        lines.append("")
    
    # Region tags (top 10)
    lines.append("## Top Region Tags (After Filtering)")
    lines.append("")
    region_dist = after_dist.get("region_tag", {})
    top_regions = sorted(region_dist.items(), key=lambda x: x[1], reverse=True)[:10]
    total_region_mentions = sum(region_dist.values())
    for tag, count in top_regions:
        pct = (count / total_region_mentions * 100) if total_region_mentions > 0 else 0
        lines.append(f"- **{tag}:** {count} ({pct:.1f}%)")
    lines.append("")
    
    # Write file
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def write_qa_summary_json(
    path: Path,
    stats: Dict[str, Any],
    before_dist: Dict[str, Dict[str, int]],
    after_dist: Dict[str, Dict[str, int]],
    rejection_reasons: Dict[str, int],
    rejected_ids: List[str],
    example_pairs: Dict[str, List[Dict[str, Any]]],
    near_dup_stats: Dict[str, Any],
    balancing_stats: Dict[str, Any] = None,
    cap_multiplier: float = 1.5,
    cap_multiplier_hard: float = 3.0
) -> None:
    """
    Write machine-readable QA summary JSON.
    
    Args:
        path: Output path
        stats: Statistics dict
        before_dist: Distributions before filtering
        after_dist: Distributions after filtering
        rejection_reasons: Rejection reason counts
        rejected_ids: List of rejected record IDs (capped)
        example_pairs: Example duplicate pairs by reason
        near_dup_stats: Near-duplicate statistics
        balancing_stats: Balancing statistics (optional)
        cap_multiplier: Cap multiplier for normal buckets
        cap_multiplier_hard: Cap multiplier for hard buckets
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "metadata": {
            "timestamp": stats["timestamp"],
            "input_path": stats["input_path"],
            "seed": stats["seed"],
        },
        "counts": {
            "total_in": stats["total_in"],
            "accepted": stats["accepted"],
            "rejected": stats["rejected"],
            "exact_dup_count": stats.get("exact_dup_count", 0),
            "exact_dup_rate": stats.get("exact_dup_rate", 0),
            "near_dup_count": stats.get("near_dup_count", 0),
            "near_dup_rate": stats.get("near_dup_rate", 0),
            "quality_filter_count": stats.get("quality_filter_count", 0),
            "quality_filter_rate": stats.get("quality_filter_rate", 0),
        },
        "rejection_reasons": rejection_reasons,
        "near_dup_config": {
            "threshold": near_dup_stats.get("threshold"),
            "shingle_size": near_dup_stats.get("shingle_size"),
            "bucket_modulo": near_dup_stats.get("bucket_modulo"),
        },
        "near_dup_stats": {
            "near_dup_groups": near_dup_stats.get("near_dup_groups"),
            "max_group_size": near_dup_stats.get("max_group_size"),
        },
        "quality_filter_config": {
            "min_words": stats.get("qf_min_words"),
            "max_words": stats.get("qf_max_words"),
            "vague_phrase_threshold": stats.get("qf_vague_phrase_threshold"),
            "concrete_action_threshold": stats.get("qf_concrete_action_threshold"),
            "overclaim_threshold": stats.get("qf_overclaim_threshold"),
        },
        "quality_filter_stats": {
            k: v for k, v in rejection_reasons.items()
            if k in ["too_short", "too_long", "vague_low_specificity", "overconfident_time_sensitive"]
        },
        "balancing_config": {
            "cap_multiplier": cap_multiplier,
            "cap_multiplier_hard": cap_multiplier_hard,
            "median_bucket_size": balancing_stats.get("median_bucket_size") if balancing_stats else None,
        } if balancing_stats else None,
        "balancing_stats": {
            "total_removed": balancing_stats.get("total_removed", 0),
            "removal_rate": balancing_stats.get("removal_rate", 0.0),
            "total_buckets": len(balancing_stats.get("pre_balance_bucket_stats", {})) if balancing_stats else 0,
        } if balancing_stats else None,
        "distributions": {
            "before": before_dist,
            "after": after_dist,
        },
        "example_pairs": example_pairs,
        "rejected_ids_sample": rejected_ids[:100],  # Cap at 100
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QA Pipeline v1: Exact and near-duplicate detection"
    )
    parser.add_argument(
        "--in_path",
        type=Path,
        default=Path("data/processed/dataset_v1.jsonl"),
        help="Input dataset JSONL path"
    )
    parser.add_argument(
        "--out_clean",
        type=Path,
        default=Path("data/processed/dataset_v1_clean.jsonl"),
        help="Output clean dataset JSONL path"
    )
    parser.add_argument(
        "--out_report",
        type=Path,
        default=Path("data/processed/dataset_v1_qa_report.md"),
        help="Output QA report markdown path"
    )
    parser.add_argument(
        "--out_summary",
        type=Path,
        default=Path("data/processed/dataset_v1_qa_summary.json"),
        help="Output QA summary JSON path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism"
    )
    parser.add_argument(
        "--near_dup_threshold",
        type=float,
        default=0.88,
        help="Jaccard similarity threshold for near-duplicates (default 0.88; higher = fewer removed)"
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=60,
        help=f"Minimum words for length filter (default 60; use {MIN_WORDS_DEFAULT} for stricter)"
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=1200,
        help=f"Maximum words for length filter (default 1200; use {MAX_WORDS_DEFAULT} for stricter)"
    )
    parser.add_argument(
        "--vague_phrase_threshold",
        type=int,
        default=VAGUE_PHRASE_THRESHOLD,
        help=f"Minimum vague phrases to consider rejection (default {VAGUE_PHRASE_THRESHOLD})"
    )
    parser.add_argument(
        "--concrete_action_threshold",
        type=int,
        default=CONCRETE_ACTION_THRESHOLD,
        help=f"Minimum concrete actions to pass vagueness filter (default {CONCRETE_ACTION_THRESHOLD})"
    )
    parser.add_argument(
        "--cap_multiplier",
        type=float,
        default=2.0,
        help="Cap multiplier for normal difficulty buckets (default 2.0; higher keeps more)"
    )
    parser.add_argument(
        "--cap_multiplier_hard",
        type=float,
        default=3.0,
        help="Cap multiplier for hard difficulty buckets (default 3.0)"
    )
    
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.in_path}...")
    records = load_dataset_stream(args.in_path)
    print(f"Loaded {len(records)} records")
    
    # Compute before distributions
    print("Computing distributions (before)...")
    before_dist = compute_distributions(records)
    
    # Apply exact dedup
    print("Applying exact deduplication...")
    after_exact_dedup, exact_dup_rejected = apply_exact_dedup(records, args.seed)
    print(f"  Exact duplicates removed: {len(exact_dup_rejected)}")
    
    # Apply near dedup
    print(f"Applying near-deduplication (threshold={args.near_dup_threshold})...")
    after_near_dedup, near_dup_rejected = apply_near_dedup(
        after_exact_dedup,
        threshold=args.near_dup_threshold,
        shingle_size=SHINGLE_SIZE,
        bucket_modulo=BUCKET_MODULO
    )
    print(f"  Near-duplicates removed: {len(near_dup_rejected)}")
    
    # Apply quality filters
    print(f"Applying quality filters (min_words={args.min_words}, max_words={args.max_words})...")
    after_quality_filters, quality_filter_rejected = apply_quality_filters(
        after_near_dedup,
        min_words=args.min_words,
        max_words=args.max_words,
        vague_phrase_threshold=args.vague_phrase_threshold,
        concrete_action_threshold=args.concrete_action_threshold,
        overclaim_threshold=OVERCLAIM_REJECTION_THRESHOLD,
    )
    print(f"  Quality filters removed: {len(quality_filter_rejected)}")
    
    # Apply balancing
    print(f"Applying balancing (cap_multiplier={args.cap_multiplier}, cap_multiplier_hard={args.cap_multiplier_hard})...")
    after_balancing, balancing_rejected, balancing_stats = apply_balancing(
        after_quality_filters,
        cap_multiplier=args.cap_multiplier,
        cap_multiplier_hard=args.cap_multiplier_hard,
        seed=args.seed
    )
    print(f"  Balancing removed: {len(balancing_rejected)}")
    
    # Compute after distributions
    print("Computing distributions (after)...")
    after_dist = compute_distributions(after_balancing)
    
    # Combine rejection tracking
    all_rejected = exact_dup_rejected + near_dup_rejected + quality_filter_rejected + balancing_rejected
    
    # Collect rejection reasons and metadata
    rejection_reasons = defaultdict(int)
    rejected_ids = []
    
    for record, reason, metadata in all_rejected:
        rejection_reasons[reason] += 1
        rejected_ids.append(record.get("id", "unknown"))
    
    # Stats
    from datetime import datetime
    exact_dup_count = len(exact_dup_rejected)
    near_dup_count = len(near_dup_rejected)
    quality_filter_count = len(quality_filter_rejected)
    balance_count = len(balancing_rejected)
    
    stats = {
        "timestamp": datetime.now().isoformat(),
        "input_path": str(args.in_path),
        "seed": args.seed,
        "total_in": len(records),
        "after_exact_dup": len(after_exact_dedup),
        "after_near_dup": len(after_near_dedup),
        "after_quality_filters": len(after_quality_filters),
        "accepted": len(after_balancing),
        "rejected": len(all_rejected),
        "exact_dup_count": exact_dup_count,
        "exact_dup_rate": exact_dup_count / len(records) if len(records) > 0 else 0,
        "near_dup_count": near_dup_count,
        "near_dup_rate": near_dup_count / len(records) if len(records) > 0 else 0,
        "quality_filter_count": quality_filter_count,
        "quality_filter_rate": quality_filter_count / len(records) if len(records) > 0 else 0,
        "balancing_count": balance_count,
        "balancing_rate": balance_count / len(records) if len(records) > 0 else 0,
        "qf_min_words": args.min_words,
        "qf_max_words": args.max_words,
        "qf_vague_phrase_threshold": args.vague_phrase_threshold,
        "qf_concrete_action_threshold": args.concrete_action_threshold,
        "qf_overclaim_threshold": OVERCLAIM_REJECTION_THRESHOLD,
    }
    
    # Near-dup statistics
    near_dup_stats = {
        "threshold": args.near_dup_threshold,
        "shingle_size": SHINGLE_SIZE,
        "bucket_modulo": BUCKET_MODULO,
        "near_dup_groups": None,
        "max_group_size": None,
    }
    
    # Collect example pairs
    print("Collecting example pairs...")
    example_pairs = collect_duplicate_examples(records, all_rejected)
    
    # Write outputs
    print(f"Writing clean dataset to {args.out_clean}...")
    write_clean_jsonl(after_balancing, args.out_clean)
    
    print(f"Writing QA report to {args.out_report}...")
    write_qa_report_markdown(
        args.out_report,
        stats,
        before_dist,
        after_dist,
        dict(rejection_reasons),
        example_pairs,
        near_dup_stats,
        balancing_stats
    )
    
    print(f"Writing QA summary to {args.out_summary}...")
    write_qa_summary_json(
        args.out_summary,
        stats,
        before_dist,
        after_dist,
        dict(rejection_reasons),
        rejected_ids,
        example_pairs,
        near_dup_stats,
        balancing_stats,
        args.cap_multiplier,
        args.cap_multiplier_hard
    )
    
    # Print summary
    print("\n" + "="*70)
    print("QA Pipeline Complete")
    print("="*70)
    print(f"Total input:              {stats['total_in']}")
    print(f"Exact duplicates removed: {exact_dup_count} ({stats['exact_dup_rate']:.2%})")
    print(f"Near-duplicates removed:  {near_dup_count} ({stats['near_dup_rate']:.2%})")
    print(f"Quality filters removed:  {quality_filter_count} ({stats['quality_filter_rate']:.2%})")
    print(f"Balancing removed:        {balance_count} ({stats['balancing_rate']:.2%})")
    print(f"Total rejected:           {stats['rejected']} ({stats['rejected']/stats['total_in']:.2%})")
    print(f"Final accepted:           {stats['accepted']}")
    print("="*70)


if __name__ == "__main__":
    main()
