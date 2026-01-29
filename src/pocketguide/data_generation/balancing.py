"""
Balancing module for reducing dataset skew.

This module implements deterministic balancing by downsampling overrepresented
buckets while preserving hard and rare cases.

Key concepts:
- Buckets are defined by (category, difficulty, payload_type)
- Each bucket gets a target cap based on median bucket size and cap multipliers
- Hard difficulty buckets get higher cap multiplier to preserve hard cases
- Downsampling is deterministic using stable random seed
- Rejected records are marked with reason_code: "balanced_downsample"
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import defaultdict


@dataclass
class BalancingResult:
    """Result of balancing operation."""
    passed: bool
    reason_code: Optional[str] = None
    details: Dict[str, Any] = None


def compute_bucket_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Compute bucket key for a record.
    
    Bucket is defined by (category, difficulty, payload_type).
    These are extracted from the record to group similar items together.
    
    Args:
        record: Record dict with 'category', 'difficulty', 'payload_type'
        
    Returns:
        Tuple of (category, difficulty, payload_type)
        
    Raises:
        KeyError: If required fields missing
    """
    return (
        record.get("category", "unknown"),
        record.get("difficulty", "medium"),
        record.get("payload_type", "unknown")
    )


def compute_bucket_stats(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], int]:
    """
    Compute size of each bucket.
    
    Args:
        records: List of record dicts
        
    Returns:
        Dict mapping bucket_key -> count
    """
    bucket_stats = defaultdict(int)
    for record in records:
        key = compute_bucket_key(record)
        bucket_stats[key] += 1
    
    return dict(bucket_stats)


def compute_target_caps(
    bucket_stats: Dict[Tuple[str, str, str], int],
    cap_multiplier: float = 1.5,
    cap_multiplier_hard: float = 3.0
) -> Dict[Tuple[str, str, str], int]:
    """
    Compute target cap for each bucket.
    
    Strategy:
    1. Compute median bucket size
    2. Hard difficulty buckets get cap = median * cap_multiplier_hard
    3. Other buckets get cap = median * cap_multiplier
    4. Buckets with fewer items than cap are untouched
    
    Args:
        bucket_stats: Dict of bucket_key -> count
        cap_multiplier: Multiplier for normal buckets (default 1.5)
        cap_multiplier_hard: Multiplier for hard difficulty buckets (default 3.0)
        
    Returns:
        Dict mapping bucket_key -> target_cap
    """
    if not bucket_stats:
        return {}
    
    # Compute median bucket size
    sizes = sorted(bucket_stats.values())
    median = sizes[len(sizes) // 2]
    
    target_caps = {}
    for bucket_key, count in bucket_stats.items():
        category, difficulty, payload_type = bucket_key
        
        # Hard difficulty buckets get higher cap
        if difficulty == "hard":
            cap = int(median * cap_multiplier_hard)
        else:
            cap = int(median * cap_multiplier)
        
        # Never set cap to 0
        cap = max(1, cap)
        
        target_caps[bucket_key] = cap
    
    return target_caps


def downsample_records(
    records: List[Dict[str, Any]],
    bucket_caps: Dict[Tuple[str, str, str], int],
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], str]]]:
    """
    Deterministically downsample records per bucket.
    
    For each bucket:
    - If count <= cap: keep all records
    - If count > cap: randomly sample cap records (deterministic with seed)
    
    The sampling is stable: records are shuffled with fixed seed before sampling,
    ensuring consistent results across runs.
    
    Args:
        records: List of record dicts
        bucket_caps: Dict of bucket_key -> target_cap
        seed: Random seed for deterministic sampling (default 42)
        
    Returns:
        Tuple of (accepted_records, rejected_records_with_reason)
        where rejected_records_with_reason is list of (record, reason_code) tuples
    """
    # Group records by bucket
    bucket_records = defaultdict(list)
    for record in records:
        key = compute_bucket_key(record)
        bucket_records[key].append(record)
    
    # Deterministically downsample each bucket
    rng = random.Random(seed)
    accepted = []
    rejected = []
    
    for bucket_key, bucket_recs in bucket_records.items():
        cap = bucket_caps.get(bucket_key, len(bucket_recs))
        
        if len(bucket_recs) <= cap:
            # Keep all records in this bucket
            accepted.extend(bucket_recs)
        else:
            # Downsample: shuffle with seed, then take first cap records
            shuffled = bucket_recs.copy()
            rng.shuffle(shuffled)
            
            accepted.extend(shuffled[:cap])
            for record in shuffled[cap:]:
                rejected.append((record, "balanced_downsample"))
    
    return accepted, rejected


def apply_balancing(
    records: List[Dict[str, Any]],
    cap_multiplier: float = 1.5,
    cap_multiplier_hard: float = 3.0,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], str]], Dict[str, Any]]:
    """
    Apply full balancing pipeline.
    
    Args:
        records: List of record dicts
        cap_multiplier: Multiplier for normal buckets
        cap_multiplier_hard: Multiplier for hard difficulty buckets
        seed: Random seed for deterministic downsampling
        
    Returns:
        Tuple of (accepted_records, rejected_with_reason, stats_dict)
        where stats_dict contains balancing metrics
    """
    # Compute bucket statistics
    pre_balance_stats = compute_bucket_stats(records)
    
    # Compute target caps per bucket
    bucket_caps = compute_target_caps(
        pre_balance_stats,
        cap_multiplier=cap_multiplier,
        cap_multiplier_hard=cap_multiplier_hard
    )
    
    # Apply downsampling
    accepted, rejected = downsample_records(records, bucket_caps, seed=seed)
    
    # Compute post-balance statistics
    post_balance_stats = compute_bucket_stats(accepted)
    
    # Build stats dict
    stats = {
        "total_in": len(records),
        "total_out": len(accepted),
        "total_removed": len(rejected),
        "removal_rate": len(rejected) / len(records) if records else 0.0,
        "cap_multiplier": cap_multiplier,
        "cap_multiplier_hard": cap_multiplier_hard,
        "median_bucket_size": int(sorted(pre_balance_stats.values())[len(pre_balance_stats) // 2]) if pre_balance_stats else 0,
        "pre_balance_bucket_stats": pre_balance_stats,
        "post_balance_bucket_stats": post_balance_stats,
        "bucket_caps": bucket_caps,
    }
    
    return accepted, rejected, stats
