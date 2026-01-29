"""
Tests for balancing module.

Tests cover:
- Bucket key computation
- Bucket statistics
- Target cap calculation
- Downsampling with hard difficulty handling
- Deterministic behavior
- Edge cases
"""

import pytest
from collections import defaultdict

from pocketguide.data_generation.balancing import (
    compute_bucket_key,
    compute_bucket_stats,
    compute_target_caps,
    downsample_records,
    apply_balancing,
)


class TestComputeBucketKey:
    """Test bucket key computation."""
    
    def test_basic_bucket_key(self):
        """Should return (category, difficulty, payload_type)."""
        record = {
            "category": "visa",
            "difficulty": "easy",
            "payload_type": "checklist"
        }
        
        key = compute_bucket_key(record)
        
        assert key == ("visa", "easy", "checklist")
    
    def test_bucket_key_with_missing_fields(self):
        """Should use defaults for missing fields."""
        record = {}
        
        key = compute_bucket_key(record)
        
        assert key == ("unknown", "medium", "unknown")
    
    def test_bucket_key_partial_fields(self):
        """Should use defaults for partially missing fields."""
        record = {"category": "passport"}
        
        key = compute_bucket_key(record)
        
        assert key == ("passport", "medium", "unknown")


class TestComputeBucketStats:
    """Test bucket statistics computation."""
    
    def test_single_bucket(self):
        """Should count single bucket."""
        records = [
            {"category": "visa", "difficulty": "easy", "payload_type": "checklist"},
            {"category": "visa", "difficulty": "easy", "payload_type": "checklist"},
            {"category": "visa", "difficulty": "easy", "payload_type": "checklist"},
        ]
        
        stats = compute_bucket_stats(records)
        
        assert len(stats) == 1
        assert stats[("visa", "easy", "checklist")] == 3
    
    def test_multiple_buckets(self):
        """Should count multiple distinct buckets."""
        records = [
            {"category": "visa", "difficulty": "easy", "payload_type": "checklist"},
            {"category": "visa", "difficulty": "easy", "payload_type": "guide"},
            {"category": "passport", "difficulty": "hard", "payload_type": "checklist"},
        ]
        
        stats = compute_bucket_stats(records)
        
        assert len(stats) == 3
        assert stats[("visa", "easy", "checklist")] == 1
        assert stats[("visa", "easy", "guide")] == 1
        assert stats[("passport", "hard", "checklist")] == 1
    
    def test_empty_records(self):
        """Should return empty dict for no records."""
        records = []
        
        stats = compute_bucket_stats(records)
        
        assert stats == {}


class TestComputeTargetCaps:
    """Test target cap computation."""
    
    def test_equal_sized_buckets(self):
        """Should apply multiplier to median size."""
        bucket_stats = {
            ("visa", "easy", "checklist"): 10,
            ("passport", "easy", "checklist"): 10,
            ("visa", "hard", "checklist"): 10,
        }
        
        caps = compute_target_caps(bucket_stats, cap_multiplier=1.5, cap_multiplier_hard=3.0)
        
        # Median is 10
        # Normal buckets get cap = 10 * 1.5 = 15
        # Hard bucket gets cap = 10 * 3.0 = 30
        assert caps[("visa", "easy", "checklist")] == 15
        assert caps[("passport", "easy", "checklist")] == 15
        assert caps[("visa", "hard", "checklist")] == 30
    
    def test_hard_difficulty_higher_cap(self):
        """Should give hard difficulty buckets higher cap."""
        bucket_stats = {
            ("visa", "easy", "checklist"): 100,
            ("visa", "hard", "checklist"): 5,
        }
        
        caps = compute_target_caps(bucket_stats, cap_multiplier=1.5, cap_multiplier_hard=3.0)
        
        # Median is (5 + 100) / 2 = 52 (or 5 or 100 depending on implementation)
        # For odd number of buckets, median is the middle value when sorted
        # So: 5 (hard=15), 100 (normal=150)
        # Median of [5, 100] is 5 + (100-5)/2 = 52.5, which as int is 52
        # But Python's // gives 52, * 1.5 = 78, * 3.0 = 156
        
        # Let's just check that hard gets more cap than easy
        assert caps[("visa", "hard", "checklist")] > caps[("visa", "easy", "checklist")]
    
    def test_cap_never_zero(self):
        """Should never set cap to 0."""
        bucket_stats = {
            ("visa", "easy", "checklist"): 1,
        }
        
        caps = compute_target_caps(bucket_stats, cap_multiplier=0.1, cap_multiplier_hard=0.1)
        
        # Even with low multiplier, cap should be >= 1
        assert caps[("visa", "easy", "checklist")] >= 1
    
    def test_empty_bucket_stats(self):
        """Should return empty dict for no buckets."""
        bucket_stats = {}
        
        caps = compute_target_caps(bucket_stats)
        
        assert caps == {}


class TestDownsampleRecords:
    """Test downsampling records."""
    
    def test_no_downsampling_needed(self):
        """Should keep all records if under cap."""
        records = [
            {"id": "1", "category": "visa", "difficulty": "easy", "payload_type": "checklist"},
            {"id": "2", "category": "visa", "difficulty": "easy", "payload_type": "checklist"},
        ]
        bucket_caps = {
            ("visa", "easy", "checklist"): 5,
        }
        
        accepted, rejected = downsample_records(records, bucket_caps, seed=42)
        
        assert len(accepted) == 2
        assert len(rejected) == 0
    
    def test_downsampling_applied(self):
        """Should downsample bucket above cap."""
        records = [
            {"id": str(i), "category": "visa", "difficulty": "easy", "payload_type": "checklist"}
            for i in range(10)
        ]
        bucket_caps = {
            ("visa", "easy", "checklist"): 5,
        }
        
        accepted, rejected = downsample_records(records, bucket_caps, seed=42)
        
        assert len(accepted) == 5
        assert len(rejected) == 5
        
        # Check that rejected records have correct reason code
        for record, reason in rejected:
            assert reason == "balanced_downsample"
    
    def test_deterministic_downsampling(self):
        """Should produce same results with same seed."""
        records = [
            {"id": str(i), "category": "visa", "difficulty": "easy", "payload_type": "checklist"}
            for i in range(10)
        ]
        bucket_caps = {
            ("visa", "easy", "checklist"): 5,
        }
        
        # Run twice with same seed
        accepted1, rejected1 = downsample_records(records, bucket_caps, seed=42)
        accepted2, rejected2 = downsample_records(records, bucket_caps, seed=42)
        
        # Results should be identical
        assert len(accepted1) == len(accepted2)
        accepted_ids_1 = sorted([r["id"] for r in accepted1])
        accepted_ids_2 = sorted([r["id"] for r in accepted2])
        assert accepted_ids_1 == accepted_ids_2
    
    def test_different_seeds_different_results(self):
        """Should produce different results with different seeds."""
        records = [
            {"id": str(i), "category": "visa", "difficulty": "easy", "payload_type": "checklist"}
            for i in range(20)
        ]
        bucket_caps = {
            ("visa", "easy", "checklist"): 10,
        }
        
        # Run with different seeds
        accepted1, _ = downsample_records(records, bucket_caps, seed=42)
        accepted2, _ = downsample_records(records, bucket_caps, seed=123)
        
        accepted_ids_1 = set([r["id"] for r in accepted1])
        accepted_ids_2 = set([r["id"] for r in accepted2])
        
        # At least some IDs should differ (not guaranteed but very likely with 20->10)
        # This is a probabilistic test, might flake
        assert len(accepted_ids_1 & accepted_ids_2) <= 9 or len(accepted_ids_1) != len(accepted2)
    
    def test_multiple_buckets(self):
        """Should downsample each bucket independently."""
        records = [
            {"id": f"visa_{i}", "category": "visa", "difficulty": "easy", "payload_type": "checklist"}
            for i in range(20)
        ] + [
            {"id": f"passport_{i}", "category": "passport", "difficulty": "easy", "payload_type": "guide"}
            for i in range(5)
        ]
        
        bucket_caps = {
            ("visa", "easy", "checklist"): 10,
            ("passport", "easy", "guide"): 5,
        }
        
        accepted, rejected = downsample_records(records, bucket_caps, seed=42)
        
        assert len(accepted) == 15  # 10 + 5
        assert len(rejected) == 10  # 20 - 10
        
        # Check distribution
        visa_accepted = [r for r in accepted if "visa" in r["id"]]
        passport_accepted = [r for r in accepted if "passport" in r["id"]]
        
        assert len(visa_accepted) == 10
        assert len(passport_accepted) == 5


class TestApplyBalancing:
    """Test full balancing pipeline."""
    
    def test_apply_balancing_basic(self):
        """Should apply balancing and return stats."""
        records = [
            {"id": str(i), "category": "visa", "difficulty": "easy", "payload_type": "checklist"}
            for i in range(20)
        ]
        
        accepted, rejected, stats = apply_balancing(records, cap_multiplier=1.5, seed=42)
        
        assert len(accepted) + len(rejected) == len(records)
        assert "total_in" in stats
        assert "total_out" in stats
        assert "total_removed" in stats
        assert stats["total_in"] == 20
    
    def test_hard_difficulty_preserved(self):
        """Should preserve hard difficulty buckets."""
        # Create a heavily overrepresented easy bucket and a small hard bucket
        records = [
            {"id": f"easy_{i}", "category": "visa", "difficulty": "easy", "payload_type": "checklist"}
            for i in range(100)
        ] + [
            {"id": f"hard_{i}", "category": "visa", "difficulty": "hard", "payload_type": "checklist"}
            for i in range(2)
        ]
        
        accepted, rejected, stats = apply_balancing(
            records,
            cap_multiplier=1.5,
            cap_multiplier_hard=3.0,
            seed=42
        )
        
        # Hard bucket should be preserved (2 records)
        hard_accepted = [r for r in accepted if "hard" in r["id"]]
        assert len(hard_accepted) == 2, f"Expected 2 hard records, got {len(hard_accepted)}"
    
    def test_deterministic_balancing(self):
        """Should produce deterministic results with fixed seed."""
        records = [
            {"id": str(i), "category": "visa", "difficulty": "easy", "payload_type": "checklist"}
            for i in range(50)
        ]
        
        # Run twice with same seed
        accepted1, _, stats1 = apply_balancing(records, seed=42)
        accepted2, _, stats2 = apply_balancing(records, seed=42)
        
        # Stats should match
        assert stats1["total_out"] == stats2["total_out"]
        assert stats1["total_removed"] == stats2["total_removed"]
        
        # Accepted sets should match
        ids1 = sorted([r["id"] for r in accepted1])
        ids2 = sorted([r["id"] for r in accepted2])
        assert ids1 == ids2
    
    def test_cap_multiplier_effect(self):
        """Should respect cap multiplier settings."""
        records = [
            {"id": str(i), "category": "visa", "difficulty": "easy", "payload_type": "checklist"}
            for i in range(100)
        ]
        
        # Run with higher multiplier - should keep more records
        accepted_low, _, stats_low = apply_balancing(records, cap_multiplier=1.0, seed=42)
        accepted_high, _, stats_high = apply_balancing(records, cap_multiplier=2.0, seed=42)
        
        # Higher multiplier should result in more accepted records
        assert len(accepted_high) >= len(accepted_low)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_records(self):
        """Should handle empty record list."""
        records = []
        bucket_caps = {}
        
        accepted, rejected = downsample_records(records, bucket_caps)
        
        assert len(accepted) == 0
        assert len(rejected) == 0
    
    def test_single_record(self):
        """Should handle single record."""
        records = [{"id": "1", "category": "visa", "difficulty": "easy", "payload_type": "checklist"}]
        
        accepted, rejected, stats = apply_balancing(records)
        
        assert len(accepted) == 1
        assert len(rejected) == 0
    
    def test_records_missing_bucket_fields(self):
        """Should handle records with missing bucket fields."""
        records = [
            {"id": "1"},  # Missing all bucket fields
            {"id": "2", "category": "visa"},  # Partial fields
        ]
        
        accepted, rejected, stats = apply_balancing(records)
        
        # Should still process without error
        assert len(accepted) + len(rejected) == len(records)
    
    def test_very_unbalanced_distribution(self):
        """Should handle heavily skewed distributions."""
        records = [
            {"id": str(i), "category": "visa", "difficulty": "easy", "payload_type": "checklist"}
            for i in range(1000)
        ] + [
            {"id": f"rare_{i}", "category": "passport", "difficulty": "hard", "payload_type": "guide"}
            for i in range(1)
        ]
        
        accepted, rejected, stats = apply_balancing(records)
        
        # Should not delete the rare bucket entirely
        rare_accepted = [r for r in accepted if "rare" in r["id"]]
        assert len(rare_accepted) == 1
    
    def test_all_same_bucket(self):
        """Should handle all records in same bucket."""
        records = [
            {"id": str(i), "category": "visa", "difficulty": "easy", "payload_type": "checklist"}
            for i in range(50)
        ]
        
        accepted, rejected, stats = apply_balancing(records, cap_multiplier=0.5)
        
        assert len(accepted) > 0
        assert len(rejected) > 0


class TestStatsTracking:
    """Test that stats are properly tracked."""
    
    def test_stats_contain_required_fields(self):
        """Should track all required statistics."""
        records = [
            {"id": str(i), "category": "visa", "difficulty": "easy", "payload_type": "checklist"}
            for i in range(20)
        ] + [
            {"id": f"hard_{i}", "category": "visa", "difficulty": "hard", "payload_type": "checklist"}
            for i in range(2)
        ]
        
        accepted, rejected, stats = apply_balancing(records, seed=42)
        
        required_keys = [
            "total_in", "total_out", "total_removed", "removal_rate",
            "cap_multiplier", "cap_multiplier_hard", "median_bucket_size",
            "pre_balance_bucket_stats", "post_balance_bucket_stats", "bucket_caps"
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing required stat: {key}"
    
    def test_stats_math_consistency(self):
        """Should have consistent math in stats."""
        records = [
            {"id": str(i), "category": "visa", "difficulty": "easy", "payload_type": "checklist"}
            for i in range(50)
        ]
        
        accepted, rejected, stats = apply_balancing(records)
        
        # total_in should equal len(records)
        assert stats["total_in"] == len(records)
        
        # total_out should equal len(accepted)
        assert stats["total_out"] == len(accepted)
        
        # total_removed should equal len(rejected)
        assert stats["total_removed"] == len(rejected)
        
        # removal_rate should be removal / total
        expected_rate = len(rejected) / len(records)
        assert abs(stats["removal_rate"] - expected_rate) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
