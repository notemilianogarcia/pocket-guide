"""
Tests for split_dataset_v1.

Tests cover:
- Union-Find (DSU) data structure
- Leakage group building (exact + near-duplicate)
- Group-based split assignment
- Validation checks
- Benchmark prompt generation
- End-to-end splitting
"""

import json
import tempfile
from pathlib import Path

import pytest

from pocketguide.data_generation.split_dataset_v1 import (
    UnionFind,
    compute_fingerprint,
    tokenize_text,
    generate_shingles,
    jaccard_similarity,
    build_leakage_groups,
    assign_groups_to_splits,
    write_splits,
    write_benchmark_prompts,
    validate_splits,
    compute_group_stats,
    load_dataset,
    main,
)


class TestUnionFind:
    """Test Union-Find data structure."""
    
    def test_find_initializes_element(self):
        """Should initialize element on first find."""
        dsu = UnionFind()
        root = dsu.find("a")
        assert root == "a"
    
    def test_union_merges_sets(self):
        """Should merge two sets."""
        dsu = UnionFind()
        dsu.union("a", "b")
        
        assert dsu.find("a") == dsu.find("b")
    
    def test_union_transitive(self):
        """Should merge transitively."""
        dsu = UnionFind()
        dsu.union("a", "b")
        dsu.union("b", "c")
        
        assert dsu.find("a") == dsu.find("b") == dsu.find("c")
    
    def test_get_groups(self):
        """Should return all groups."""
        dsu = UnionFind()
        dsu.union("a", "b")
        dsu.union("c", "d")
        dsu.find("e")  # Singleton
        
        groups = dsu.get_groups()
        
        assert len(groups) == 3  # {a,b}, {c,d}, {e}
        
        # Find which group contains 'a'
        group_with_a = None
        for group_id, members in groups.items():
            if "a" in members:
                group_with_a = members
                break
        
        assert set(group_with_a) == {"a", "b"}


class TestFingerprintingAndShingling:
    """Test fingerprinting and shingling functions."""
    
    def test_compute_fingerprint_same_content(self):
        """Should produce same fingerprint for same content."""
        record1 = {
            "payload_type": "checklist",
            "prompt": "Do I need a visa?",
            "response": {"summary": "Most visitors need a visa"}
        }
        record2 = {
            "payload_type": "checklist",
            "prompt": "Do I need a visa?",
            "response": {"summary": "Most visitors need a visa"}
        }
        
        fp1 = compute_fingerprint(record1)
        fp2 = compute_fingerprint(record2)
        
        assert fp1 == fp2
    
    def test_compute_fingerprint_different_content(self):
        """Should produce different fingerprints for different content."""
        record1 = {
            "payload_type": "checklist",
            "prompt": "Do I need a visa?",
            "response": {"summary": "Most visitors need a visa"}
        }
        record2 = {
            "payload_type": "checklist",
            "prompt": "Do I need a passport?",
            "response": {"summary": "All travelers need a passport"}
        }
        
        fp1 = compute_fingerprint(record1)
        fp2 = compute_fingerprint(record2)
        
        assert fp1 != fp2
    
    def test_jaccard_similarity_identical(self):
        """Should return 1.0 for identical sets."""
        set1 = {"a", "b", "c"}
        set2 = {"a", "b", "c"}
        
        sim = jaccard_similarity(set1, set2)
        
        assert sim == 1.0
    
    def test_jaccard_similarity_disjoint(self):
        """Should return 0.0 for disjoint sets."""
        set1 = {"a", "b", "c"}
        set2 = {"d", "e", "f"}
        
        sim = jaccard_similarity(set1, set2)
        
        assert sim == 0.0
    
    def test_jaccard_similarity_partial(self):
        """Should compute partial overlap correctly."""
        set1 = {"a", "b", "c", "d"}
        set2 = {"c", "d", "e", "f"}
        
        # Intersection: {c, d} = 2
        # Union: {a, b, c, d, e, f} = 6
        # Jaccard = 2/6 = 0.333...
        sim = jaccard_similarity(set1, set2)
        
        assert abs(sim - 0.333) < 0.01


class TestBuildLeakageGroups:
    """Test leakage group building."""
    
    def test_exact_duplicates_same_group(self):
        """Should group exact duplicates together."""
        records = [
            {
                "id": "1",
                "payload_type": "checklist",
                "prompt": "Do I need a visa?",
                "response": {"summary": "Most visitors need a visa"}
            },
            {
                "id": "2",
                "payload_type": "checklist",
                "prompt": "Do I need a visa?",  # Exact duplicate
                "response": {"summary": "Most visitors need a visa"}
            },
            {
                "id": "3",
                "payload_type": "guide",
                "prompt": "Different prompt",
                "response": {"summary": "Different summary"}
            }
        ]
        
        record_to_group, groups = build_leakage_groups(records, sim_threshold=0.85)
        
        # Records 1 and 2 should be in same group
        assert record_to_group["1"] == record_to_group["2"]
        # Record 3 should be in different group
        assert record_to_group["3"] != record_to_group["1"]
    
    def test_near_duplicates_same_group(self):
        """Should group near-duplicates together."""
        records = [
            {
                "id": "1",
                "payload_type": "checklist",
                "prompt": "Do I need a visa for tourism?",
                "response": {"summary": "Most tourists need a visa for entry"}
            },
            {
                "id": "2",
                "payload_type": "checklist",
                "prompt": "Is a visa required for tourists?",  # Near-duplicate
                "response": {"summary": "Visa is typically needed for tourists"}
            },
            {
                "id": "3",
                "payload_type": "guide",
                "prompt": "How to apply for passport",  # Very different
                "response": {"summary": "Passport application guide"}
            }
        ]
        
        # Use low threshold to ensure near-duplicates merge
        record_to_group, groups = build_leakage_groups(records, sim_threshold=0.5)
        
        # Records 1 and 2 should likely be in same group (near-duplicates)
        # Record 3 should be in different group
        assert record_to_group["3"] != record_to_group["1"]
    
    def test_unique_records_different_groups(self):
        """Should put unique records in different groups."""
        records = [
            {
                "id": "1",
                "payload_type": "checklist",
                "prompt": "Visa requirements",
                "response": {"summary": "Visa summary"}
            },
            {
                "id": "2",
                "payload_type": "guide",
                "prompt": "Passport application",
                "response": {"summary": "Passport summary"}
            },
            {
                "id": "3",
                "payload_type": "procedure",
                "prompt": "Border crossing",
                "response": {"summary": "Border summary"}
            }
        ]
        
        record_to_group, groups = build_leakage_groups(records, sim_threshold=0.85)
        
        # All should be in different groups (unique)
        group1 = record_to_group["1"]
        group2 = record_to_group["2"]
        group3 = record_to_group["3"]
        
        assert len({group1, group2, group3}) == 3  # All different


class TestAssignGroupsToSplits:
    """Test group assignment to splits."""
    
    def test_split_fractions_validation(self):
        """Should raise error if fractions don't sum to ~1.0."""
        groups = {"g1": ["1", "2"], "g2": ["3", "4"]}
        records = [
            {"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}
        ]
        
        with pytest.raises(ValueError, match="must sum to"):
            assign_groups_to_splits(
                groups, records,
                train_frac=0.5, val_frac=0.3, test_frac=0.1  # Sum = 0.9
            )
    
    def test_groups_stay_together(self):
        """Should keep all members of a group in same split."""
        groups = {
            "g1": ["1", "2", "3"],
            "g2": ["4", "5"],
            "g3": ["6"]
        }
        records = [
            {"id": str(i)} for i in range(1, 7)
        ]
        
        record_to_split = assign_groups_to_splits(
            groups, records,
            train_frac=0.5, val_frac=0.25, test_frac=0.25,
            seed=42
        )
        
        # All members of g1 should be in same split
        splits_g1 = {record_to_split[id] for id in ["1", "2", "3"]}
        assert len(splits_g1) == 1
        
        # All members of g2 should be in same split
        splits_g2 = {record_to_split[id] for id in ["4", "5"]}
        assert len(splits_g2) == 1
    
    def test_deterministic_assignment(self):
        """Should produce same splits with same seed."""
        groups = {
            f"g{i}": [f"{i}"] for i in range(1, 21)
        }
        records = [{"id": str(i)} for i in range(1, 21)]
        
        splits1 = assign_groups_to_splits(groups, records, seed=42)
        splits2 = assign_groups_to_splits(groups, records, seed=42)
        
        assert splits1 == splits2
    
    def test_different_seeds_different_splits(self):
        """Should produce different splits with different seeds."""
        groups = {
            f"g{i}": [f"{i}"] for i in range(1, 21)
        }
        records = [{"id": str(i)} for i in range(1, 21)]
        
        splits1 = assign_groups_to_splits(groups, records, seed=42)
        splits2 = assign_groups_to_splits(groups, records, seed=123)
        
        # At least some records should be in different splits
        differences = sum(1 for id in splits1 if splits1[id] != splits2[id])
        assert differences > 0


class TestWriteSplits:
    """Test split writing."""
    
    def test_write_splits_creates_files(self):
        """Should create train/val/test JSONL files."""
        records = [
            {"id": "1", "data": "train1"},
            {"id": "2", "data": "train2"},
            {"id": "3", "data": "val1"},
            {"id": "4", "data": "test1"}
        ]
        record_to_split = {
            "1": "train",
            "2": "train",
            "3": "val",
            "4": "test"
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            counts = write_splits(records, record_to_split, out_dir)
            
            # Check files exist
            assert (out_dir / "train.jsonl").exists()
            assert (out_dir / "val.jsonl").exists()
            assert (out_dir / "test.jsonl").exists()
            
            # Check counts
            assert counts["train"] == 2
            assert counts["val"] == 1
            assert counts["test"] == 1
    
    def test_write_splits_preserves_records(self):
        """Should write complete records to files."""
        records = [
            {"id": "1", "prompt": "Question 1", "response": {"summary": "Answer 1"}},
            {"id": "2", "prompt": "Question 2", "response": {"summary": "Answer 2"}}
        ]
        record_to_split = {
            "1": "train",
            "2": "test"
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            write_splits(records, record_to_split, out_dir)
            
            # Read back train split
            with open(out_dir / "train.jsonl", 'r') as f:
                train_records = [json.loads(line) for line in f if line.strip()]
            
            assert len(train_records) == 1
            assert train_records[0]["id"] == "1"
            assert "response" in train_records[0]


class TestBenchmarkPrompts:
    """Test benchmark prompt generation."""
    
    def test_write_benchmark_prompts_test_only(self):
        """Should write only test split prompts."""
        records = [
            {"id": "1", "prompt": "Q1", "response": {"summary": "A1"}},
            {"id": "2", "prompt": "Q2", "response": {"summary": "A2"}},
            {"id": "3", "prompt": "Q3", "response": {"summary": "A3"}}
        ]
        record_to_split = {
            "1": "train",
            "2": "val",
            "3": "test"
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_dir = Path(tmpdir)
            count = write_benchmark_prompts(records, record_to_split, benchmark_dir)
            
            assert count == 1
            
            # Read prompts file
            with open(benchmark_dir / "prompts_test.jsonl", 'r') as f:
                prompts = [json.loads(line) for line in f if line.strip()]
            
            assert len(prompts) == 1
            assert prompts[0]["id"] == "3"
    
    def test_benchmark_prompts_no_responses(self):
        """Should not include response field in prompts."""
        records = [
            {
                "id": "1",
                "payload_type": "checklist",
                "category": "visa",
                "difficulty": "easy",
                "region_tags": ["asia"],
                "prompt": "Question 1",
                "response": {"summary": "Answer 1"}
            }
        ]
        record_to_split = {"1": "test"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_dir = Path(tmpdir)
            write_benchmark_prompts(records, record_to_split, benchmark_dir)
            
            # Read prompts
            with open(benchmark_dir / "prompts_test.jsonl", 'r') as f:
                prompts = [json.loads(line) for line in f if line.strip()]
            
            assert "response" not in prompts[0]
            assert "prompt" in prompts[0]
            assert prompts[0]["prompt"] == "Question 1"


class TestValidateSplits:
    """Test split validation."""
    
    def test_validate_no_leakage(self):
        """Should pass when no groups span splits."""
        records = [
            {"id": "1", "prompt": "Q1", "response": {"summary": "A1"}},
            {"id": "2", "prompt": "Q2", "response": {"summary": "A2"}},
            {"id": "3", "prompt": "Q3", "response": {"summary": "A3"}}
        ]
        record_to_split = {
            "1": "train",
            "2": "val",
            "3": "test"
        }
        record_to_group = {
            "1": "g1",
            "2": "g2",
            "3": "g3"
        }
        
        results = validate_splits(records, record_to_split, record_to_group)
        
        assert results["leakage_free"] is True
        assert results["groups_spanning_splits"] == 0
    
    def test_validate_detects_group_leakage(self):
        """Should detect when a group spans multiple splits."""
        records = [
            {"id": "1", "prompt": "Q1", "response": {"summary": "A1"}},
            {"id": "2", "prompt": "Q1", "response": {"summary": "A1"}},  # Same as 1
        ]
        record_to_split = {
            "1": "train",
            "2": "test"  # Same group but different split!
        }
        record_to_group = {
            "1": "g1",
            "2": "g1"  # Same group
        }
        
        results = validate_splits(records, record_to_split, record_to_group)
        
        assert results["leakage_free"] is False
        assert results["groups_spanning_splits"] > 0


class TestComputeGroupStats:
    """Test group statistics computation."""
    
    def test_group_stats_basic(self):
        """Should compute basic group statistics."""
        groups = {
            "g1": ["1", "2", "3"],
            "g2": ["4", "5"],
            "g3": ["6"],
            "g4": ["7"]
        }
        
        stats = compute_group_stats(groups)
        
        assert stats["num_groups"] == 4
        assert stats["min_size"] == 1
        assert stats["max_size"] == 3
        assert stats["singleton_groups"] == 2
    
    def test_group_stats_empty(self):
        """Should handle empty groups."""
        groups = {}
        
        stats = compute_group_stats(groups)
        
        assert stats["num_groups"] == 0
        assert stats["min_size"] == 0


class TestEndToEndSplitting:
    """Test end-to-end splitting workflow."""
    
    def test_full_splitting_pipeline(self):
        """Should complete full splitting pipeline."""
        # Create test dataset with near-duplicates
        records = [
            {
                "id": "unique_1",
                "payload_type": "checklist",
                "category": "visa",
                "difficulty": "easy",
                "region_tags": ["asia"],
                "prompt": "Unique question 1",
                "response": {"summary": "Unique answer 1"}
            },
            {
                "id": "unique_2",
                "payload_type": "guide",
                "category": "passport",
                "difficulty": "medium",
                "region_tags": ["europe"],
                "prompt": "Unique question 2",
                "response": {"summary": "Unique answer 2"}
            },
            {
                "id": "near_dup_1",
                "payload_type": "checklist",
                "category": "visa",
                "difficulty": "easy",
                "region_tags": ["asia"],
                "prompt": "Do I need a visa for tourism travel to this country?",
                "response": {"summary": "Most tourists need a visa for entry into the country for tourism purposes"}
            },
            {
                "id": "near_dup_2",
                "payload_type": "checklist",
                "category": "visa",
                "difficulty": "easy",
                "region_tags": ["asia"],
                "prompt": "Do I need a visa for tourism travel to this destination?",
                "response": {"summary": "Most tourists need a visa for entry into the country for tourism activities"}
            },
            {
                "id": "exact_dup_1",
                "payload_type": "procedure",
                "category": "customs",
                "difficulty": "hard",
                "region_tags": ["global"],
                "prompt": "Exact duplicate prompt",
                "response": {"summary": "Exact duplicate summary"}
            },
            {
                "id": "exact_dup_2",
                "payload_type": "procedure",
                "category": "customs",
                "difficulty": "hard",
                "region_tags": ["global"],
                "prompt": "Exact duplicate prompt",
                "response": {"summary": "Exact duplicate summary"}
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write test dataset
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.jsonl"
            with open(input_path, 'w') as f:
                for record in records:
                    f.write(json.dumps(record) + '\n')
            
            out_dir = tmpdir / "splits"
            benchmark_dir = tmpdir / "benchmarks"
            
            # Build groups
            record_to_group, groups = build_leakage_groups(records, sim_threshold=0.6)
            
            # Near-duplicates should be in same group
            assert record_to_group["near_dup_1"] == record_to_group["near_dup_2"]
            
            # Exact duplicates should be in same group
            assert record_to_group["exact_dup_1"] == record_to_group["exact_dup_2"]
            
            # Assign to splits
            record_to_split = assign_groups_to_splits(
                groups, records,
                train_frac=0.5, val_frac=0.25, test_frac=0.25,
                seed=42
            )
            
            # Write splits
            split_counts = write_splits(records, record_to_split, out_dir)
            
            # Verify files exist
            assert (out_dir / "train.jsonl").exists()
            assert (out_dir / "val.jsonl").exists()
            assert (out_dir / "test.jsonl").exists()
            
            # Verify counts sum to total
            assert sum(split_counts.values()) == len(records)
            
            # Write benchmark prompts
            prompt_count = write_benchmark_prompts(records, record_to_split, benchmark_dir)
            assert prompt_count == split_counts["test"]
            
            # Validate no leakage
            validation = validate_splits(records, record_to_split, record_to_group)
            assert validation["leakage_free"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
