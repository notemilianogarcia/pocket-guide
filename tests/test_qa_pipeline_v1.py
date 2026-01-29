"""
Tests for QA Pipeline v1.

Tests cover:
- Exact deduplication behavior
- Fingerprint computation
- Distribution tracking
- Output generation
- Determinism
"""

import json
import tempfile
from pathlib import Path

import pytest

from pocketguide.data_generation.qa_pipeline_v1 import (
    normalize_text,
    compute_fingerprint,
    load_dataset_stream,
    compute_distributions,
    apply_exact_dedup,
    write_clean_jsonl,
    collect_duplicate_examples,
    write_qa_report_markdown,
    write_qa_summary_json,
    main,
)


class TestNormalizeText:
    """Test text normalization."""
    
    def test_strip_whitespace(self):
        """Should strip leading/trailing whitespace."""
        assert normalize_text("  hello  ") == "hello"
    
    def test_collapse_whitespace(self):
        """Should collapse multiple spaces to single space."""
        assert normalize_text("hello    world") == "hello world"
    
    def test_collapse_newlines(self):
        """Should collapse newlines to single space."""
        assert normalize_text("hello\n\nworld") == "hello world"
    
    def test_lowercase(self):
        """Should convert to lowercase."""
        assert normalize_text("Hello WORLD") == "hello world"
    
    def test_combined(self):
        """Should handle all normalizations together."""
        text = "  Hello   \n  World\t\t!  "
        assert normalize_text(text) == "hello world !"


class TestComputeFingerprint:
    """Test fingerprint computation."""
    
    def test_same_content_same_fingerprint(self):
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
        
        assert compute_fingerprint(record1) == compute_fingerprint(record2)
    
    def test_whitespace_insensitive(self):
        """Should ignore whitespace differences."""
        record1 = {
            "payload_type": "checklist",
            "prompt": "Do I need a visa?",
            "response": {"summary": "Most visitors need a visa"}
        }
        record2 = {
            "payload_type": "checklist",
            "prompt": "Do  I  need  a  visa?",  # Extra spaces
            "response": {"summary": "Most visitors\nneed a visa"}  # Newline
        }
        
        assert compute_fingerprint(record1) == compute_fingerprint(record2)
    
    def test_case_insensitive(self):
        """Should ignore case differences."""
        record1 = {
            "payload_type": "checklist",
            "prompt": "Do I need a visa?",
            "response": {"summary": "Most visitors need a visa"}
        }
        record2 = {
            "payload_type": "CHECKLIST",
            "prompt": "DO I NEED A VISA?",
            "response": {"summary": "MOST VISITORS NEED A VISA"}
        }
        
        assert compute_fingerprint(record1) == compute_fingerprint(record2)
    
    def test_different_content_different_fingerprint(self):
        """Should produce different fingerprints for different content."""
        record1 = {
            "payload_type": "checklist",
            "prompt": "Do I need a visa?",
            "response": {"summary": "Most visitors need a visa"}
        }
        record2 = {
            "payload_type": "checklist",
            "prompt": "Do I need a visa?",
            "response": {"summary": "Different summary"}  # Different summary
        }
        
        assert compute_fingerprint(record1) != compute_fingerprint(record2)
    
    def test_payload_type_from_response(self):
        """Should extract payload_type from response if not at root."""
        record = {
            "prompt": "Test",
            "response": {
                "payload_type": "itinerary",
                "summary": "Test summary"
            }
        }
        
        fingerprint = compute_fingerprint(record)
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 64  # SHA256 hex length


class TestLoadDatasetStream:
    """Test dataset loading."""
    
    def test_load_valid_jsonl(self):
        """Should load valid JSONL."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            f.write(json.dumps({"id": "1", "category": "visa"}) + "\n")
            f.write(json.dumps({"id": "2", "category": "safety"}) + "\n")
            path = Path(f.name)
        
        try:
            records = load_dataset_stream(path)
            assert len(records) == 2
            assert records[0]["id"] == "1"
            assert records[1]["id"] == "2"
        finally:
            path.unlink()
    
    def test_skip_blank_lines(self):
        """Should skip blank lines."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            f.write(json.dumps({"id": "1"}) + "\n")
            f.write("\n")
            f.write(json.dumps({"id": "2"}) + "\n")
            path = Path(f.name)
        
        try:
            records = load_dataset_stream(path)
            assert len(records) == 2
        finally:
            path.unlink()


class TestComputeDistributions:
    """Test distribution computation."""
    
    def test_compute_all_dimensions(self):
        """Should compute counts across all dimensions."""
        records = [
            {
                "category": "visa",
                "difficulty": "medium",
                "payload_type": "checklist",
                "region_tags": ["asia", "southeast-asia"]
            },
            {
                "category": "visa",
                "difficulty": "easy",
                "payload_type": "itinerary",
                "region_tags": ["asia", "europe"]
            },
            {
                "category": "safety",
                "difficulty": "medium",
                "payload_type": "checklist",
                "region_tags": ["africa"]
            }
        ]
        
        dist = compute_distributions(records)
        
        # Payload type
        assert dist["payload_type"]["checklist"] == 2
        assert dist["payload_type"]["itinerary"] == 1
        
        # Category
        assert dist["category"]["visa"] == 2
        assert dist["category"]["safety"] == 1
        
        # Difficulty
        assert dist["difficulty"]["medium"] == 2
        assert dist["difficulty"]["easy"] == 1
        
        # Region tags
        assert dist["region_tag"]["asia"] == 2
        assert dist["region_tag"]["southeast-asia"] == 1
        assert dist["region_tag"]["europe"] == 1
        assert dist["region_tag"]["africa"] == 1
    
    def test_payload_type_from_response(self):
        """Should extract payload_type from response if not at root."""
        records = [
            {
                "category": "visa",
                "response": {"payload_type": "checklist"}
            }
        ]
        
        dist = compute_distributions(records)
        assert dist["payload_type"]["checklist"] == 1


class TestApplyExactDedup:
    """Test exact deduplication."""
    
    def test_no_duplicates(self):
        """Should accept all unique records."""
        records = [
            {
                "id": "1",
                "payload_type": "checklist",
                "prompt": "Prompt 1",
                "response": {"summary": "Summary 1"}
            },
            {
                "id": "2",
                "payload_type": "checklist",
                "prompt": "Prompt 2",
                "response": {"summary": "Summary 2"}
            }
        ]
        
        accepted, rejected = apply_exact_dedup(records, seed=42)
        
        assert len(accepted) == 2
        assert len(rejected) == 0
    
    def test_exact_duplicates_rejected(self):
        """Should reject exact duplicates, keep first."""
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
                "prompt": "Do I need a visa?",
                "response": {"summary": "Most visitors need a visa"}
            },
            {
                "id": "3",
                "payload_type": "checklist",
                "prompt": "Do I need a visa?",
                "response": {"summary": "Most visitors need a visa"}
            }
        ]
        
        accepted, rejected = apply_exact_dedup(records, seed=42)
        
        # Should keep first, reject second and third
        assert len(accepted) == 1
        assert accepted[0]["id"] == "1"
        
        assert len(rejected) == 2
        assert rejected[0][0]["id"] == "2"
        assert rejected[0][1] == "exact_duplicate"
        assert "anchor_id" in rejected[0][2]
        assert rejected[1][0]["id"] == "3"
        assert rejected[1][1] == "exact_duplicate"
        assert "anchor_id" in rejected[1][2]
    
    def test_whitespace_variations_are_duplicates(self):
        """Should treat whitespace variations as duplicates."""
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
                "prompt": "Do  I  need  a  visa?",  # Extra spaces
                "response": {"summary": "Most visitors\nneed a visa"}  # Newline
            }
        ]
        
        accepted, rejected = apply_exact_dedup(records, seed=42)
        
        assert len(accepted) == 1
        assert len(rejected) == 1
    
    def test_case_variations_are_duplicates(self):
        """Should treat case variations as duplicates."""
        records = [
            {
                "id": "1",
                "payload_type": "checklist",
                "prompt": "Do I need a visa?",
                "response": {"summary": "Most visitors need a visa"}
            },
            {
                "id": "2",
                "payload_type": "CHECKLIST",
                "prompt": "DO I NEED A VISA?",
                "response": {"summary": "MOST VISITORS NEED A VISA"}
            }
        ]
        
        accepted, rejected = apply_exact_dedup(records, seed=42)
        
        assert len(accepted) == 1
        assert len(rejected) == 1
    
    def test_different_payload_type_not_duplicate(self):
        """Should not treat records with different payload_type as duplicates."""
        records = [
            {
                "id": "1",
                "payload_type": "checklist",
                "prompt": "Do I need a visa?",
                "response": {"summary": "Most visitors need a visa"}
            },
            {
                "id": "2",
                "payload_type": "itinerary",  # Different payload_type
                "prompt": "Do I need a visa?",
                "response": {"summary": "Most visitors need a visa"}
            }
        ]
        
        accepted, rejected = apply_exact_dedup(records, seed=42)
        
        assert len(accepted) == 2
        assert len(rejected) == 0


class TestCollectDuplicateExamples:
    """Test duplicate example collection."""
    
    def test_collect_duplicate_pairs(self):
        """Should collect duplicate pairs from rejected records."""
        rejected_with_reason = [
            (
                {"id": "dup_1", "prompt": "Test", "response": {"summary": "Test"}},
                "exact_duplicate",
                {"anchor_id": "orig_1", "similarity": 1.0}
            ),
            (
                {"id": "dup_2", "prompt": "Test similar", "response": {"summary": "Test similar"}},
                "near_duplicate",
                {"anchor_id": "orig_2", "similarity": 0.85}
            )
        ]
        
        examples = collect_duplicate_examples([], rejected_with_reason, max_pairs=5)
        
        assert "exact_duplicate" in examples
        assert "near_duplicate" in examples
        assert len(examples["exact_duplicate"]) == 1
        assert len(examples["near_duplicate"]) == 1


class TestWriteOutputs:
    """Test output file generation."""
    
    def test_write_clean_jsonl(self):
        """Should write clean JSONL with accepted records."""
        records = [
            {"id": "1", "category": "visa"},
            {"id": "2", "category": "safety"}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clean.jsonl"
            write_clean_jsonl(records, path)
            
            assert path.exists()
            
            written = load_dataset_stream(path)
            assert len(written) == 2
            assert written[0]["id"] == "1"
            assert written[1]["id"] == "2"
    
    def test_write_qa_report_markdown(self):
        """Should write markdown report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            
            stats = {
                "timestamp": "2024-01-01",
                "input_path": "test.jsonl",
                "seed": 42,
                "total_in": 100,
                "accepted": 90,
                "rejected": 10,
                "exact_dup_count": 8,
                "exact_dup_rate": 0.08,
                "near_dup_count": 2,
                "near_dup_rate": 0.02,
            }
            
            before_dist = {
                "payload_type": {"checklist": 60, "itinerary": 40},
                "category": {"visa": 50, "safety": 50},
                "difficulty": {"easy": 30, "medium": 40, "hard": 30},
                "region_tag": {"asia": 50, "europe": 30, "africa": 20}
            }
            
            after_dist = {
                "payload_type": {"checklist": 54, "itinerary": 36},
                "category": {"visa": 45, "safety": 45},
                "difficulty": {"easy": 27, "medium": 36, "hard": 27},
                "region_tag": {"asia": 45, "europe": 27, "africa": 18}
            }
            
            rejection_reasons = {"exact_duplicate": 8, "near_duplicate": 2}
            example_pairs = {
                "exact_duplicate": [{"duplicate_id": "d1", "anchor_id": "a1", "similarity": 1.0}],
                "near_duplicate": [{"duplicate_id": "d2", "anchor_id": "a2", "similarity": 0.85}]
            }
            near_dup_stats = {"threshold": 0.85, "shingle_size": 3, "near_dup_groups": 2, "max_group_size": 2}
            
            write_qa_report_markdown(
                path, stats, before_dist, after_dist, rejection_reasons, example_pairs, near_dup_stats
            )
            
            assert path.exists()
            
            content = path.read_text()
            assert "# Dataset QA Report v1" in content
            assert "Total input:** 100" in content
            assert "Accepted:** 90" in content
            assert "exact_duplicate:** 8" in content
            assert "near_duplicate:** 2" in content
    
    def test_write_qa_summary_json(self):
        """Should write JSON summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            
            stats = {
                "timestamp": "2024-01-01",
                "input_path": "test.jsonl",
                "seed": 42,
                "total_in": 100,
                "accepted": 90,
                "rejected": 10,
                "exact_dup_count": 8,
                "exact_dup_rate": 0.08,
                "near_dup_count": 2,
                "near_dup_rate": 0.02,
            }
            
            before_dist = {"payload_type": {"checklist": 60}}
            after_dist = {"payload_type": {"checklist": 54}}
            rejection_reasons = {"exact_duplicate": 8, "near_duplicate": 2}
            rejected_ids = ["id1", "id2"]
            example_pairs = {"exact_duplicate": [], "near_duplicate": []}
            near_dup_stats = {"threshold": 0.85, "shingle_size": 3, "bucket_modulo": 1000}
            
            write_qa_summary_json(
                path, stats, before_dist, after_dist, rejection_reasons, rejected_ids, example_pairs, near_dup_stats
            )
            
            assert path.exists()
            
            with open(path, 'r') as f:
                summary = json.load(f)
            
            assert summary["counts"]["total_in"] == 100
            assert summary["counts"]["accepted"] == 90
            assert summary["counts"]["exact_dup_count"] == 8
            assert summary["counts"]["near_dup_count"] == 2
            assert summary["rejection_reasons"]["exact_duplicate"] == 8
            assert summary["rejected_ids_sample"] == ["id1", "id2"]


class TestQAPipelineIntegration:
    """Integration tests for full QA pipeline."""
    
    def test_full_pipeline(self):
        """Should run full pipeline end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create input dataset with duplicates
            input_path = tmpdir / "input.jsonl"
            with open(input_path, 'w') as f:
                # 3 unique + 2 duplicates = 5 total
                for i in range(3):
                    f.write(json.dumps({
                        "id": f"unique_{i}",
                        "category": "visa",
                        "difficulty": "medium",
                        "payload_type": "checklist",
                        "region_tags": ["asia"],
                        "prompt": f"Unique prompt {i}",
                        "response": {
                            "summary": f"Unique summary {i}. " + "word " * 40,
                            "next_steps": [
                                {"description": "Call the office"},
                                {"description": "Submit documents"},
                                {"description": "Verify status"}
                            ],
                            "verification_steps": [
                                {"description": "Check website"}
                            ],
                            "uncertainty_notes": ""
                        }
                    }) + "\n")
                
                # Add 2 duplicates of first record
                for i in range(2):
                    f.write(json.dumps({
                        "id": f"duplicate_{i}",
                        "category": "visa",
                        "difficulty": "medium",
                        "payload_type": "checklist",
                        "region_tags": ["asia"],
                        "prompt": "Unique prompt 0",  # Same as first
                        "response": {
                            "summary": "Unique summary 0. " + "word " * 40,
                            "next_steps": [
                                {"description": "Call the office"},
                                {"description": "Submit documents"},
                                {"description": "Verify status"}
                            ],
                            "verification_steps": [
                                {"description": "Check website"}
                            ],
                            "uncertainty_notes": ""
                        }
                    }) + "\n")
            
            # Run pipeline
            import sys
            from unittest.mock import patch
            
            out_clean = tmpdir / "clean.jsonl"
            out_report = tmpdir / "report.md"
            out_summary = tmpdir / "summary.json"
            
            args = [
                "qa_pipeline_v1.py",
                "--in_path", str(input_path),
                "--out_clean", str(out_clean),
                "--out_report", str(out_report),
                "--out_summary", str(out_summary),
                "--seed", "42",
                "--min_words", "50",  # Lower threshold for test
                "--max_words", "500"
            ]
            
            with patch.object(sys, 'argv', args):
                main()
            
            # Verify outputs exist
            assert out_clean.exists()
            assert out_report.exists()
            assert out_summary.exists()
            
            # Verify clean dataset
            clean_records = load_dataset_stream(out_clean)
            assert len(clean_records) == 3  # 3 unique, 2 duplicates removed
            
            # Verify summary
            with open(out_summary, 'r') as f:
                summary = json.load(f)
            
            assert summary["counts"]["total_in"] == 5
            assert summary["counts"]["accepted"] == 3
            assert summary["counts"]["rejected"] == 2
            assert summary["counts"]["exact_dup_count"] == 2
            assert summary["rejection_reasons"]["exact_duplicate"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
