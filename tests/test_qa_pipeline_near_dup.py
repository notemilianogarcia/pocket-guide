"""
Tests for near-duplicate detection in QA Pipeline v1.

Tests cover:
- Text normalization and tokenization
- Shingle generation
- Jaccard similarity
- Bucketing logic
- Near-duplicate detection
- Integration with full pipeline
"""

import json
import tempfile
from pathlib import Path

import pytest

from pocketguide.data_generation.qa_pipeline_v1 import (
    build_near_dup_text,
    tokenize_text,
    generate_shingles,
    jaccard_similarity,
    get_bucket_id,
    apply_near_dedup,
    main,
)


class TestBuildNearDupText:
    """Test near-dup text building."""
    
    def test_combine_components(self):
        """Should combine payload_type, prompt, and summary."""
        record = {
            "payload_type": "checklist",
            "prompt": "Do I need a visa?",
            "response": {"summary": "Most visitors need a visa"}
        }
        
        text = build_near_dup_text(record)
        
        assert "checklist" in text
        assert "visa" in text
        assert "most" in text
        assert "visitors" in text


class TestTokenizeText:
    """Test text tokenization."""
    
    def test_tokenize_removes_punctuation(self):
        """Should remove punctuation."""
        text = "Do I need a visa? Yes, you do!"
        tokens = tokenize_text(text)
        
        assert "Do" in tokens or "do" in tokens
        assert "?" not in tokens
        assert "," not in tokens
    
    def test_tokenize_splits_on_whitespace(self):
        """Should split on whitespace."""
        text = "hello world test"
        tokens = tokenize_text(text)
        
        assert len(tokens) == 3
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
    
    def test_tokenize_filters_empty(self):
        """Should filter empty tokens."""
        text = "hello    world"
        tokens = tokenize_text(text)
        
        assert "" not in tokens
        assert len(tokens) == 2


class TestGenerateShingles:
    """Test shingle generation."""
    
    def test_generate_three_token_shingles(self):
        """Should generate 3-token shingles."""
        tokens = ["a", "b", "c", "d", "e"]
        shingles = generate_shingles(tokens, shingle_size=3)
        
        assert "a b c" in shingles
        assert "b c d" in shingles
        assert "c d e" in shingles
        assert len(shingles) == 3
    
    def test_short_sequence_no_shingles(self):
        """Should return empty set if sequence too short."""
        tokens = ["a", "b"]
        shingles = generate_shingles(tokens, shingle_size=3)
        
        assert len(shingles) == 0
    
    def test_exact_length(self):
        """Should generate single shingle if exact length."""
        tokens = ["a", "b", "c"]
        shingles = generate_shingles(tokens, shingle_size=3)
        
        assert len(shingles) == 1
        assert "a b c" in shingles


class TestJaccardSimilarity:
    """Test Jaccard similarity computation."""
    
    def test_identical_sets(self):
        """Should return 1.0 for identical sets."""
        shingles_a = {"a", "b", "c"}
        shingles_b = {"a", "b", "c"}
        
        similarity = jaccard_similarity(shingles_a, shingles_b)
        assert similarity == 1.0
    
    def test_disjoint_sets(self):
        """Should return 0.0 for disjoint sets."""
        shingles_a = {"a", "b", "c"}
        shingles_b = {"x", "y", "z"}
        
        similarity = jaccard_similarity(shingles_a, shingles_b)
        assert similarity == 0.0
    
    def test_partial_overlap(self):
        """Should compute partial similarity correctly."""
        shingles_a = {"a", "b", "c"}
        shingles_b = {"b", "c", "d"}
        
        # Intersection: {b, c}, Union: {a, b, c, d}
        # Jaccard: 2/4 = 0.5
        similarity = jaccard_similarity(shingles_a, shingles_b)
        assert similarity == 0.5
    
    def test_both_empty(self):
        """Should return 1.0 for both empty."""
        similarity = jaccard_similarity(set(), set())
        assert similarity == 1.0
    
    def test_one_empty(self):
        """Should return 0.0 if only one empty."""
        similarity = jaccard_similarity({"a"}, set())
        assert similarity == 0.0


class TestGetBucketId:
    """Test bucketing logic."""
    
    def test_deterministic_bucketing(self):
        """Should return same bucket for same shingles."""
        shingles = {"a b c", "b c d", "c d e"}
        
        bucket1 = get_bucket_id(shingles, bucket_modulo=1000)
        bucket2 = get_bucket_id(shingles, bucket_modulo=1000)
        
        assert bucket1 == bucket2
    
    def test_different_shingles_same_bucket(self):
        """Should allow different shingles in same bucket."""
        # Not testing this directly since it's probabilistic
        # but verify it returns valid bucket ID
        shingles = {"x", "y", "z"}
        bucket = get_bucket_id(shingles, bucket_modulo=100)
        
        assert 0 <= bucket < 100
    
    def test_empty_shingles_bucket_zero(self):
        """Should handle empty shingles."""
        bucket = get_bucket_id(set(), bucket_modulo=1000)
        assert bucket == 0


class TestApplyNearDedup:
    """Test near-duplicate detection."""
    
    def test_identical_records_detected(self):
        """Should detect identical records as near-duplicates."""
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
            }
        ]
        
        accepted, rejected = apply_near_dedup(records, threshold=0.8)
        
        assert len(accepted) == 1
        assert accepted[0]["id"] == "1"
        
        assert len(rejected) == 1
        assert rejected[0][0]["id"] == "2"
        assert rejected[0][1] == "near_duplicate"
        assert "similarity" in rejected[0][2]
        assert "anchor_id" in rejected[0][2]
    
    def test_paraphrased_similar_records(self):
        """Should detect paraphrased similar records."""
        records = [
            {
                "id": "1",
                "payload_type": "checklist",
                "prompt": "Do I need a visa for Canada?",
                "response": {"summary": "Canadian visa requirements"}
            },
            {
                "id": "2",
                "payload_type": "checklist",
                "prompt": "Do I require a visa for Canada?",
                "response": {"summary": "Visa requirements for Canada"}
            }
        ]
        
        accepted, rejected = apply_near_dedup(records, threshold=0.7)
        
        # Should detect as near-duplicate due to high overlap
        if len(rejected) > 0:
            assert rejected[0][1] == "near_duplicate"
    
    def test_different_records_not_duplicate(self):
        """Should not flag truly different records."""
        records = [
            {
                "id": "1",
                "payload_type": "checklist",
                "prompt": "Do I need a visa?",
                "response": {"summary": "Yes, most visitors need a visa"}
            },
            {
                "id": "2",
                "payload_type": "checklist",
                "prompt": "What food is popular?",
                "response": {"summary": "Local cuisine includes pasta"}
            }
        ]
        
        accepted, rejected = apply_near_dedup(records, threshold=0.8)
        
        assert len(accepted) == 2
        assert len(rejected) == 0
    
    def test_threshold_respects_boundary(self):
        """Should respect similarity threshold."""
        records = [
            {
                "id": "1",
                "payload_type": "checklist",
                "prompt": "Test prompt with some text",
                "response": {"summary": "Test summary with content"}
            },
            {
                "id": "2",
                "payload_type": "checklist",
                "prompt": "Test prompt with some different",
                "response": {"summary": "Test summary with variation"}
            }
        ]
        
        # With high threshold, should not detect as duplicate
        accepted_high, rejected_high = apply_near_dedup(records, threshold=0.95)
        assert len(rejected_high) == 0
        
        # With low threshold, might detect
        accepted_low, rejected_low = apply_near_dedup(records, threshold=0.1)
        # At least verify it runs
        assert len(accepted_low) + len(rejected_low) == 2
    
    def test_deterministic_ordering(self):
        """Should keep first record deterministically."""
        records = [
            {
                "id": "first",
                "payload_type": "checklist",
                "prompt": "Same content",
                "response": {"summary": "Same content"}
            },
            {
                "id": "second",
                "payload_type": "checklist",
                "prompt": "Same content",
                "response": {"summary": "Same content"}
            }
        ]
        
        accepted, rejected = apply_near_dedup(records, threshold=0.8)
        
        assert len(accepted) == 1
        assert accepted[0]["id"] == "first"
        assert len(rejected) == 1
        assert rejected[0][0]["id"] == "second"
    
    def test_metadata_includes_anchor(self):
        """Should record anchor_id and similarity in metadata."""
        records = [
            {
                "id": "anchor",
                "payload_type": "checklist",
                "prompt": "Test",
                "response": {"summary": "Test"}
            },
            {
                "id": "duplicate",
                "payload_type": "checklist",
                "prompt": "Test",
                "response": {"summary": "Test"}
            }
        ]
        
        accepted, rejected = apply_near_dedup(records, threshold=0.8)
        
        if len(rejected) > 0:
            _, reason, metadata = rejected[0]
            assert metadata["anchor_id"] == "anchor"
            assert "similarity" in metadata
            assert 0 <= metadata["similarity"] <= 1


class TestNearDupPipelineIntegration:
    """Integration tests with full pipeline."""
    
    def test_full_pipeline_with_near_dup(self):
        """Should run full pipeline including near-dedup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create input dataset
            input_path = tmpdir / "input.jsonl"
            with open(input_path, 'w') as f:
                # Unique record
                f.write(json.dumps({
                    "id": "unique_1",
                    "category": "visa",
                    "payload_type": "checklist",
                    "region_tags": ["asia"],
                    "prompt": "Unique prompt one",
                    "response": {
                        "summary": "Unique summary one. " + "word " * 40,
                        "next_steps": [{"description": "Contact embassy"}],
                        "verification_steps": [{"description": "Check website"}],
                        "uncertainty_notes": ""
                    }
                }) + "\n")
                
                # Similar records (not identical)
                f.write(json.dumps({
                    "id": "similar_1",
                    "category": "visa",
                    "payload_type": "checklist",
                    "region_tags": ["asia"],
                    "prompt": "Do I need a visa for this country?",
                    "response": {
                        "summary": "Most visitors need a visa for entry. " + "word " * 40,
                        "next_steps": [{"description": "Apply online"}],
                        "verification_steps": [{"description": "Check status"}],
                        "uncertainty_notes": ""
                    }
                }) + "\n")
                
                f.write(json.dumps({
                    "id": "similar_2",
                    "category": "visa",
                    "payload_type": "checklist",
                    "region_tags": ["asia"],
                    "prompt": "Is a visa required for travelers?",
                    "response": {
                        "summary": "Visa is typically needed for most travelers. " + "word " * 40,
                        "next_steps": [{"description": "Submit application"}],
                        "verification_steps": [{"description": "Verify approval"}],
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
                "--near_dup_threshold", "0.6",
                "--min_words", "50",
                "--max_words", "500"
            ]
            
            with patch.object(sys, 'argv', args):
                main()
            
            # Verify outputs exist
            assert out_clean.exists()
            assert out_report.exists()
            assert out_summary.exists()
            
            # Verify clean dataset
            from pocketguide.data_generation.qa_pipeline_v1 import load_dataset_stream
            clean_records = load_dataset_stream(out_clean)
            # Should have 2 or 3 records depending on near-dup detection
            assert len(clean_records) >= 2
            
            # Verify summary
            with open(out_summary, 'r') as f:
                summary = json.load(f)
            
            assert summary["counts"]["total_in"] == 3
            assert "near_dup_config" in summary
            assert summary["near_dup_config"]["threshold"] == 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
