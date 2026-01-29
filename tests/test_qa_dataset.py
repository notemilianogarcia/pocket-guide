"""
Tests for QA dataset module.

Tests cover:
- Schema validation (valid/invalid records)
- Quality gate enforcement (pass/fail)
- Distribution computation
- Seeded sampling determinism
- QA report generation
"""

import json
import tempfile
from pathlib import Path

import pytest

from pocketguide.data_generation.qa_dataset import (
    load_dataset_stream,
    validate_record_schema,
    compute_distributions,
    compute_percentages,
    compute_coverage_metrics,
    sample_spotcheck,
    write_spotcheck_markdown,
    write_spotcheck_jsonl,
    write_qa_report,
    main,
)


class TestLoadDataset:
    """Test dataset loading."""
    
    def test_load_valid_dataset(self):
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
    
    def test_load_skips_blank_lines(self):
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


class TestValidateRecordSchema:
    """Test schema validation."""
    
    def test_valid_record(self):
        """Should validate correct record."""
        record = {
            "id": "test_1",
            "response": {
                "summary": "Test summary",
                "assumptions": [],
                "uncertainty_notes": "",
                "next_steps": [],
                "verification_steps": ["Check official source"],
                "payload_type": "checklist",
                "payload": {
                    "title": "Test Checklist",
                    "groups": [{
                        "name": "Documents",
                        "items": [{
                            "text": "Check passport",
                            "priority": "must"
                        }]
                    }]
                }
            }
        }
        
        is_valid, error = validate_record_schema(record)
        assert is_valid
        assert error is None
    
    def test_missing_response(self):
        """Should reject record without response."""
        record = {"id": "test_1"}
        
        is_valid, error = validate_record_schema(record)
        assert not is_valid
        assert "Missing 'response' field" in error
    
    def test_invalid_payload_type(self):
        """Should reject invalid payload_type."""
        record = {
            "id": "test_1",
            "response": {
                "summary": "Test",
                "assumptions": [],
                "uncertainty_notes": "",
                "next_steps": [],
                "verification_steps": [],
                "payload_type": "invalid_type",  # Not in enum
                "payload": {}
            }
        }
        
        is_valid, error = validate_record_schema(record)
        assert not is_valid
        assert "payload_type" in error.lower() or "not one of" in error.lower()


class TestComputeDistributions:
    """Test distribution computation."""
    
    def test_compute_distributions(self):
        """Should compute counts across dimensions."""
        records = [
            {
                "id": "1",
                "payload_type": "checklist",
                "category": "visa",
                "difficulty": "medium",
                "region_tags": ["asia", "southeast-asia"]
            },
            {
                "id": "2",
                "payload_type": "itinerary",
                "category": "visa",
                "difficulty": "easy",
                "region_tags": ["asia", "europe"]
            },
            {
                "id": "3",
                "payload_type": "checklist",
                "category": "safety",
                "difficulty": "medium",
                "region_tags": ["africa"]
            }
        ]
        
        distributions = compute_distributions(records)
        
        # Payload type
        assert distributions["payload_type"]["checklist"] == 2
        assert distributions["payload_type"]["itinerary"] == 1
        
        # Category
        assert distributions["category"]["visa"] == 2
        assert distributions["category"]["safety"] == 1
        
        # Difficulty
        assert distributions["difficulty"]["medium"] == 2
        assert distributions["difficulty"]["easy"] == 1
        
        # Region tags (individual counts)
        assert distributions["region_tag"]["asia"] == 2
        assert distributions["region_tag"]["southeast-asia"] == 1
        assert distributions["region_tag"]["europe"] == 1
        assert distributions["region_tag"]["africa"] == 1
    
    def test_compute_percentages(self):
        """Should compute percentages from counts."""
        distributions = {
            "category": {
                "visa": 7,
                "safety": 3
            }
        }
        
        percentages = compute_percentages(distributions, total=10)
        
        assert percentages["category"]["visa"] == 70.0
        assert percentages["category"]["safety"] == 30.0


class TestCoverageMetrics:
    """Test coverage metrics computation."""
    
    def test_coverage_metrics(self):
        """Should compute unique counts and min/max."""
        distributions = {
            "category": {
                "visa": 10,
                "safety": 5,
                "transportation": 15
            },
            "payload_type": {
                "checklist": 8,
                "itinerary": 8
            }
        }
        
        coverage = compute_coverage_metrics(distributions)
        
        # Category
        assert coverage["category"]["unique_count"] == 3
        assert coverage["category"]["min_count"] == 5
        assert coverage["category"]["max_count"] == 15
        assert coverage["category"]["min_key"] == "safety"
        assert coverage["category"]["max_key"] == "transportation"
        
        # Payload type
        assert coverage["payload_type"]["unique_count"] == 2
        assert coverage["payload_type"]["min_count"] == 8
        assert coverage["payload_type"]["max_count"] == 8


class TestSeededSampling:
    """Test spot-check sampling."""
    
    def test_sample_deterministic(self):
        """Should produce same samples with same seed."""
        records = [{"id": str(i)} for i in range(100)]
        
        sample1 = sample_spotcheck(records, sample_n=10, seed=42)
        sample2 = sample_spotcheck(records, sample_n=10, seed=42)
        
        ids1 = [s["id"] for s in sample1]
        ids2 = [s["id"] for s in sample2]
        
        assert ids1 == ids2
    
    def test_sample_different_seeds(self):
        """Should produce different samples with different seeds."""
        records = [{"id": str(i)} for i in range(100)]
        
        sample1 = sample_spotcheck(records, sample_n=10, seed=42)
        sample2 = sample_spotcheck(records, sample_n=10, seed=43)
        
        ids1 = [s["id"] for s in sample1]
        ids2 = [s["id"] for s in sample2]
        
        assert ids1 != ids2
    
    def test_sample_all_if_small_dataset(self):
        """Should return all records if sample_n >= dataset size."""
        records = [{"id": str(i)} for i in range(5)]
        
        sample = sample_spotcheck(records, sample_n=10, seed=42)
        
        assert len(sample) == 5


class TestSpotcheckOutputs:
    """Test spot-check output generation."""
    
    def test_write_spotcheck_markdown(self):
        """Should write markdown with expected structure."""
        samples = [
            {
                "id": "test_1",
                "category": "visa",
                "difficulty": "medium",
                "region_tags": ["asia"],
                "payload_type": "checklist",
                "prompt": "Do I need a visa?",
                "response": {
                    "summary": "Most visitors need a visa",
                    "assumptions": ["Tourist purpose"],
                    "uncertainty_notes": "Requirements may change",
                    "next_steps": ["Check embassy"],
                    "verification_steps": ["Visit embassy.gov"],
                    "payload": {"title": "Visa Checklist"}
                }
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            path = Path(f.name)
        
        try:
            write_spotcheck_markdown(samples, path, seed=42)
            
            content = path.read_text()
            
            # Check structure
            assert "# Dataset Spot-Check Report" in content
            assert "**Seed:** 42" in content
            assert "## Sample 1: test_1" in content
            assert "**Category:** visa" in content
            assert "**Prompt:**" in content
            assert "**Summary:**" in content
            assert "**Verification Steps:**" in content
        finally:
            path.unlink()
    
    def test_write_spotcheck_jsonl(self):
        """Should write JSONL with all samples."""
        samples = [
            {"id": "1", "category": "visa"},
            {"id": "2", "category": "safety"}
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            path = Path(f.name)
        
        try:
            write_spotcheck_jsonl(samples, path)
            
            written = load_dataset_stream(path)
            assert len(written) == 2
            assert written[0]["id"] == "1"
            assert written[1]["id"] == "2"
        finally:
            path.unlink()


class TestQAReport:
    """Test QA report generation."""
    
    def test_write_qa_report(self):
        """Should write comprehensive QA report."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            path = Path(f.name)
        
        distributions = {
            "category": {"visa": 7, "safety": 3}
        }
        percentages = {
            "category": {"visa": 70.0, "safety": 30.0}
        }
        coverage = {
            "category": {
                "unique_count": 2,
                "min_count": 3,
                "max_count": 7
            }
        }
        
        try:
            write_qa_report(
                output_path=path,
                total_samples=10,
                schema_valid_count=9,
                invalid_example_ids=["bad_1: missing field"],
                distributions=distributions,
                percentages=percentages,
                coverage=coverage,
                drift=None,
                sampled_ids=["1", "2", "3"],
                seed=42,
                gate_threshold=0.85,
                gate_passed=True
            )
            
            with open(path, "r") as f:
                report = json.load(f)
            
            # Check structure
            assert report["total_samples"] == 10
            assert report["schema_validation"]["valid_count"] == 9
            assert report["schema_validation"]["valid_rate"] == 0.9
            assert report["quality_gate"]["passed"] is True
            assert report["quality_gate"]["threshold"] == 0.85
            assert report["distributions"]["counts"]["category"]["visa"] == 7
            assert report["distributions"]["percentages"]["category"]["visa"] == 70.0
            assert report["spotcheck"]["sample_size"] == 3
        finally:
            path.unlink()


class TestQAMainIntegration:
    """Integration tests for main QA CLI."""
    
    def test_qa_main_gate_pass(self):
        """Should pass gate when validity >= threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create dataset with 9 valid, 1 invalid
            dataset_path = tmpdir / "dataset.jsonl"
            
            with open(dataset_path, "w") as f:
                # 9 valid records
                for i in range(9):
                    f.write(json.dumps({
                        "id": str(i),
                        "category": "visa",
                        "difficulty": "medium",
                        "region_tags": ["asia"],
                        "payload_type": "checklist",
                        "response": {
                            "summary": f"Summary {i}",
                            "assumptions": [],
                            "uncertainty_notes": "",
                            "next_steps": [],
                            "verification_steps": [],
                            "payload_type": "checklist",
                            "payload": {
                                "title": "Test",
                                "groups": [{
                                    "name": "Test",
                                    "items": [{"text": "Test", "priority": "must"}]
                                }]
                            }
                        }
                    }) + "\n")
                
                # 1 invalid record (missing response)
                f.write(json.dumps({"id": "bad", "category": "visa"}) + "\n")
            
            # Run QA with 0.85 threshold (9/10 = 0.9 should pass)
            import sys
            from io import StringIO
            from unittest.mock import patch
            
            args = [
                "qa_dataset.py",
                "--dataset", str(dataset_path),
                "--out_dir", str(tmpdir),
                "--seed", "42",
                "--sample_n", "5",
                "--min_schema_valid_rate", "0.85"
            ]
            
            with patch.object(sys, "argv", args):
                # Should not raise SystemExit or should exit with 0
                try:
                    main()
                except SystemExit as e:
                    assert e.code == 0, "QA should pass with 90% validity"
            
            # Verify outputs exist
            qa_report = tmpdir / "dataset_v1_qa.json"
            spotcheck_md = tmpdir / "spotcheck_v1_seed_42.md"
            spotcheck_jsonl = tmpdir / "spotcheck_v1_seed_42.jsonl"
            
            assert qa_report.exists()
            assert spotcheck_md.exists()
            assert spotcheck_jsonl.exists()
            
            # Verify report content
            with open(qa_report, "r") as f:
                report = json.load(f)
            
            assert report["total_samples"] == 10
            assert report["schema_validation"]["valid_count"] == 9
            assert report["quality_gate"]["passed"] is True
    
    def test_qa_main_gate_fail(self):
        """Should fail gate when validity < threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create dataset with 5 valid, 5 invalid
            dataset_path = tmpdir / "dataset.jsonl"
            
            with open(dataset_path, "w") as f:
                # 5 valid records
                for i in range(5):
                    f.write(json.dumps({
                        "id": str(i),
                        "category": "visa",
                        "payload_type": "checklist",
                        "response": {
                            "summary": f"Summary {i}",
                            "assumptions": [],
                            "uncertainty_notes": "",
                            "next_steps": [],
                            "verification_steps": [],
                            "payload_type": "checklist",
                            "payload": {
                                "title": "Test",
                                "groups": [{
                                    "name": "Test",
                                    "items": [{"text": "Test", "priority": "must"}]
                                }]
                            }
                        }
                    }) + "\n")
                
                # 5 invalid records (missing response)
                for i in range(5, 10):
                    f.write(json.dumps({"id": str(i), "category": "visa"}) + "\n")
            
            # Run QA with 0.85 threshold (5/10 = 0.5 should fail)
            import sys
            from unittest.mock import patch
            
            args = [
                "qa_dataset.py",
                "--dataset", str(dataset_path),
                "--out_dir", str(tmpdir),
                "--seed", "42",
                "--sample_n", "3",
                "--min_schema_valid_rate", "0.85"
            ]
            
            with patch.object(sys, "argv", args):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                
                # Should exit with non-zero code
                assert exc_info.value.code == 1
            
            # Verify outputs still exist
            qa_report = tmpdir / "dataset_v1_qa.json"
            assert qa_report.exists()
            
            # Verify report shows failure
            with open(qa_report, "r") as f:
                report = json.load(f)
            
            assert report["quality_gate"]["passed"] is False
    
    def test_qa_seeded_sampling_deterministic(self):
        """Should produce same spot-check samples with same seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create valid dataset
            dataset_path = tmpdir / "dataset.jsonl"
            
            with open(dataset_path, "w") as f:
                for i in range(50):
                    f.write(json.dumps({
                        "id": f"sample_{i}",
                        "category": "visa",
                        "payload_type": "checklist",
                        "response": {
                            "summary": f"Summary {i}",
                            "assumptions": [],
                            "uncertainty_notes": "",
                            "next_steps": [],
                            "verification_steps": [],
                            "payload_type": "checklist",
                            "payload": {
                                "title": "Test",
                                "groups": [{
                                    "name": "Test",
                                    "items": [{"text": "Test", "priority": "must"}]
                                }]
                            }
                        }
                    }) + "\n")
            
            # Run twice with same seed
            import sys
            from unittest.mock import patch
            
            for run in [1, 2]:
                out_dir = tmpdir / f"run{run}"
                
                args = [
                    "qa_dataset.py",
                    "--dataset", str(dataset_path),
                    "--out_dir", str(out_dir),
                    "--seed", "123",
                    "--sample_n", "10",
                    "--min_schema_valid_rate", "0.85"
                ]
                
                with patch.object(sys, "argv", args):
                    try:
                        main()
                    except SystemExit:
                        pass
            
            # Compare sampled IDs
            with open(tmpdir / "run1" / "dataset_v1_qa.json", "r") as f:
                report1 = json.load(f)
            
            with open(tmpdir / "run2" / "dataset_v1_qa.json", "r") as f:
                report2 = json.load(f)
            
            assert report1["spotcheck"]["sampled_ids"] == report2["spotcheck"]["sampled_ids"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
