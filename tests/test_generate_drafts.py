"""Tests for draft generation pass (Lesson 3.3)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pocketguide.data_generation.generate_drafts import (
    build_draft_record_from_parse,
    build_teacher_request,
    compute_pass_rates,
    generate_drafts,
    hash_file,
    load_prompt_plan,
    load_teacher_config,
)
from pocketguide.teachers.base import TeacherResponse


class TestHashFile:
    """Test file hashing for reproducibility."""

    def test_hash_file_deterministic(self):
        """Test that hash is deterministic."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()
            temp_path = Path(f.name)

        try:
            hash1 = hash_file(temp_path)
            hash2 = hash_file(temp_path)
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex is 64 chars
        finally:
            temp_path.unlink()


class TestLoadPromptPlan:
    """Test prompt plan loading and validation."""

    def test_load_valid_prompt_plan(self):
        """Test loading valid prompt plan."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(3):
                record = {
                    "id": f"test_{i}",
                    "payload_type": "itinerary",
                    "category": "planning",
                    "difficulty": "easy",
                    "prompt": f"Test prompt {i}",
                    "template_version": "v1",
                    "template_name": "itinerary_draft",
                    "seed": 42,
                }
                f.write(json.dumps(record) + "\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            prompts = load_prompt_plan(temp_path)
            assert len(prompts) == 3
            assert prompts[0]["id"] == "test_0"
            assert prompts[1]["payload_type"] == "itinerary"
        finally:
            temp_path.unlink()

    def test_load_missing_file(self):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            load_prompt_plan(Path("/nonexistent/path.jsonl"))

    def test_load_missing_required_field(self):
        """Test error on missing required field."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Missing 'payload_type'
            record = {
                "id": "test_0",
                "prompt": "Test prompt",
                "template_version": "v1",
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Missing required field"):
                load_prompt_plan(temp_path)
        finally:
            temp_path.unlink()

    def test_load_skips_blank_lines(self):
        """Test that blank lines are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"id": "test_0", "payload_type": "itinerary", "prompt": "test", "template_version": "v1"}) + "\n")
            f.write("\n")  # Blank line
            f.write(json.dumps({"id": "test_1", "payload_type": "checklist", "prompt": "test", "template_version": "v1"}) + "\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            prompts = load_prompt_plan(temp_path)
            assert len(prompts) == 2
        finally:
            temp_path.unlink()


class TestBuildTeacherRequest:
    """Test TeacherRequest building."""

    def test_build_request_with_config(self):
        """Test building request with generation config."""
        prompt_record = {
            "id": "test_0",
            "payload_type": "itinerary",
            "prompt": "Plan a 3-day Paris trip",
            "template_version": "v1",
        }

        config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 500,
            "seed": 42,
        }

        request = build_teacher_request(prompt_record, config)

        assert len(request.messages) == 2
        assert request.messages[0]["role"] == "system"
        assert request.messages[1]["role"] == "user"
        assert request.messages[1]["content"] == "Plan a 3-day Paris trip"
        assert request.temperature == 0.3
        assert request.top_p == 0.95
        assert request.max_tokens == 500
        assert request.seed == 42
        assert request.metadata["prompt_id"] == "test_0"


class TestBuildDraftRecord:
    """Test draft record building."""

    def test_build_record_success(self):
        """Test building draft record with successful parsing."""
        from pocketguide.eval.parsing import ParseResult

        prompt_record = {
            "id": "test_0",
            "payload_type": "itinerary",
            "category": "planning",
            "difficulty": "medium",
            "region_tags": ["europe", "france"],
            "template_version": "v1",
            "template_name": "itinerary_draft",
            "seed": 42,
            "prompt": "Plan a trip",
        }

        teacher_response = TeacherResponse(
            text='{"summary": "test"}',
            model="test-model",
            provider="openrouter",
            request_id="req-123",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            timing={"latency_s": 0.5},
            raw={"selected_model": "test-model", "attempted_models": ["test-model"]},
        )

        parse_result = ParseResult(
            success=True,
            data={"payload_type": "itinerary", "summary": "test"}
        )

        draft = build_draft_record_from_parse(prompt_record, teacher_response, parse_result)

        assert draft["id"] == "test_0"
        assert draft["prompt_plan"]["payload_type"] == "itinerary"
        assert draft["prompt_plan"]["region_tags"] == ["europe", "france"]
        assert draft["teacher"]["provider"] == "openrouter"
        assert draft["teacher"]["usage"]["total_tokens"] == 30
        assert draft["output_text"] == '{"summary": "test"}'
        assert draft["contract"]["overall_ok"] is True
        assert "created_at" in draft

    def test_build_record_with_error(self):
        """Test building draft record with parsing error."""
        from pocketguide.eval.parsing import ParseResult, ParseValidationError

        prompt_record = {
            "id": "test_0",
            "payload_type": "itinerary",
            "prompt": "Test",
            "template_version": "v1",
        }

        teacher_response = TeacherResponse(
            text="invalid json",
            model="test-model",
            provider="openrouter",
            request_id="req-123",
            usage=None,
            timing={"latency_s": 0.5},
            raw={"selected_model": "test-model"},
        )

        error = ParseValidationError(
            code="JSON_PARSE_ERROR",
            error_type="json_parse",
            message="Invalid JSON",
            failed_at="strict",
        )
        parse_result = ParseResult(success=False, error=error)

        draft = build_draft_record_from_parse(prompt_record, teacher_response, parse_result)

        assert draft["contract"]["overall_ok"] is False
        assert draft["contract"]["error"]["code"] == "JSON_PARSE_ERROR"


class TestComputePassRates:
    """Test pass rate computation."""

    def test_compute_rates(self):
        """Test computing pass rate percentages."""
        stats = {
            "processed": 100,
            "strict_json_pass": 90,
            "lenient_json_pass": 95,
            "envelope_pass": 85,
            "payload_pass": 80,
            "overall_pass": 75,
        }

        rates = compute_pass_rates(stats)

        assert rates["strict_json_rate"] == 90.0
        assert rates["lenient_json_rate"] == 95.0
        assert rates["envelope_rate"] == 85.0
        assert rates["payload_rate"] == 80.0
        assert rates["overall_rate"] == 75.0

    def test_compute_rates_zero_processed(self):
        """Test with no processed records."""
        stats = {
            "processed": 0,
            "strict_json_pass": 0,
            "lenient_json_pass": 0,
            "envelope_pass": 0,
            "payload_pass": 0,
            "overall_pass": 0,
        }

        rates = compute_pass_rates(stats)

        assert all(v == 0 for v in rates.values())


class TestGenerateDrafts:
    """Test the main draft generation function."""

    def test_generate_drafts_dry_run(self):
        """Test draft generation with dry-run mode (no network)."""
        # Create temporary directories
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create prompt plan
            plan_path = tmpdir_path / "prompt_plan.jsonl"
            with open(plan_path, "w") as f:
                for i in range(3):
                    record = {
                        "id": f"test_{i}",
                        "payload_type": "itinerary" if i % 2 == 0 else "checklist",
                        "category": "planning",
                        "difficulty": "easy",
                        "prompt": f"Test prompt {i}",
                        "template_version": "v1",
                        "template_name": "test",
                    }
                    f.write(json.dumps(record) + "\n")

            # Create config
            config = {
                "models": ["test-model-1", "test-model-2"],
                "generation": {
                    "temperature": 0.2,
                    "max_tokens": 100,
                },
                "runtime": {"dry_run": True},
                "rate_limit": {"rpm": 15},
            }

            # Mock teacher
            mock_teacher = Mock()

            def mock_generate(request):
                return TeacherResponse(
                    text='{"summary": "test", "sections": []}',
                    model="test-model-1",
                    provider="test",
                    request_id="req-123",
                    usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                    timing={"latency_s": 0.1},
                    raw={"selected_model": "test-model-1", "attempted_models": ["test-model-1"]},
                )

            mock_teacher.generate = mock_generate

            # Run generation
            stats = generate_drafts(
                plan_path=plan_path,
                out_dir=tmpdir_path,
                teacher=mock_teacher,
                config=config,
            )

            # Verify outputs exist
            assert (tmpdir_path / "drafts_v1.jsonl").exists()
            assert (tmpdir_path / "drafts_v1_manifest.json").exists()
            assert (tmpdir_path / "drafts_v1_stats.json").exists()

            # Verify stats
            assert stats["total"] == 3
            assert stats["processed"] == 3
            assert stats["skipped"] == 0

            # Verify draft records
            drafts = []
            with open(tmpdir_path / "drafts_v1.jsonl") as f:
                for line in f:
                    if line.strip():
                        drafts.append(json.loads(line))

            assert len(drafts) == 3
            for draft in drafts:
                assert "id" in draft
                assert "prompt_plan" in draft
                assert "teacher" in draft
                assert "output_text" in draft
                assert "contract" in draft

            # Verify manifest
            with open(tmpdir_path / "drafts_v1_manifest.json") as f:
                manifest = json.load(f)
                assert manifest["version"] == "v1"
                assert manifest["counts"]["processed"] == 3

    def test_generate_drafts_with_limit(self):
        """Test limiting number of records to process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create prompt plan with 5 records
            plan_path = tmpdir_path / "prompt_plan.jsonl"
            with open(plan_path, "w") as f:
                for i in range(5):
                    record = {
                        "id": f"test_{i}",
                        "payload_type": "itinerary",
                        "category": "planning",
                        "difficulty": "easy",
                        "prompt": f"Test prompt {i}",
                        "template_version": "v1",
                        "template_name": "test",
                    }
                    f.write(json.dumps(record) + "\n")

            config = {
                "models": ["test-model"],
                "generation": {"temperature": 0.2},
                "runtime": {"dry_run": True},
                "rate_limit": {"rpm": 15},
            }

            mock_teacher = Mock()
            mock_teacher.generate = Mock(
                return_value=TeacherResponse(
                    text='{"summary": "test"}',
                    model="test-model",
                    provider="test",
                    request_id="req-123",
                    usage=None,
                    timing={"latency_s": 0.1},
                    raw={"selected_model": "test-model", "attempted_models": ["test-model"]},
                )
            )

            # Generate with limit
            stats = generate_drafts(
                plan_path=plan_path,
                out_dir=tmpdir_path,
                teacher=mock_teacher,
                config=config,
                limit=2,
            )

            assert stats["total"] == 2  # Limited to 2
            assert stats["processed"] == 2

    def test_generate_drafts_resume(self):
        """Test resume functionality skips existing records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create prompt plan
            plan_path = tmpdir_path / "prompt_plan.jsonl"
            with open(plan_path, "w") as f:
                for i in range(3):
                    record = {
                        "id": f"test_{i}",
                        "payload_type": "itinerary",
                        "category": "planning",
                        "difficulty": "easy",
                        "prompt": f"Test prompt {i}",
                        "template_version": "v1",
                        "template_name": "test",
                    }
                    f.write(json.dumps(record) + "\n")

            # Create existing drafts file with one record
            drafts_path = tmpdir_path / "drafts_v1.jsonl"
            with open(drafts_path, "w") as f:
                existing = {
                    "id": "test_0",
                    "prompt_plan": {"payload_type": "itinerary"},
                    "contract": {"overall_ok": True},
                }
                f.write(json.dumps(existing) + "\n")

            config = {
                "models": ["test-model"],
                "generation": {"temperature": 0.2},
                "runtime": {"dry_run": True},
                "rate_limit": {"rpm": 15},
            }

            mock_teacher = Mock()
            mock_teacher.generate = Mock(
                return_value=TeacherResponse(
                    text='{"summary": "test"}',
                    model="test-model",
                    provider="test",
                    request_id="req-123",
                    usage=None,
                    timing={"latency_s": 0.1},
                    raw={"selected_model": "test-model", "attempted_models": ["test-model"]},
                )
            )

            # Generate with resume
            stats = generate_drafts(
                plan_path=plan_path,
                out_dir=tmpdir_path,
                teacher=mock_teacher,
                config=config,
                resume=True,
            )

            assert stats["total"] == 3
            assert stats["skipped"] == 1
            assert stats["processed"] == 2

            # Verify file has 3 records (1 existing + 2 new)
            with open(drafts_path) as f:
                records = [json.loads(line) for line in f if line.strip()]
            assert len(records) == 3
