"""Tests for critique generation (Lesson 3.4)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from pocketguide.data_generation.generate_critiques import (
    build_critique_prompt,
    build_critique_record,
    build_critique_request,
    generate_critiques,
    hash_file,
    load_critique_schema,
    parse_critique_json,
    validate_critique_schema,
)
from pocketguide.teachers.base import TeacherResponse


class TestHashFile:
    """Test file hashing for reproducibility."""

    def test_hash_file_deterministic(self):
        """Test that hash is deterministic."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content for critique")
            f.flush()
            temp_path = Path(f.name)

        try:
            hash1 = hash_file(temp_path)
            hash2 = hash_file(temp_path)
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex is 64 chars
        finally:
            temp_path.unlink()


class TestCritiqueSchema:
    """Test critique schema validation."""

    def test_load_critique_schema(self):
        """Test loading critique schema."""
        schema = load_critique_schema()
        assert "$schema" in schema
        assert schema["type"] == "object"
        assert "id" in schema["required"]
        assert "verdict" in schema["required"]

    def test_valid_critique_passes_schema(self):
        """Test valid critique passes schema validation."""
        schema = load_critique_schema()
        
        valid_critique = {
            "id": "test_0",
            "verdict": "pass",
            "issues": [],
            "scores": {
                "actionability": 5,
                "clarity": 5,
                "schema_compliance": 5,
                "safety_risk": 5,
            },
            "hallucination": {
                "risk_level": "low",
                "risky_claims": [],
                "rationale": "No risky claims detected",
            },
            "verification": {
                "missing_when_needed": False,
                "suggested_steps": [],
            },
            "schema": {
                "envelope_ok": True,
                "payload_ok": True,
                "notes": "Schema compliant",
                "missing_fields": [],
            },
            "rewrite_instructions": [],
        }
        
        valid, error = validate_critique_schema(valid_critique, schema)
        assert valid is True
        assert error is None

    def test_invalid_critique_fails_schema(self):
        """Test invalid critique fails schema validation."""
        schema = load_critique_schema()
        
        # Missing required field
        invalid_critique = {
            "id": "test_0",
            "verdict": "pass",
            # Missing issues, scores, hallucination, etc.
        }
        
        valid, error = validate_critique_schema(invalid_critique, schema)
        assert valid is False
        assert error is not None

    def test_invalid_verdict_enum(self):
        """Test invalid verdict enum value fails."""
        schema = load_critique_schema()
        
        invalid_critique = {
            "id": "test_0",
            "verdict": "maybe",  # Invalid enum
            "issues": [],
            "scores": {
                "actionability": 5,
                "clarity": 5,
                "schema_compliance": 5,
                "safety_risk": 5,
            },
            "hallucination": {
                "risk_level": "low",
                "risky_claims": [],
                "rationale": "Test",
            },
            "verification": {
                "missing_when_needed": False,
                "suggested_steps": [],
            },
            "schema": {
                "envelope_ok": True,
                "payload_ok": True,
                "notes": "Test",
                "missing_fields": [],
            },
            "rewrite_instructions": [],
        }
        
        valid, error = validate_critique_schema(invalid_critique, schema)
        assert valid is False


class TestParseCritiqueJson:
    """Test critique JSON parsing."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        json_text = '{"id": "test_0", "verdict": "pass"}'
        success, data, error = parse_critique_json(json_text)
        assert success is True
        assert data["id"] == "test_0"
        assert error is None

    def test_parse_json_in_markdown(self):
        """Test parsing JSON wrapped in markdown code fence."""
        json_text = """Here is the critique:
```json
{"id": "test_0", "verdict": "pass"}
```
"""
        success, data, error = parse_critique_json(json_text)
        assert success is True
        assert data["id"] == "test_0"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns error."""
        json_text = "This is not JSON at all"
        success, data, error = parse_critique_json(json_text)
        assert success is False
        assert data is None
        assert error is not None


class TestBuildCritiquePrompt:
    """Test critique prompt building."""

    def test_build_critique_prompt(self):
        """Test building critique prompt from draft."""
        draft = {
            "id": "test_0",
            "prompt": "Plan a trip to Paris",
            "output_text": '{"summary": "Paris trip plan"}',
            "contract": {
                "overall_ok": True,
                "strict_json_ok": True,
            },
        }
        
        template = "PROMPT: {prompt}\nOUTPUT: {output_text}\nCONTRACT: {contract_json}\nID: {draft_id}"
        
        prompt = build_critique_prompt(draft, template)
        
        assert "Plan a trip to Paris" in prompt
        assert '{"summary": "Paris trip plan"}' in prompt
        assert "test_0" in prompt
        assert "overall_ok" in prompt


class TestBuildCritiqueRequest:
    """Test critique request building."""

    def test_build_request(self):
        """Test building TeacherRequest for critique."""
        config = {
            "generation": {
                "temperature": 0.3,
                "top_p": 0.95,
                "max_tokens": 1500,
            },
            "seed": 42,
        }
        
        critique_prompt = "Evaluate this response..."
        
        request = build_critique_request(critique_prompt, config)
        
        assert len(request.messages) == 2
        assert request.messages[0]["role"] == "system"
        assert request.messages[1]["role"] == "user"
        assert request.messages[1]["content"] == critique_prompt
        assert request.temperature == 0.3
        assert request.max_tokens == 1500


class TestBuildCritiqueRecord:
    """Test critique record building."""

    def test_build_record_with_valid_critique(self):
        """Test building record with valid critique."""
        draft = {"id": "test_0"}
        
        teacher_response = TeacherResponse(
            text='{"id": "test_0", "verdict": "pass"}',
            model="test-model",
            provider="test",
            request_id="req-123",
            usage={"total_tokens": 100},
            timing={"latency_s": 1.0},
            raw={"selected_model": "test-model", "attempted_models": ["test-model"]},
        )
        
        critique_data = {"id": "test_0", "verdict": "pass"}
        critique_contract = {"strict_json_ok": True, "schema_ok": True}
        
        record = build_critique_record(
            draft=draft,
            teacher_response=teacher_response,
            critique_data=critique_data,
            critique_contract=critique_contract,
            raw_critique_text=teacher_response.text,
        )
        
        assert record["id"] == "test_0"
        assert record["draft_id"] == "test_0"
        assert record["critique"] == critique_data
        assert record["raw_critique_text"] is None  # Not stored when valid
        assert record["teacher"]["provider"] == "test"
        assert record["critique_contract"]["strict_json_ok"] is True

    def test_build_record_with_invalid_critique(self):
        """Test building record with invalid critique."""
        draft = {"id": "test_0"}
        
        teacher_response = TeacherResponse(
            text="This is not JSON",
            model="test-model",
            provider="test",
            request_id="req-123",
            usage=None,
            timing={"latency_s": 1.0},
            raw={"selected_model": "test-model"},
        )
        
        critique_contract = {
            "strict_json_ok": False,
            "schema_ok": False,
            "error": {"code": "JSON_PARSE_ERROR", "message": "Failed to parse"},
        }
        
        record = build_critique_record(
            draft=draft,
            teacher_response=teacher_response,
            critique_data=None,
            critique_contract=critique_contract,
            raw_critique_text=teacher_response.text,
        )
        
        assert record["critique"] is None
        assert record["raw_critique_text"] == "This is not JSON"
        assert record["critique_contract"]["strict_json_ok"] is False

    def test_truncate_long_raw_text(self):
        """Test that very long raw text is truncated."""
        draft = {"id": "test_0"}
        
        # Create text longer than MAX_CRITIQUE_TEXT_LENGTH (20000)
        long_text = "x" * 25000
        
        teacher_response = TeacherResponse(
            text=long_text,
            model="test-model",
            provider="test",
            request_id="req-123",
            usage=None,
            timing={"latency_s": 1.0},
            raw={},
        )
        
        critique_contract = {"strict_json_ok": False, "schema_ok": False}
        
        record = build_critique_record(
            draft=draft,
            teacher_response=teacher_response,
            critique_data=None,
            critique_contract=critique_contract,
            raw_critique_text=long_text,
        )
        
        assert record["raw_critique_text"].endswith("...[truncated]")
        assert len(record["raw_critique_text"]) <= 20020  # MAX + suffix


class TestGenerateCritiques:
    """Test end-to-end critique generation."""

    def test_generate_critiques_with_mock(self):
        """Test critique generation with mocked teacher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create sample drafts file
            drafts_path = tmpdir_path / "drafts.jsonl"
            drafts = [
                {
                    "id": "draft_0",
                    "prompt": "Best cafes in Rome?",
                    "output_text": '{"summary": "Rome cafes"}',
                    "contract": {"overall_ok": True},
                },
                {
                    "id": "draft_1",
                    "prompt": "Plan Paris trip",
                    "output_text": "Not valid JSON",
                    "contract": {"overall_ok": False},
                },
            ]
            
            with open(drafts_path, "w") as f:
                for draft in drafts:
                    f.write(json.dumps(draft) + "\n")
            
            # Load schema and template
            schema = load_critique_schema()
            template = "PROMPT: {prompt}\nOUTPUT: {output_text}\nID: {draft_id}\nCONTRACT: {contract_json}"
            
            # Mock teacher
            mock_teacher = Mock()
            
            def mock_generate(request):
                # Return valid critique for first, invalid for second
                if "draft_0" in request.messages[1]["content"]:
                    critique_json = {
                        "id": "draft_0",
                        "verdict": "pass",
                        "issues": [],
                        "scores": {
                            "actionability": 5,
                            "clarity": 5,
                            "schema_compliance": 5,
                            "safety_risk": 5,
                        },
                        "hallucination": {
                            "risk_level": "low",
                            "risky_claims": [],
                            "rationale": "No issues",
                        },
                        "verification": {
                            "missing_when_needed": False,
                            "suggested_steps": [],
                        },
                        "schema": {
                            "envelope_ok": True,
                            "payload_ok": True,
                            "notes": "Good",
                            "missing_fields": [],
                        },
                        "rewrite_instructions": [],
                    }
                    text = json.dumps(critique_json)
                else:
                    text = "This is not valid JSON"
                
                return TeacherResponse(
                    text=text,
                    model="test-model",
                    provider="test",
                    request_id=f"req-{id(request)}",
                    usage={"total_tokens": 100},
                    timing={"latency_s": 0.5},
                    raw={"selected_model": "test-model", "attempted_models": ["test-model"]},
                )
            
            mock_teacher.generate = mock_generate
            
            config = {
                "models": ["test-model"],
                "generation": {"temperature": 0.2, "max_tokens": 1500},
                "seed": 42,
            }
            
            # Run generation
            stats = generate_critiques(
                drafts_path=drafts_path,
                out_dir=tmpdir_path,
                teacher=mock_teacher,
                config=config,
                schema=schema,
                template=template,
            )
            
            # Verify outputs exist
            assert (tmpdir_path / "critiques_v1.jsonl").exists()
            assert (tmpdir_path / "critiques_v1_manifest.json").exists()
            assert (tmpdir_path / "critiques_v1_stats.json").exists()
            
            # Verify stats
            assert stats["total"] == 2
            assert stats["processed"] == 2
            assert stats["critique_parse_ok"] == 1
            assert stats["critique_schema_ok"] == 1
            assert stats["verdicts"]["pass"] == 1
            
            # Verify critique records
            critiques = []
            with open(tmpdir_path / "critiques_v1.jsonl") as f:
                for line in f:
                    if line.strip():
                        critiques.append(json.loads(line))
            
            assert len(critiques) == 2
            
            # First should have valid critique
            assert critiques[0]["draft_id"] == "draft_0"
            assert critiques[0]["critique"] is not None
            assert critiques[0]["critique"]["verdict"] == "pass"
            assert critiques[0]["critique_contract"]["schema_ok"] is True
            
            # Second should have failed critique
            assert critiques[1]["draft_id"] == "draft_1"
            assert critiques[1]["critique"] is None
            assert critiques[1]["raw_critique_text"] == "This is not valid JSON"
            assert critiques[1]["critique_contract"]["strict_json_ok"] is False

    def test_generate_critiques_with_limit(self):
        """Test critique generation with limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create sample drafts
            drafts_path = tmpdir_path / "drafts.jsonl"
            drafts = [
                {"id": f"draft_{i}", "prompt": "Test", "output_text": "{}", "contract": {}}
                for i in range(5)
            ]
            
            with open(drafts_path, "w") as f:
                for draft in drafts:
                    f.write(json.dumps(draft) + "\n")
            
            schema = load_critique_schema()
            template = "ID: {draft_id}"
            
            mock_teacher = Mock()
            mock_teacher.generate = Mock(
                return_value=TeacherResponse(
                    text='{"id": "x", "verdict": "pass", "issues": [], "scores": {"actionability": 5, "clarity": 5, "schema_compliance": 5, "safety_risk": 5}, "hallucination": {"risk_level": "low", "risky_claims": [], "rationale": ""}, "verification": {"missing_when_needed": false, "suggested_steps": []}, "schema": {"envelope_ok": true, "payload_ok": true, "notes": "", "missing_fields": []}, "rewrite_instructions": []}',
                    model="test",
                    provider="test",
                    request_id="req",
                    usage={},
                    timing={},
                    raw={},
                )
            )
            
            config = {"generation": {}, "seed": 42}
            
            stats = generate_critiques(
                drafts_path=drafts_path,
                out_dir=tmpdir_path,
                teacher=mock_teacher,
                config=config,
                schema=schema,
                template=template,
                limit=3,
            )
            
            assert stats["total"] == 3
            assert stats["processed"] == 3

    def test_generate_critiques_resume(self):
        """Test critique generation resume mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create drafts
            drafts_path = tmpdir_path / "drafts.jsonl"
            drafts = [
                {"id": "draft_0", "prompt": "Test 0", "output_text": "{}", "contract": {}},
                {"id": "draft_1", "prompt": "Test 1", "output_text": "{}", "contract": {}},
            ]
            
            with open(drafts_path, "w") as f:
                for draft in drafts:
                    f.write(json.dumps(draft) + "\n")
            
            # Create existing critiques (draft_0 already done)
            critiques_path = tmpdir_path / "critiques_v1.jsonl"
            existing_critique = {
                "draft_id": "draft_0",
                "critique": {"verdict": "pass"},
                "critique_contract": {"strict_json_ok": True},
            }
            with open(critiques_path, "w") as f:
                f.write(json.dumps(existing_critique) + "\n")
            
            schema = load_critique_schema()
            template = "ID: {draft_id}"
            
            mock_teacher = Mock()
            mock_teacher.generate = Mock(
                return_value=TeacherResponse(
                    text='{"id": "x", "verdict": "pass", "issues": [], "scores": {"actionability": 5, "clarity": 5, "schema_compliance": 5, "safety_risk": 5}, "hallucination": {"risk_level": "low", "risky_claims": [], "rationale": ""}, "verification": {"missing_when_needed": false, "suggested_steps": []}, "schema": {"envelope_ok": true, "payload_ok": true, "notes": "", "missing_fields": []}, "rewrite_instructions": []}',
                    model="test",
                    provider="test",
                    request_id="req",
                    usage={},
                    timing={},
                    raw={},
                )
            )
            
            config = {"generation": {}, "seed": 42}
            
            stats = generate_critiques(
                drafts_path=drafts_path,
                out_dir=tmpdir_path,
                teacher=mock_teacher,
                config=config,
                schema=schema,
                template=template,
                resume=True,
            )
            
            # Should skip draft_0, only process draft_1
            assert stats["processed"] == 1
            assert stats["skipped"] == 1
