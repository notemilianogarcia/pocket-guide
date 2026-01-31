"""
Tests for dataset generation (v1) with quality gates.

All tests use mocked teacher responses - no network calls required.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pocketguide.data_generation.generate_dataset_v1 import (
    hash_file,
    load_jsonl,
    join_inputs,
    build_refinement_prompt,
    parse_refined_output,
    build_accepted_record,
    build_rejected_record,
    generate_dataset,
    JoinedSample,
)
from pocketguide.data_generation.quality_gates import (
    check_contract_ok,
    check_verification_when_needed,
    check_overconfidence_guard,
    apply_all_gates,
    compute_overall_quality,
    get_rejection_reasons,
    GateResult,
)


class TestHashFile:
    """Test file hashing for reproducibility."""
    
    def test_hash_file_deterministic(self):
        """Hash should be deterministic for same content."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test content\n")
            path = Path(f.name)
        
        try:
            hash1 = hash_file(path)
            hash2 = hash_file(path)
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex length
        finally:
            path.unlink()


class TestLoadJsonl:
    """Test JSONL loading."""
    
    def test_load_valid_jsonl(self):
        """Should load valid JSONL."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            f.write('{"id": "1", "value": "a"}\n')
            f.write('{"id": "2", "value": "b"}\n')
            path = Path(f.name)
        
        try:
            records = load_jsonl(path)
            assert len(records) == 2
            assert records[0]["id"] == "1"
            assert records[1]["id"] == "2"
        finally:
            path.unlink()
    
    def test_load_jsonl_skips_blank_lines(self):
        """Should skip blank lines."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            f.write('{"id": "1"}\n')
            f.write('\n')
            f.write('{"id": "2"}\n')
            path = Path(f.name)
        
        try:
            records = load_jsonl(path)
            assert len(records) == 2
        finally:
            path.unlink()


class TestJoinInputs:
    """Test joining prompt plans, drafts, and critiques."""
    
    def test_join_complete_samples(self):
        """Should join samples with all components."""
        plans = [{"id": "1", "prompt": "test"}]
        drafts = [{"id": "1", "output_text": "response"}]
        critiques = [{"draft_id": "1", "verdict": "pass"}]
        
        joined, missing = join_inputs(plans, drafts, critiques)
        
        assert len(joined) == 1
        assert len(missing) == 0
        assert joined[0].id == "1"
    
    def test_join_missing_draft(self):
        """Should reject samples missing draft."""
        plans = [{"id": "1", "prompt": "test"}]
        drafts = []
        critiques = [{"draft_id": "1", "verdict": "pass"}]
        
        joined, missing = join_inputs(plans, drafts, critiques)
        
        assert len(joined) == 0
        assert len(missing) == 1
        assert "draft" in missing[0]["missing_components"]
    
    def test_join_missing_critique_allowed(self):
        """Should allow samples without critique."""
        plans = [{"id": "1", "prompt": "test"}]
        drafts = [{"id": "1", "output_text": "response"}]
        critiques = []
        
        joined, missing = join_inputs(plans, drafts, critiques)
        
        assert len(joined) == 1
        assert joined[0].critique is None


class TestBuildRefinementPrompt:
    """Test refinement prompt construction."""
    
    def test_build_prompt_with_all_fields(self):
        """Should format template with all fields."""
        template = "PROMPT: {prompt}\nDRAFT: {draft_output}\nCRITIQUE: {critique_json}\nINSTRUCTIONS: {rewrite_instructions}\nTYPE: {payload_type}"
        
        prompt = build_refinement_prompt(
            prompt_text="What are visa requirements?",
            draft_output="You need a visa.",
            critique_json={"verdict": "revise", "issues": []},
            rewrite_instructions=["Add verification steps", "Soften certainty"],
            payload_type="info_card",
            template=template
        )
        
        assert "What are visa requirements?" in prompt
        assert "You need a visa." in prompt
        assert "Add verification steps" in prompt
        assert "info_card" in prompt


class TestQualityGates:
    """Test individual quality gates."""
    
    def test_contract_ok_passes(self):
        """Should pass when both envelope and payload valid."""
        result = check_contract_ok(
            envelope_ok=True,
            payload_ok=True,
            parse_mode="strict",
            parsed_envelope={"summary": "test"}
        )
        assert result.passed
        assert result.reason_code == "ok"
    
    def test_contract_ok_fails_envelope(self):
        """Should fail when envelope invalid."""
        result = check_contract_ok(
            envelope_ok=False,
            payload_ok=True,
            parse_mode="strict",
            parsed_envelope=None
        )
        assert not result.passed
        assert result.reason_code == "envelope_invalid"
    
    def test_verification_when_needed_passes_non_sensitive(self):
        """Should pass for non-time-sensitive content."""
        result = check_verification_when_needed(
            full_response_text="Pack light clothing for warm weather",
            verification_steps=[],
            uncertainty_notes=""
        )
        assert result.passed
        assert not result.details["time_sensitive"]
    
    def test_verification_when_needed_passes_with_steps(self):
        """Should pass time-sensitive content with verification."""
        result = check_verification_when_needed(
            full_response_text="Most visitors need a visa to enter",
            verification_steps=["Check embassy website"],
            uncertainty_notes="Requirements may change"
        )
        assert result.passed
        assert result.details["time_sensitive"]
    
    def test_verification_when_needed_fails_missing_steps(self):
        """Should fail time-sensitive content without verification."""
        result = check_verification_when_needed(
            full_response_text="Visa fees are $150",
            verification_steps=[],
            uncertainty_notes=""
        )
        assert not result.passed
        assert "missing_verification_requirements" in result.reason_code
        assert "verification_steps" in result.details["missing"]
    
    def test_overconfidence_guard_passes_careful_language(self):
        """Should pass time-sensitive content with careful language."""
        result = check_overconfidence_guard(
            full_response_text="Visitors typically need a visa. Check embassy for current requirements.",
            is_time_sensitive=True
        )
        assert result.passed
    
    def test_overconfidence_guard_fails_absolute_claims(self):
        """Should fail time-sensitive content with absolute claims."""
        result = check_overconfidence_guard(
            full_response_text="You definitely don't need a visa. Guaranteed entry for all tourists.",
            is_time_sensitive=True
        )
        assert not result.passed
        assert result.reason_code == "overconfident_time_sensitive"
        assert "definitely" in result.details["found_phrases"]

    def test_overconfidence_guard_passes_always_verify_style(self):
        """Should pass when 'always' appears only in caveat phrases like 'always verify'."""
        result = check_overconfidence_guard(
            full_response_text="Fees and schedules change. Always verify real-time data from official sources.",
            is_time_sensitive=True
        )
        assert result.passed

    def test_apply_all_gates_integration(self):
        """Should apply all gates and return results."""
        gates = apply_all_gates(
            envelope_ok=True,
            payload_ok=True,
            parse_mode="strict",
            parsed_envelope={
                "summary": "Visa required",
                "verification_steps": ["Check embassy"],
                "uncertainty_notes": "May change"
            },
            full_response_text="Most visitors need a visa. Policies may change."
        )
        
        assert "contract_ok" in gates
        assert "verification_when_needed" in gates
        assert "overconfidence_guard" in gates
        assert all(gate.passed for gate in gates.values())
    
    def test_compute_overall_quality_all_pass(self):
        """Should return True when all gates pass."""
        gates = {
            "gate1": GateResult(passed=True, reason_code="ok", details={}),
            "gate2": GateResult(passed=True, reason_code="ok", details={})
        }
        assert compute_overall_quality(gates)
    
    def test_compute_overall_quality_one_fails(self):
        """Should return False when any gate fails."""
        gates = {
            "gate1": GateResult(passed=True, reason_code="ok", details={}),
            "gate2": GateResult(passed=False, reason_code="failed", details={})
        }
        assert not compute_overall_quality(gates)
    
    def test_get_rejection_reasons(self):
        """Should extract reason codes from failed gates."""
        gates = {
            "gate1": GateResult(passed=True, reason_code="ok", details={}),
            "gate2": GateResult(passed=False, reason_code="error_a", details={}),
            "gate3": GateResult(passed=False, reason_code="error_b", details={})
        }
        reasons = get_rejection_reasons(gates)
        assert len(reasons) == 2
        assert "error_a" in reasons
        assert "error_b" in reasons


class TestParseRefinedOutput:
    """Test parsing and validation of refined outputs."""
    
    def test_parse_strict_valid_json(self):
        """Should parse strict JSON successfully."""
        raw = json.dumps({
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "checklist",
            "payload": {
                "title": "Test Checklist",
                "groups": [{
                    "name": "Test Group",
                    "items": [{
                        "text": "Test task",
                        "priority": "must"
                    }]
                }]
            }
        })
        
        envelope, contract = parse_refined_output(raw, gating_mode="strict")
        
        assert envelope is not None
        assert contract["envelope_ok"]
        assert contract["payload_ok"]
        assert contract["parse_mode"] == "strict"
    
    def test_parse_lenient_allows_markdown(self):
        """Should extract JSON from markdown in lenient mode."""
        raw = "```json\n" + json.dumps({
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "checklist",
            "payload": {
                "title": "Test Checklist",
                "groups": [{
                    "name": "Test Group",
                    "items": [{
                        "text": "Test task",
                        "priority": "must"
                    }]
                }]
            }
        }) + "\n```"
        
        envelope, contract = parse_refined_output(raw, gating_mode="lenient")
        
        assert envelope is not None
        assert contract["parse_mode"] == "lenient"
    
    def test_parse_strict_rejects_markdown(self):
        """Should reject markdown in strict mode."""
        raw = "```json\n{\"summary\": \"test\"}\n```"
        
        envelope, contract = parse_refined_output(raw, gating_mode="strict")
        
        assert envelope is None
        assert not contract["envelope_ok"] or not contract["payload_ok"]


class TestBuildRecords:
    """Test building accepted and rejected records."""
    
    def test_build_accepted_record(self):
        """Should build complete accepted record."""
        sample = JoinedSample(
            id="test_1",
            prompt_plan={
                "id": "test_1",
                "category": "visa",
                "difficulty": "medium",
                "region_tags": ["europe"],
                "payload_type": "info_card",
                "template_version": "v1",
                "template_name": "visa_info",
                "prompt": "What are visa requirements?"
            },
            draft={"id": "test_1", "output_text": "Draft response"},
            critique={"draft_id": "test_1", "verdict": "revise"}
        )
        
        envelope = {"summary": "Final response"}
        teacher_meta = {"provider": "openrouter", "model": "test"}
        gates = {
            "gate1": GateResult(passed=True, reason_code="ok", details={})
        }
        
        record = build_accepted_record(sample, envelope, teacher_meta, gates, True)
        
        assert record["id"] == "test_1"
        assert record["category"] == "visa"
        assert record["response"] == envelope
        assert record["quality"]["overall_ok"]
    
    def test_build_rejected_record(self):
        """Should build complete rejected record."""
        sample = JoinedSample(
            id="test_1",
            prompt_plan={"id": "test_1"},
            draft={"id": "test_1", "output_text": "Draft"},
            critique={
                "draft_id": "test_1",
                "verdict": "revise",
                "issues": [
                    {"type": "hallucination", "severity": "high", "message": "Unsupported claim"}
                ]
            }
        )
        
        contract = {"envelope_ok": False, "errors": ["Invalid JSON"]}
        gates = {
            "gate1": GateResult(passed=False, reason_code="failed", details={})
        }
        
        record = build_rejected_record(
            sample=sample,
            reason_codes=["parse_failed"],
            contract=contract,
            gates=gates,
            raw_text="Invalid response text",
            teacher_metadata={"provider": "test"}
        )
        
        assert record["id"] == "test_1"
        assert "parse_failed" in record["reason_codes"]
        assert record["critique_summary"]["verdict"] == "revise"


class TestGenerateDataset:
    """Test full dataset generation pipeline."""
    
    def test_generate_dataset_dry_run(self):
        """Should generate dataset in dry run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create input files
            plan_path = tmpdir / "plans.jsonl"
            drafts_path = tmpdir / "drafts.jsonl"
            critiques_path = tmpdir / "critiques.jsonl"
            out_dir = tmpdir / "output"
            
            # Write test data
            with open(plan_path, "w") as f:
                f.write(json.dumps({
                    "id": "1",
                    "category": "visa",
                    "difficulty": "medium",
                    "region_tags": ["asia"],
                    "payload_type": "checklist",
                    "template_version": "v1",
                    "template_name": "visa_info",
                    "prompt": "Do I need a visa?"
                }) + "\n")
                f.write(json.dumps({
                    "id": "2",
                    "category": "safety",
                    "difficulty": "easy",
                    "region_tags": ["africa"],
                    "payload_type": "checklist",
                    "template_version": "v1",
                    "template_name": "safety_info",
                    "prompt": "Is it safe?"
                }) + "\n")
            
            with open(drafts_path, "w") as f:
                f.write(json.dumps({"id": "1", "output_text": "Draft about visa"}) + "\n")
                f.write(json.dumps({"id": "2", "output_text": "Draft about safety"}) + "\n")
            
            with open(critiques_path, "w") as f:
                f.write(json.dumps({
                    "draft_id": "1",
                    "verdict": "revise",
                    "rewrite_instructions": ["Add verification steps"]
                }) + "\n")
                f.write(json.dumps({
                    "draft_id": "2",
                    "verdict": "pass",
                    "rewrite_instructions": []
                }) + "\n")
            
            # Run generation
            stats = generate_dataset(
                plan_path=plan_path,
                drafts_path=drafts_path,
                critiques_path=critiques_path,
                out_dir=out_dir,
                limit=None,
                resume=False,
                dry_run=True,
                gating_mode="lenient",
                run_id="test_run",
                teacher_config={"rate_limit_rpm": 30}
            )
            
            # Verify outputs exist
            assert (out_dir / "dataset_v1.jsonl").exists()
            assert (out_dir / "dataset_v1_rejected.jsonl").exists()
            assert (out_dir / "dataset_v1_manifest.json").exists()
            assert (out_dir / "dataset_v1_stats.json").exists()
            
            # Verify stats
            assert stats["attempted"] > 0
            assert stats["accepted"] + stats["rejected"] == stats["attempted"]
    
    @patch("pocketguide.data_generation.generate_dataset_v1.create_teacher_router")
    def test_generate_dataset_with_mocked_teacher(self, mock_create_router):
        """Should generate dataset with mocked teacher (TeacherRequest/TeacherResponse API)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            refined_json = json.dumps({
                "summary": "Visa typically required for most visitors",
                "assumptions": ["Tourist purpose"],
                "uncertainty_notes": "Requirements may change",
                "next_steps": ["Check embassy website"],
                "verification_steps": ["Visit embassy.gov"],
                "payload_type": "checklist",
                "payload": {
                    "title": "Visa Checklist",
                    "groups": [{
                        "name": "Documents",
                        "items": [{
                            "text": "Check visa requirements",
                            "priority": "must"
                        }]
                    }]
                }
            })
            mock_response = MagicMock()
            mock_response.text = refined_json
            mock_response.model = "test-model"
            mock_response.provider = "openrouter"
            mock_response.request_id = "mock_123"
            mock_response.timing = {"latency_s": 0.1}
            mock_response.usage = {"prompt_tokens": 50, "completion_tokens": 30}
            mock_response.raw = {
                "chosen_model": "test-model",
                "attempted_models": ["test-model"],
                "used_structured_outputs": False,
                "structured_outputs_schema_name": None,
            }

            mock_teacher = MagicMock()
            mock_teacher.generate.return_value = mock_response
            mock_create_router.return_value = mock_teacher

            # Create input files
            plan_path = tmpdir / "plans.jsonl"
            drafts_path = tmpdir / "drafts.jsonl"
            critiques_path = tmpdir / "critiques.jsonl"
            out_dir = tmpdir / "output"
            
            with open(plan_path, "w") as f:
                f.write(json.dumps({
                    "id": "1",
                    "category": "visa",
                    "difficulty": "medium",
                    "region_tags": ["asia"],
                    "payload_type": "checklist",
                    "template_version": "v1",
                    "template_name": "visa_info",
                    "prompt": "Do I need a visa?"
                }) + "\n")
            
            with open(drafts_path, "w") as f:
                f.write(json.dumps({"id": "1", "output_text": "You need a visa"}) + "\n")
            
            with open(critiques_path, "w") as f:
                f.write(json.dumps({
                    "draft_id": "1",
                    "verdict": "revise",
                    "rewrite_instructions": ["Add verification steps", "Soften certainty"]
                }) + "\n")
            
            # Run generation (teacher_config must have models for create_teacher_router when not patched; we patch create_teacher_router so minimal config is ok)
            stats = generate_dataset(
                plan_path=plan_path,
                drafts_path=drafts_path,
                critiques_path=critiques_path,
                out_dir=out_dir,
                limit=1,
                resume=False,
                dry_run=False,
                gating_mode="lenient",
                run_id="test_run",
                teacher_config={"models": ["test-model"], "generation": {}, "runtime": {}, "rate_limit": {}},
            )

            # Verify teacher was called with a TeacherRequest (single positional arg)
            mock_teacher.generate.assert_called_once()
            call_arg = mock_teacher.generate.call_args[0][0]
            assert hasattr(call_arg, "messages")
            assert call_arg.messages[0]["role"] == "system"
            
            # Verify outputs
            dataset = load_jsonl(out_dir / "dataset_v1.jsonl")
            assert len(dataset) >= 1
            assert dataset[0]["id"] == "1"
            assert dataset[0]["quality"]["overall_ok"]
    
    def test_generate_dataset_resume_skips_existing(self):
        """Should skip already-processed IDs in resume mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create input files
            plan_path = tmpdir / "plans.jsonl"
            drafts_path = tmpdir / "drafts.jsonl"
            critiques_path = tmpdir / "critiques.jsonl"
            out_dir = tmpdir / "output"
            out_dir.mkdir()
            
            # Write minimal test data
            with open(plan_path, "w") as f:
                f.write(json.dumps({"id": "1", "prompt": "test", "category": "visa", "payload_type": "checklist", "template_version": "v1", "template_name": "test"}) + "\n")
            
            with open(drafts_path, "w") as f:
                f.write(json.dumps({"id": "1", "output_text": "test"}) + "\n")
            
            with open(critiques_path, "w") as f:
                f.write(json.dumps({"draft_id": "1", "verdict": "pass"}) + "\n")
            
            # Create existing dataset with ID 1
            dataset_path = out_dir / "dataset_v1.jsonl"
            with open(dataset_path, "w") as f:
                f.write(json.dumps({"id": "1"}) + "\n")
            
            # Run with resume
            stats = generate_dataset(
                plan_path=plan_path,
                drafts_path=drafts_path,
                critiques_path=critiques_path,
                out_dir=out_dir,
                limit=None,
                resume=True,
                dry_run=True,
                gating_mode="lenient",
                run_id="test_run",
                teacher_config={}
            )
            
            # Should skip ID 1
            assert stats["skipped_resume"] == 1
            assert stats["attempted"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
