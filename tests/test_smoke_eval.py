"""Smoke tests for evaluation benchmark."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml
from pocketguide.eval.benchmark import load_jsonl, run_benchmark, run_suite_directory, save_jsonl


def test_load_jsonl_valid_file():
    """Test loading a valid JSONL file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "1", "prompt": "test1"}\n')
        f.write('{"id": "2", "prompt": "test2"}\n')
        temp_path = Path(f.name)

    try:
        data = load_jsonl(temp_path)
        assert len(data) == 2
        assert data[0]["id"] == "1"
        assert data[1]["prompt"] == "test2"
    finally:
        temp_path.unlink()


def test_load_jsonl_skips_blank_lines():
    """Test that blank lines are skipped."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "1", "prompt": "test1"}\n')
        f.write("\n")
        f.write('{"id": "2", "prompt": "test2"}\n')
        temp_path = Path(f.name)

    try:
        data = load_jsonl(temp_path)
        assert len(data) == 2
    finally:
        temp_path.unlink()


def test_load_jsonl_raises_on_invalid_json():
    """Test that invalid JSON raises ValueError with file and line number."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "1", "prompt": "test1"}\n')
        f.write("invalid json line\n")
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_jsonl(temp_path)
    finally:
        temp_path.unlink()


def test_load_jsonl_raises_on_missing_id():
    """Test that missing 'id' field raises ValueError with file and line number."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "1", "prompt": "test1"}\n')
        f.write('{"prompt": "test2"}\n')  # Missing 'id'
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Missing required fields.*line 2.*\\['id'\\]"):
            load_jsonl(temp_path)
    finally:
        temp_path.unlink()


def test_load_jsonl_raises_on_missing_prompt():
    """Test that missing 'prompt' field raises ValueError with file and line number."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "1", "prompt": "test1"}\n')
        f.write('{"id": "2"}\n')  # Missing 'prompt'
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Missing required fields.*line 2.*\\['prompt'\\]"):
            load_jsonl(temp_path)
    finally:
        temp_path.unlink()


def test_run_benchmark_creates_outputs():
    """Test that benchmark creates expected output files with new config structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test suite
        suite_path = tmpdir_path / "test_suite.jsonl"
        test_data = [
            {"id": "test_1", "prompt": "First test prompt"},
            {"id": "test_2", "prompt": "Second test prompt"},
            {"id": "test_3", "prompt": "Third test prompt"},
        ]
        save_jsonl(test_data, suite_path)

        # Create eval config
        config_path = tmpdir_path / "eval.yaml"
        config_data = {
            "model": {"id": "test-model", "revision": "main"},
            "seed": 42,
            "gen": {
                "max_new_tokens": 256,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "suites": [{"path": str(suite_path), "name": "test"}],
            "out_root": str(tmpdir_path / "runs"),
            "device": "cpu",
            "dtype": "float32",
            "save_predictions": True,
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Run benchmark
        run_benchmark(
            config_path=config_path,
            verbose=False,
        )

        # Check outputs exist
        out_dir = tmpdir_path / "runs"
        run_dirs = list(out_dir.iterdir())
        assert len(run_dirs) == 1

        run_dir = run_dirs[0]
        predictions_path = run_dir / "predictions.jsonl"
        meta_path = run_dir / "meta.json"

        assert predictions_path.exists()
        assert meta_path.exists()

        # Check predictions content
        predictions = load_jsonl(predictions_path)
        assert len(predictions) == 3
        assert all("id" in p for p in predictions)
        assert all("prompt" in p for p in predictions)
        assert all("response" in p for p in predictions)
        assert all("response_text" in p for p in predictions)

        # Check metadata content with new structure
        with open(meta_path) as f:
            meta = json.load(f)
        assert "run_id" in meta
        assert "model_id" in meta
        assert meta["model_id"] == "test-model"
        assert "revision" in meta
        assert meta["revision"] == "main"
        assert "seed" in meta
        assert meta["seed"] == 42
        assert "generation_config" in meta
        assert "device" in meta
        assert meta["device"] == "cpu"
        assert "dtype" in meta
        assert meta["dtype"] == "float32"
        assert "suites" in meta
        assert len(meta["suites"]) == 1
        assert meta["suites"][0]["num_examples"] == 3
        assert "environment" in meta
        assert "python_version" in meta["environment"]
        assert "package_versions" in meta
        assert "git_commit" in meta
        assert "git_is_dirty" in meta


def test_run_benchmark_with_actual_smoke_suite():
    """Integration test with actual smoke.jsonl and eval.yaml."""
    suite_path = Path("eval/suites/smoke.jsonl")
    config_path = Path("configs/eval.yaml")

    if not suite_path.exists() or not config_path.exists():
        pytest.skip("smoke.jsonl or eval.yaml not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "runs"

        # Run with override out_dir
        run_benchmark(
            config_path=config_path,
            out_dir=out_dir,
            verbose=False,
        )

        # Verify outputs
        run_dirs = list(out_dir.iterdir())
        assert len(run_dirs) == 1

        run_dir = run_dirs[0]
        meta_path = run_dir / "meta.json"
        predictions_path = run_dir / "predictions.jsonl"

        assert meta_path.exists()
        assert predictions_path.exists()

        # Check meta.json has required fields with new structure
        with open(meta_path) as f:
            meta = json.load(f)

        assert "model_id" in meta
        assert "seed" in meta
        assert "suites" in meta
        assert len(meta["suites"]) > 0

        # Smoke suite should have exactly 5 examples
        predictions = load_jsonl(predictions_path)
        assert len(predictions) == 5


def test_run_suite_directory_multiple_suites():
    """Test running benchmark on multiple JSONL files in a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create multiple test suites
        suite1_path = tmpdir_path / "suite1.jsonl"
        suite1_data = [
            {"id": "s1_1", "prompt": "Suite 1 prompt 1"},
            {"id": "s1_2", "prompt": "Suite 1 prompt 2"},
        ]
        save_jsonl(suite1_data, suite1_path)

        suite2_path = tmpdir_path / "suite2.jsonl"
        suite2_data = [
            {"id": "s2_1", "prompt": "Suite 2 prompt 1", "required_fields": ["name", "age"]},
            {"id": "s2_2", "prompt": "Suite 2 prompt 2"},
            {"id": "s2_3", "prompt": "Suite 2 prompt 3"},
        ]
        save_jsonl(suite2_data, suite2_path)

        # Create eval config
        config_path = tmpdir_path / "eval.yaml"
        config_data = {
            "model": {"id": "test-model", "revision": None},
            "seed": 42,
            "gen": {
                "max_new_tokens": 128,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "suite_dir": str(tmpdir_path),
            "out_root": str(tmpdir_path / "runs"),
            "device": "cpu",
            "dtype": "float32",
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Run suite directory with fixed run_id for deterministic testing
        run_suite_directory(
            config_path=config_path,
            suite_dir=tmpdir_path,
            run_id="20250101_000000",
            verbose=False,
        )

        # Check outputs - should be in fixed run_id directory
        out_dir = tmpdir_path / "runs"
        run_dir = out_dir / "20250101_000000"
        assert run_dir.exists(), "Run directory with fixed run_id should exist"

        outputs_path = run_dir / "base_model_outputs.jsonl"
        meta_path = run_dir / "meta.json"
        metrics_path = run_dir / "metrics.json"

        assert outputs_path.exists(), "base_model_outputs.jsonl should exist"
        assert meta_path.exists(), "meta.json should exist"
        assert metrics_path.exists(), "metrics.json should exist"

        # Check outputs content
        outputs = load_jsonl(outputs_path)
        assert len(outputs) == 5, f"Expected 5 outputs, got {len(outputs)}"

        # Verify each output has required fields including checks
        for output in outputs:
            assert "id" in output
            assert "suite" in output
            assert "prompt" in output
            assert "output_text" in output
            assert "model" in output
            assert "seed" in output
            assert "gen" in output
            assert "checks" in output, "Output should include checks"

            # Verify checks structure
            checks = output["checks"]
            assert "strict_json_ok" in checks
            assert "lenient_json_ok" in checks
            assert "required_fields_ok" in checks
            assert "missing_fields" in checks
            assert "uncertainty" in checks
            assert isinstance(checks["strict_json_ok"], bool)
            assert isinstance(checks["lenient_json_ok"], bool)

        # Check that suites are named correctly (without .jsonl extension)
        suite_names = {output["suite"] for output in outputs}
        assert "suite1" in suite_names
        assert "suite2" in suite_names

        # Check metadata with new structure
        with open(meta_path) as f:
            meta = json.load(f)
        assert "run_id" in meta
        assert meta["run_id"] == "20250101_000000"
        assert "model_id" in meta
        assert meta["model_id"] == "test-model"
        assert "suites" in meta
        assert len(meta["suites"]) == 2
        # Check new provenance fields
        assert "git_commit" in meta
        assert "git_is_dirty" in meta
        assert "package_versions" in meta
        assert "eval_config_resolved" in meta
        # Total examples should be 5
        total_examples = sum(s.get("num_examples", 0) for s in meta["suites"])
        assert total_examples == 5

        # Check metrics
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "overall" in metrics
        assert "by_suite" in metrics
        assert "definitions" in metrics

        # Verify overall metrics structure
        overall = metrics["overall"]
        assert "n" in overall
        assert overall["n"] == 5
        assert "strict_json_parse_rate" in overall
        assert "lenient_json_parse_rate" in overall
        assert "assumptions_marker_rate" in overall
        assert "verification_marker_rate" in overall
        assert "clarifying_questions_rate" in overall

        # Verify by-suite metrics
        assert "suite1" in metrics["by_suite"]
        assert "suite2" in metrics["by_suite"]
        assert metrics["by_suite"]["suite1"]["n"] == 2
        assert metrics["by_suite"]["suite2"]["n"] == 3


def test_contract_integration_with_schema_validation():
    """Test that contract validation is integrated and records schema compliance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test suite with prompts that will generate various outputs
        suite_path = tmpdir_path / "contract_test.jsonl"
        test_data = [
            {"id": "valid_all", "prompt": "Generate valid envelope and payload"},
            {"id": "fenced_json", "prompt": "Generate JSON in markdown fence"},
            {"id": "invalid_payload", "prompt": "Generate invalid payload"},
            {"id": "parse_fail", "prompt": "Generate non-JSON output"},
        ]
        save_jsonl(test_data, suite_path)

        # Create eval config
        config_path = tmpdir_path / "eval.yaml"
        config_data = {
            "model": {"id": "test-model", "revision": "main"},
            "seed": 42,
            "gen": {
                "max_new_tokens": 256,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "suites": [{"path": str(suite_path), "name": "contract_test"}],
            "out_root": str(tmpdir_path / "runs"),
            "device": "cpu",
            "dtype": "float32",
            "save_predictions": True,
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

            # Mock generate_stub_response to return different envelope+payload dicts
            from pocketguide.eval import benchmark
            from pocketguide.inference import cli

            call_count = [0]  # Use list to allow modification in closure

            def mock_generate_stub(prompt: str):
                """Return different response dicts based on call order."""
                call_count[0] += 1

                if call_count[0] == 1:  # valid_all - return valid envelope+payload
                    return {
                        "summary": "Valid response",
                        "assumptions": ["Test assumption"],
                        "uncertainty_notes": "None",
                        "next_steps": ["Step 1"],
                        "verification_steps": ["Verify 1"],
                        "payload_type": "procedure",
                        "payload": {
                            "title": "Test Procedure",
                            "steps": [{"step": 1, "instruction": "Do something"}],
                        },
                    }
                elif call_count[0] == 2:  # fenced_json - will be wrapped in markdown fence
                    return {
                        "summary": "Fenced",
                        "assumptions": [],
                        "uncertainty_notes": "",
                        "next_steps": [],
                        "verification_steps": [],
                        "payload_type": "checklist",
                        "payload": {
                            "title": "List",
                            "groups": [{"name": "Group", "items": [{"text": "Item"}]}],
                        },
                    }
                elif call_count[0] == 3:  # invalid_payload - missing required field
                    return {
                        "summary": "Invalid payload",
                        "assumptions": [],
                        "uncertainty_notes": "",
                        "next_steps": [],
                        "verification_steps": [],
                        "payload_type": "procedure",
                        "payload": {"title": "Missing steps field"},
                    }
                else:  # parse_fail - return string that won't be valid JSON
                    return {"raw_text": "This is not JSON at all, just plain text."}

            # Mock format_response to handle the different response dicts
            def mock_format(response: dict):
                """Format responses, including markdown fencing for call 2."""
                call_num = call_count[0]

                # For call 2, wrap JSON in markdown fence
                if call_num == 2:
                    return f"```json\n{json.dumps(response)}\n```"
                # For call 4, return non-JSON text
                elif call_num == 4:
                    return response.get("raw_text", "Plain text output")
                # Otherwise return JSON
                else:
                    return json.dumps(response)

            # Save originals from both modules
            original_generate_cli = cli.generate_stub_response
            original_format_cli = cli.format_response
            original_generate_bm = benchmark.generate_stub_response
            original_format_bm = benchmark.format_response

            # Apply mocks to both modules
            cli.generate_stub_response = mock_generate_stub
            cli.format_response = mock_format
            benchmark.generate_stub_response = mock_generate_stub
            benchmark.format_response = mock_format

            try:
                # Run benchmark using run_suite_directory (which has contract integration)
                run_suite_directory(
                    config_path=config_path,
                    suite_dir=tmpdir_path,  # Directory containing contract_test.jsonl
                    out_dir=tmpdir_path / "runs",
                    run_id="20250101_000000",  # Fixed run_id for easier testing
                    verbose=False,
                )

                # Check outputs
                outputs_path = tmpdir_path / "runs" / "20250101_000000" / "base_model_outputs.jsonl"
                assert outputs_path.exists()

                with open(outputs_path) as f:
                    outputs = [json.loads(line) for line in f]

                assert len(outputs) == 4

                # Check that each output has contract section
                for output in outputs:
                    assert "contract" in output
                    contract = output["contract"]

                    # Check required contract fields
                    assert "strict_json_ok" in contract
                    assert "lenient_json_ok" in contract
                    assert "envelope_ok" in contract
                    assert "payload_ok" in contract
                    assert "overall_ok" in contract
                    assert "payload_type" in contract
                    # error may be None or dict

                # Verify specific cases
                outputs_by_id = {o["id"]: o for o in outputs}

                # Valid case: should pass all checks
                valid = outputs_by_id["valid_all"]
                assert valid["contract"]["strict_json_ok"] is True
                assert valid["contract"]["envelope_ok"] is True
                assert valid["contract"]["payload_ok"] is True
                assert valid["contract"]["overall_ok"] is True
                assert valid["contract"]["payload_type"] == "procedure"
                assert valid["contract"]["error"] is None

                # Fenced JSON: strict fails, lenient passes
                fenced = outputs_by_id["fenced_json"]
                assert fenced["contract"]["strict_json_ok"] is False
                assert fenced["contract"]["lenient_json_ok"] is True
                assert fenced["contract"]["envelope_ok"] is True
                assert fenced["contract"]["payload_ok"] is True
                assert fenced["contract"]["overall_ok"] is True

                # Invalid payload: envelope passes, payload fails
                invalid_payload = outputs_by_id["invalid_payload"]
                assert invalid_payload["contract"]["strict_json_ok"] is True
                assert invalid_payload["contract"]["envelope_ok"] is True
                assert invalid_payload["contract"]["payload_ok"] is False
                assert invalid_payload["contract"]["overall_ok"] is False
                assert invalid_payload["contract"]["error"] is not None
                assert invalid_payload["contract"]["error"]["code"] == "PAYLOAD_SCHEMA_FAILED"

                # Parse fail: everything fails
                parse_fail = outputs_by_id["parse_fail"]
                assert parse_fail["contract"]["strict_json_ok"] is False
                assert parse_fail["contract"]["lenient_json_ok"] is False
                assert parse_fail["contract"]["envelope_ok"] is False
                assert parse_fail["contract"]["payload_ok"] is False
                assert parse_fail["contract"]["overall_ok"] is False
                assert parse_fail["contract"]["error"] is not None

                # Check metrics include contract rates
                metrics_path = tmpdir_path / "runs" / "20250101_000000" / "metrics.json"
                assert metrics_path.exists()

                with open(metrics_path) as f:
                    metrics = json.load(f)

                overall = metrics["overall"]
                assert "envelope_pass_rate" in overall
                assert "payload_pass_rate" in overall
                assert "overall_contract_pass_rate" in overall

                # Should have 2/4 valid (valid_all, fenced_json have envelope pass)
                # and 2/4 fully valid (valid_all, fenced_json pass all)
                assert overall["envelope_pass_rate"] == 0.75  # 3/4 (valid, fenced, invalid_payload)
                assert overall["payload_pass_rate"] == 0.5  # 2/4 (valid, fenced)
                assert overall["overall_contract_pass_rate"] == 0.5  # 2/4 (valid, fenced)

                # Check definitions include new metrics
                assert "envelope_pass_rate" in metrics["definitions"]
                assert "payload_pass_rate" in metrics["definitions"]
                assert "overall_contract_pass_rate" in metrics["definitions"]

            finally:
                # Restore originals in both modules
                cli.generate_stub_response = original_generate_cli
                cli.format_response = original_format_cli
                benchmark.generate_stub_response = original_generate_bm
                benchmark.format_response = original_format_bm
