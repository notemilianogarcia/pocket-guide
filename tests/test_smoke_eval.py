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
