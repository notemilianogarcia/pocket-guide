"""
Tests for local eval v2 pathing (Lesson 7.4).

Stub runtime only; no llama.cpp. Asserts v2 run directory naming,
local_metrics.json existence, and same metrics schema as v1.
"""

import json
from pathlib import Path

import pytest
import yaml

from pocketguide.eval.local_eval import run_local_eval

# Top-level keys expected in local_metrics.json (identical to v1)
LOCAL_METRICS_TOP_KEYS = frozenset({
    "run_id",
    "runtime_stub",
    "suites",
    "per_suite",
    "overall",
    "tokens_per_sec_note",
})


@pytest.fixture
def stub_runtime_config(tmp_path: Path) -> Path:
    """Config with runtime.stub: true (no llama.cpp)."""
    cfg = {
        "runtime": {"backend": "llamacpp", "stub": True},
        "model": {"gguf_path": "models/gguf/PLACEHOLDER.gguf", "context_length": 4096},
        "generation": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 512},
        "execution": {"llamacpp_bin": None, "extra_args": []},
    }
    path = tmp_path / "runtime_local.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


@pytest.fixture
def mini_suite(tmp_path: Path) -> Path:
    """Minimal suite: 2 prompts."""
    path = tmp_path / "mini.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"id": "x", "prompt": "Plan a day."}\n')
        f.write('{"id": "y", "prompt": "Visa?"}\n')
    return path


def test_v2_run_dir_ends_with_v2(
    tmp_path: Path,
    stub_runtime_config: Path,
    mini_suite: Path,
) -> None:
    """With run_id_suffix=_v2, run directory name ends with _v2."""
    out_dir = tmp_path / "runs" / "eval"
    run_dir = run_local_eval(
        runtime_config_path=stub_runtime_config,
        suite_paths=[mini_suite],
        out_dir=out_dir,
        run_id=None,
        project_root=tmp_path,
        write_outputs_jsonl=True,
        run_id_suffix="_v2",
    )
    assert run_dir.name.endswith("_v2"), f"Expected run dir to end with _v2, got {run_dir.name}"


def test_v2_local_metrics_exists(
    tmp_path: Path,
    stub_runtime_config: Path,
    mini_suite: Path,
) -> None:
    """v2 run produces local_metrics.json."""
    out_dir = tmp_path / "runs" / "eval"
    run_dir = run_local_eval(
        runtime_config_path=stub_runtime_config,
        suite_paths=[mini_suite],
        out_dir=out_dir,
        run_id="20260203_120000",
        project_root=tmp_path,
        write_outputs_jsonl=False,
        run_id_suffix="_v2",
    )
    metrics_path = run_dir / "local_metrics.json"
    assert metrics_path.exists(), f"Expected {metrics_path} to exist"


def test_v2_metrics_schema_matches_v1(
    tmp_path: Path,
    stub_runtime_config: Path,
    mini_suite: Path,
) -> None:
    """v2 local_metrics.json has same top-level keys as v1 (no new fields)."""
    out_dir = tmp_path / "runs" / "eval"
    run_dir = run_local_eval(
        runtime_config_path=stub_runtime_config,
        suite_paths=[mini_suite],
        out_dir=out_dir,
        run_id="schema_test",
        project_root=tmp_path,
        write_outputs_jsonl=False,
        run_id_suffix="_v2",
    )
    metrics = json.loads((run_dir / "local_metrics.json").read_text())
    top_keys = set(metrics.keys())
    missing = LOCAL_METRICS_TOP_KEYS - top_keys
    assert not missing, f"local_metrics.json missing required keys: {missing}"


def test_gguf_path_override_does_not_break_stub(
    tmp_path: Path,
    stub_runtime_config: Path,
    mini_suite: Path,
) -> None:
    """With stub mode, gguf_path_override does not require file to exist; run completes."""
    out_dir = tmp_path / "runs" / "eval"
    run_dir = run_local_eval(
        runtime_config_path=stub_runtime_config,
        suite_paths=[mini_suite],
        out_dir=out_dir,
        run_id="override_stub",
        project_root=tmp_path,
        write_outputs_jsonl=False,
        gguf_path_override="/nonexistent/v2_model.gguf",
        run_id_suffix="_v2",
    )
    assert run_dir.name.endswith("_v2")
    assert (run_dir / "local_metrics.json").exists()
