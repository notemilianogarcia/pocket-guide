from pathlib import Path

from pocketguide.artifacts.measure_perf import measure_perf_main


def test_measure_perf_smoke_stub(tmp_path: Path, monkeypatch) -> None:
    """
    Smoke test for measure_perf in stub mode (no llama.cpp).

    - Creates a temporary registry pointing to a dummy GGUF path.
    - Uses eval/suites/fixed10_v1.jsonl as suite (already in repo).
    - Calls measure_perf_main with --allow_stub so runtime.stub=true is permitted.
    - Asserts perf.json is written with required keys.
    """
    # Minimal registry with one model pointing to dummy GGUF
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "models:\n"
        "  base:\n"
        "    description: \"Base stub model\"\n"
        "    gguf_path: \"artifacts/models/base/gguf/model.Q4_K_M.gguf\"\n"
        "    quant: \"Q4_K_M\"\n",
        encoding="utf-8",
    )

    # Perf output under tmp artifacts root (override via --out)
    out_dir = tmp_path / "artifacts" / "models" / "base"
    out_path = out_dir / "perf.json"

    # Use real fixed10 suite from repo; runtime.stub=true ensures no llama.cpp call.
    # Capture stderr to avoid cluttering test output.
    import io
    import sys

    err_buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err_buf)

    measure_perf_main(
        [
            "--model_name",
            "base",
            "--registry",
            str(registry),
            "--suite",
            "eval/suites/fixed10_v1.jsonl",
            "--runtime_config",
            "configs/runtime_local.yaml",
            "--out",
            str(out_path),
            "--allow_stub",
        ]
    )

    assert out_path.exists(), "perf.json should be created"
    import json

    perf = json.loads(out_path.read_text(encoding="utf-8"))
    assert perf.get("model_name") == "base"
    assert "gguf_path" in perf
    assert perf.get("measurement", {}).get("suite") == "eval/suites/fixed10_v1.jsonl"
    assert "latency_ms" in perf.get("measurement", {})
    assert "ram_usage" in perf.get("measurement", {})

