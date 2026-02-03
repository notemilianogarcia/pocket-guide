from pathlib import Path

from pocketguide.inference.run import main as run_main


def test_inference_run_local_stub_uses_registry(tmp_path: Path, monkeypatch) -> None:
    """
    Smoke test: run local stub via registry-backed runner.

    - Creates a temporary registry pointing to a dummy GGUF path.
    - Ensures the runner executes without requiring llama.cpp or a real GGUF.
    - Asserts the output envelope has required top-level keys.
    """
    # Minimal registry with one model pointing to a dummy GGUF path
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "models:\n"
        "  base:\n"
        "    description: \"Base stub model\"\n"
        "    gguf_path: \"artifacts/models/base/gguf/model.Q4_K_M.gguf\"\n"
        "    quant: \"Q4_K_M\"\n",
        encoding="utf-8",
    )

    # Ensure runtime_local.yaml stub is respected (no llama.cpp / GGUF needed).
    # We don't need to modify configs/runtime_local.yaml because runtime.stub=true by default.

    # Capture stdout
    import io
    import sys
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)

    prompt = "Plan a 1-day trip to Montreal"
    run_main(
        [
            "--prompt",
            prompt,
            "--local",
            "--model",
            "base",
            "--registry",
            str(registry),
            "--runtime_config",
            "configs/runtime_local.yaml",
        ]
    )

    output = buf.getvalue().strip()
    assert output, "Runner should print a JSON envelope"

    import json

    data = json.loads(output)
    # At minimum, check that required envelope keys exist (stub mode)
    for key in (
        "summary",
        "assumptions",
        "uncertainty_notes",
        "next_steps",
        "verification_steps",
        "payload_type",
        "payload",
    ):
        assert key in data

