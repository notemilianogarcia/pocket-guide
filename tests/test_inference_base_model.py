"""Tests for base model inference with mocked Transformers."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import yaml
from pocketguide.inference.base_model import (
    GenSpec,
    ModelSpec,
    RuntimeSpec,
    generate_one,
    load_model_and_tokenizer,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.pad_token_id = 0

    def __call__(self, text, return_tensors=None):
        """Mock tokenization."""
        # Simple mock: each word is a token
        tokens = text.split()
        input_ids = list(range(1, len(tokens) + 1))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([input_ids])}
        return {"input_ids": input_ids}

    def decode(self, token_ids, skip_special_tokens=False):
        """Mock decoding."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        # Simple mock: return numbered tokens
        return " ".join(f"token_{i}" for i in token_ids)


class MockModel:
    """Mock model for testing."""

    def __init__(self, device="cpu"):
        self.device_val = device
        self.eval_called = False

    @property
    def device(self):
        """Mock device property."""
        return self.device_val

    def to(self, device):
        """Mock to() method."""
        self.device_val = device
        return self

    def eval(self):
        """Mock eval() method."""
        self.eval_called = True
        return self

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=256,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        pad_token_id=0,
        return_dict_in_generate=True,
        return_legacy_cache=True,
        **kwargs,
    ):
        """Mock generation."""
        # Generate deterministic output based on input
        prompt_len = input_ids.shape[1]
        completion_len = min(max_new_tokens, 10)  # Generate 10 tokens
        total_len = prompt_len + completion_len

        # Create mock sequence
        sequence = torch.arange(1, total_len + 1).unsqueeze(0)

        # Return dict-like object
        result = MagicMock()
        result.sequences = sequence
        return result


def test_model_spec_creation():
    """Test ModelSpec dataclass creation."""
    spec = ModelSpec(id="test-model", revision="main")
    assert spec.id == "test-model"
    assert spec.revision == "main"

    spec_no_rev = ModelSpec(id="test-model")
    assert spec_no_rev.id == "test-model"
    assert spec_no_rev.revision is None


def test_runtime_spec_defaults():
    """Test RuntimeSpec defaults."""
    spec = RuntimeSpec()
    assert spec.device == "cpu"
    assert spec.dtype == "float32"


def test_gen_spec_defaults():
    """Test GenSpec defaults."""
    spec = GenSpec()
    assert spec.max_new_tokens == 256
    assert spec.do_sample is False
    assert spec.temperature == 0.0
    assert spec.top_p == 1.0
    assert spec.repetition_penalty == 1.0


@patch("pocketguide.inference.base_model.AutoModelForCausalLM")
@patch("pocketguide.inference.base_model.AutoTokenizer")
def test_load_model_and_tokenizer(mock_tokenizer_cls, mock_model_cls):
    """Test model and tokenizer loading."""
    # Setup mocks
    mock_tokenizer = MockTokenizer()
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

    mock_model = MockModel()
    mock_model_cls.from_pretrained.return_value = mock_model

    # Load
    model_spec = ModelSpec(id="test-model", revision="main")
    runtime_spec = RuntimeSpec(device="cpu", dtype="float32")

    model, tokenizer = load_model_and_tokenizer(model_spec, runtime_spec)

    # Assertions
    assert model.eval_called
    assert tokenizer.pad_token is not None
    mock_tokenizer_cls.from_pretrained.assert_called_once()
    mock_model_cls.from_pretrained.assert_called_once()


def test_generate_one_with_mock():
    """Test generation with mocked model and tokenizer."""
    # Create mocks
    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()

    # Generate
    prompt = "Hello world test"
    gen_spec = GenSpec(max_new_tokens=10, do_sample=False)
    result = generate_one(mock_model, mock_tokenizer, prompt, gen_spec, seed=42)

    # Assertions
    assert "text" in result
    assert "usage" in result
    assert "timing" in result

    # Check usage fields
    assert "prompt_tokens" in result["usage"]
    assert "completion_tokens" in result["usage"]
    assert "total_tokens" in result["usage"]
    assert result["usage"]["prompt_tokens"] > 0
    assert result["usage"]["completion_tokens"] > 0
    assert (
        result["usage"]["total_tokens"]
        == result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]
    )

    # Check timing fields
    assert "latency_s" in result["timing"]
    assert "tokens_per_s" in result["timing"]
    assert result["timing"]["latency_s"] > 0
    assert result["timing"]["tokens_per_s"] >= 0


def test_generate_one_deterministic():
    """Test that same input produces same output."""
    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()

    prompt = "Test prompt"
    gen_spec = GenSpec(max_new_tokens=10, do_sample=False)

    result1 = generate_one(mock_model, mock_tokenizer, prompt, gen_spec, seed=42)
    result2 = generate_one(mock_model, mock_tokenizer, prompt, gen_spec, seed=42)

    # Same seed should give same text
    assert result1["text"] == result2["text"]
    assert result1["usage"] == result2["usage"]


def test_smoke_inference_integration():
    """Integration test for smoke inference CLI mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create eval config
        config_path = tmpdir_path / "eval.yaml"
        config_data = {
            "model": {"id": "mock-model", "revision": None},
            "seed": 42,
            "gen": {
                "max_new_tokens": 10,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
            },
            "out_root": str(tmpdir_path / "runs"),
            "device": "cpu",
            "dtype": "float32",
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Mock the model loading
        with patch("pocketguide.inference.base_model.AutoTokenizer.from_pretrained") as mock_tok:
            with patch(
                "pocketguide.inference.base_model.AutoModelForCausalLM.from_pretrained"
            ) as mock_mdl:
                mock_tok.return_value = MockTokenizer()
                mock_mdl.return_value = MockModel()

                # Import and run smoke inference
                from pocketguide.eval.benchmark import run_smoke_inference

                run_smoke_inference(
                    config_path=config_path,
                    prompt="Test smoke inference",
                    verbose=False,
                )

        # Check output exists
        runs_dir = tmpdir_path / "runs"
        assert runs_dir.exists()

        run_dirs = list(runs_dir.iterdir())
        assert len(run_dirs) == 1

        smoke_file = run_dirs[0] / "smoke_infer.json"
        assert smoke_file.exists()

        # Verify content
        with open(smoke_file) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "prompt" in data
        assert "model" in data
        assert "seed" in data
        assert "result" in data

        result = data["result"]
        assert "text" in result
        assert "usage" in result
        assert "timing" in result

        # Verify usage fields
        assert "prompt_tokens" in result["usage"]
        assert "completion_tokens" in result["usage"]
        assert "total_tokens" in result["usage"]
        assert isinstance(result["usage"]["prompt_tokens"], int)
        assert isinstance(result["usage"]["completion_tokens"], int)
        assert isinstance(result["usage"]["total_tokens"], int)

        # Verify timing fields
        assert "latency_s" in result["timing"]
        assert "tokens_per_s" in result["timing"]
        assert isinstance(result["timing"]["latency_s"], int | float)
        assert isinstance(result["timing"]["tokens_per_s"], int | float)
