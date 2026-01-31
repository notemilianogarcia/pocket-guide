"""
Run logging utilities for training and evaluation.

Provides YAML/JSON writing, file hashing, and config redaction for reproducible runs.
"""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def write_yaml(path: Path, obj: Any) -> None:
    """Write a Python object to a YAML file.

    Args:
        path: Output file path.
        obj: Serializable object (dict, list, etc.).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def write_json(path: Path, obj: Any) -> None:
    """Write a Python object to a JSON file with indent.

    Args:
        path: Output file path.
        obj: Serializable object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute a content hash of a file.

    Args:
        path: Path to the file.
        algorithm: Hash algorithm (default sha256).

    Returns:
        Hex digest of the file contents.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash: {path} does not exist")
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_redact(config: dict[str, Any], keys_to_redact: set[str] | None = None) -> dict[str, Any]:
    """Return a copy of config with sensitive keys redacted for logging.

    Args:
        config: Configuration dictionary (nested dicts supported at top level only).
        keys_to_redact: Keys to replace with "<redacted>". Default: api_key, token, secret.

    Returns:
        New dict with redacted values; nested dicts are shallow-copied.
    """
    if keys_to_redact is None:
        keys_to_redact = {"api_key", "token", "secret", "password"}
    out: dict[str, Any] = {}
    for k, v in config.items():
        if k.lower() in {s.lower() for s in keys_to_redact}:
            out[k] = "<redacted>"
        elif isinstance(v, dict):
            out[k] = safe_redact(v, keys_to_redact)
        else:
            out[k] = v
    return out


def get_git_commit() -> str | None:
    """Return current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_package_version(package_name: str) -> str:
    """Return installed package version or 'unknown'."""
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except Exception:
        return "unknown"
