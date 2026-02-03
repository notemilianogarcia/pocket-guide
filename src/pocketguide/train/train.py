"""
Config-driven training entry point (Milestone 5 — Lesson 5.1 / 5.3).

Validates config and datasets, creates reproducible run directories, supports
--dry_run, and runs LoRA training end-to-end. Portable: CPU, MPS, CUDA via Accelerate.
"""

import argparse
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import yaml

from pocketguide.utils.run_id import make_run_id
from pocketguide.utils.run_logging import (
    compute_file_hash,
    get_git_commit,
    get_package_version,
    write_json,
    write_yaml,
)

# Optional torch for device/precision resolution (no model load in dry_run)
try:
    import torch
except ImportError:
    torch = None
try:
    import numpy as np
except ImportError:
    np = None


# --- Config validation -------------------------------------------------------

REQUIRED_CONFIG_KEYS = {
    "base_model_id",
    "data",
    "output",
    "training",
    "lora",
    "runtime",
    "logging",
}

REQUIRED_DATA_KEYS = {"train_path", "val_path"}
REQUIRED_OUTPUT_KEYS = {"runs_dir"}

# Keys that are allowed to differ between v1 and v2 (data paths + the single experiment change)
ALLOWED_DIFF_KEYS = frozenset({
    "data.train_path", "data.val_path", "data.train_sft_path", "data.val_sft_path",
})


def _get_nested(cfg: dict[str, Any], dot_key: str) -> Any:
    """Get value by dot path (e.g. 'training.lr'). Returns None if any segment missing."""
    parts = dot_key.split(".")
    cur: Any = cfg
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten one level only: top-level keys become 'key' or 'key.subkey'."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict) and not any(
            isinstance(v.get(x), (dict, list)) for x in v
        ):
            for sk, sv in v.items():
                out[f"{key}.{sk}"] = sv
        else:
            out[key] = v
    return out


def validate_experiment_one_change(
    cfg_v2: dict[str, Any],
    cfg_v1: dict[str, Any],
    one_change_key: str,
) -> None:
    """
    Assert that v2 config differs from v1 only in data paths and the single one_change_key.
    Raises ValueError if any other difference is found.
    """
    flat1 = _flatten_dict(cfg_v1)
    flat2 = _flatten_dict(cfg_v2)
    allowed = set(ALLOWED_DIFF_KEYS) | {one_change_key}
    for key in set(flat1) | set(flat2):
        if key in allowed or key.startswith("experiment"):
            continue
        v1 = flat1.get(key)
        v2 = flat2.get(key)
        if v1 != v2:
            raise ValueError(
                f"Experiment one-change guardrail: v2 differs from v1 on '{key}' "
                f"(only data paths and '{one_change_key}' are allowed to differ)."
            )


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate top-level config structure."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML object (dict).")
    missing = REQUIRED_CONFIG_KEYS - set(cfg)
    if missing:
        raise ValueError(f"Config missing required top-level keys: {sorted(missing)}")
    data = cfg.get("data", {})
    missing_data = REQUIRED_DATA_KEYS - set(data)
    if missing_data:
        raise ValueError(f"Config 'data' missing required keys: {sorted(missing_data)}")
    output = cfg.get("output", {})
    if "runs_dir" not in output:
        raise ValueError("Config 'output' must contain 'runs_dir'.")
    return cfg


# --- Dataset validation ------------------------------------------------------

def _load_jsonl_records(path: Path, required_fields: list[str]) -> list[dict[str, Any]]:
    """Load JSONL and validate that each record has required fields."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {line_num}: {e}") from e
            missing = [k for k in required_fields if k not in item]
            if missing:
                raise ValueError(
                    f"Missing required fields in {path} at line {line_num}: {missing}"
                )
            records.append(item)
    return records


def validate_datasets(cfg: dict[str, Any], project_root: Path) -> tuple[list[dict], list[dict]]:
    """Validate train and val paths exist and records have prompt + response. Return (train_records, val_records)."""
    data = cfg["data"]
    train_path = project_root / data["train_path"].strip()
    val_path = project_root / data["val_path"].strip()
    required = ["prompt", "response"]
    train_records = _load_jsonl_records(train_path, required)
    val_records = _load_jsonl_records(val_path, required)
    if not train_records:
        raise ValueError(f"Train file has no valid records: {train_path}")
    return train_records, val_records


# --- Runtime resolution (device + precision) ----------------------------------

def resolve_device(device_cfg: str) -> str:
    """Resolve device: auto -> cuda | mps | cpu."""
    if device_cfg and device_cfg.lower() != "auto":
        return device_cfg.strip().lower()
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_precision(precision_cfg: str, device: str) -> str:
    """Resolve precision: auto -> bf16/fp16/fp32 by device; otherwise use config value."""
    if precision_cfg and precision_cfg.lower() != "auto":
        return precision_cfg.strip().lower()
    if device == "cuda" and torch is not None:
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    if device == "mps":
        return "fp32"  # MPS: fp32 for stability
    return "fp32"


# --- Run directory and artifacts ---------------------------------------------

def create_run_dir(cfg: dict[str, Any], run_id: str, project_root: Path) -> Path:
    """Create runs/train/<run_id> and return the path."""
    runs_dir = project_root / cfg["output"]["runs_dir"].strip()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_meta(
    cfg: dict[str, Any],
    run_id: str,
    run_dir: Path,
    project_root: Path,
    device: str,
    precision: str,
    v1_run_ref: str | None = None,
) -> dict[str, Any]:
    """Build meta.json contents (environment, versions, dataset hashes)."""
    data = cfg["data"]
    train_path = project_root / data["train_path"].strip()
    val_path = project_root / data["val_path"].strip()

    def hash_or_none(p: Path) -> str | None:
        if p.exists():
            return compute_file_hash(p)
        return None

    meta = {
        "run_id": run_id,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "package_versions": {
            "torch": get_package_version("torch"),
            "transformers": get_package_version("transformers"),
            "peft": get_package_version("peft"),
            "accelerate": get_package_version("accelerate"),
        },
        "resolved": {
            "device": device,
            "precision": precision,
        },
        "base_model_id": cfg["base_model_id"],
        "dataset": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_hash": hash_or_none(train_path),
            "val_hash": hash_or_none(val_path),
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "run_dir": str(run_dir),
    }
    exp = cfg.get("experiment")
    if isinstance(exp, dict):
        meta["experiment"] = {
            "name": exp.get("name"),
            "one_change": exp.get("one_change"),
            "one_change_key": exp.get("one_change_key"),
        }
        if v1_run_ref:
            meta["experiment"]["v1_run_ref"] = v1_run_ref
    return meta


# --- LoRA target_modules default ----------------------------------------------

# LLaMA / Mistral / most decoder-only transformers
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
# GPT-2 (used by dummy model)
GPT2_LORA_TARGET_MODULES = ["c_attn", "c_proj"]


def _resolve_lora_target_modules(cfg: dict[str, Any], model: Any) -> list[str]:
    """Resolve LoRA target_modules: 'auto' -> model-appropriate list; else use config list."""
    raw = cfg.get("lora", {}).get("target_modules", "auto")
    if raw != "auto" and raw is not None:
        if isinstance(raw, list):
            return [str(m) for m in raw]
        return [str(raw)]
    # Auto: pick by model type (GPT-2 uses c_attn/c_proj; LLaMA-style uses q_proj/etc.)
    model_type = getattr(getattr(model, "config", None), "model_type", None) or ""
    if model_type == "gpt2":
        return GPT2_LORA_TARGET_MODULES
    return DEFAULT_LORA_TARGET_MODULES


# --- Training loop (Lesson 5.3) -----------------------------------------------

def _set_seed(seed: int) -> None:
    """Set seed for torch, random, numpy."""
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed)


def _create_dummy_model_and_tokenizer(
    run_dir: Path,
    dtype_name: str,
) -> tuple[Any, Any]:
    """Create a tiny randomly initialized model and minimal tokenizer (no download)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    if torch is None:
        raise ImportError("torch required for dummy model")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(dtype_name, torch.float32)

    config = GPT2Config(
        vocab_size=1000,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=256,
        bos_token_id=998,
        eos_token_id=999,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model = model.to(dtype)

    # Minimal tokenizer: character-level encoding (no tokenizers lib for dummy)
    class _DummyTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 999
            self.bos_token_id = 998
            self.model_max_length = 256

        def encode(self, text, add_special_tokens=True, truncation=False, max_length=None):
            ids = [998] if add_special_tokens else []
            for c in text:
                ids.append(4 + (ord(c) % 256))
            if add_special_tokens:
                ids.append(999)
            if max_length and len(ids) > max_length:
                ids = ids[:max_length]
            return ids

        def decode(self, ids, skip_special_tokens=True):
            out = []
            for i in ids:
                if skip_special_tokens and i in (0, 998, 999):
                    continue
                if 4 <= i < 260:
                    out.append(chr(i - 4))
            return "".join(out)

    fast = _DummyTokenizer()
    return model, fast


def _load_model_and_tokenizer(
    base_model_id: str,
    precision: str,
    device_name: str,
) -> tuple[Any, Any]:
    """Load base model and tokenizer from HuggingFace (no dummy)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    torch_dtype = dtype_map.get(precision, "float32")
    if torch_dtype == "float16":
        import torch

        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        import torch

        dtype = torch.bfloat16
    else:
        import torch

        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    return model, tokenizer


def run_training(
    cfg: dict[str, Any],
    run_dir: Path,
    project_root: Path,
    device: str,
    precision: str,
    max_steps_override: int | None = None,
) -> None:
    """Run LoRA training: load SFT data, model, apply PEFT, train, save adapter and metrics."""
    from torch.utils.data import DataLoader

    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model, TaskType

    from pocketguide.train.data import DataCollatorForPocketGuideSFT, SFTDataset, build_sft_dataset

    _set_seed(cfg.get("training", {}).get("seed", 42))

    data_cfg = cfg.get("data", {})
    train_sft_path = project_root / data_cfg.get(
        "train_sft_path", "data/processed/sft/v1/train_sft.jsonl"
    ).strip()
    val_sft_path = project_root / data_cfg.get(
        "val_sft_path", "data/processed/sft/v1/val_sft.jsonl"
    ).strip()
    if not train_sft_path.exists():
        raise FileNotFoundError(f"SFT train file not found: {train_sft_path}")
    if not val_sft_path.exists():
        raise FileNotFoundError(f"SFT val file not found: {val_sft_path}")

    max_seq_len = int(data_cfg.get("max_seq_len", 2048))
    max_train_samples = data_cfg.get("max_train_samples")
    max_val_samples = data_cfg.get("max_val_samples")

    train_list = list(build_sft_dataset(train_sft_path))
    val_list = list(build_sft_dataset(val_sft_path))
    if max_train_samples is not None:
        rng = random.Random(cfg.get("training", {}).get("seed", 42))
        rng.shuffle(train_list)
        train_list = train_list[: max_train_samples]
    if max_val_samples is not None:
        rng = random.Random(cfg.get("training", {}).get("seed", 42) + 1)
        rng.shuffle(val_list)
        val_list = val_list[: max_val_samples]

    train_dataset = SFTDataset(train_list)
    val_dataset = SFTDataset(val_list)

    runtime_cfg = cfg.get("runtime", {})
    use_dummy = runtime_cfg.get("dummy_model", False)
    base_model_id = cfg.get("base_model_id", "")

    if use_dummy:
        model, tokenizer = _create_dummy_model_and_tokenizer(run_dir, precision)
    else:
        model, tokenizer = _load_model_and_tokenizer(base_model_id, precision, device)

    if runtime_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    collator = DataCollatorForPocketGuideSFT(
        tokenizer=tokenizer,
        max_length=max_seq_len,
        pad_to_multiple_of=8,
        pad_token_id=tokenizer.pad_token_id,
    )

    train_cfg = cfg.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 1))
    grad_accum = int(train_cfg.get("grad_accum_steps", 16))
    lr = float(train_cfg.get("lr", 2e-4))
    num_epochs = int(train_cfg.get("num_epochs", 1))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    eval_every = int(train_cfg.get("eval_every_steps", 100))
    save_every = int(train_cfg.get("save_every_steps", 200))
    log_every = int(cfg.get("logging", {}).get("log_every_steps", 10))
    max_steps = max_steps_override
    if max_steps is None:
        max_steps = train_cfg.get("max_steps")

    lora_cfg = cfg.get("lora", {})
    target_modules = _resolve_lora_target_modules(cfg, model)
    lora_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=precision if precision in ("fp16", "bf16") else "no",
    )
    # Dummy model uses a local tokenizer class that can't be pickled; use single process
    num_workers_train = 0 if use_dummy else runtime_cfg.get("num_workers", 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers_train,
        pin_memory=runtime_cfg.get("pin_memory", False),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = (
        min(max_steps, len(train_loader) * num_epochs)
        if max_steps is not None
        else len(train_loader) * num_epochs
    )
    if num_training_steps <= 0:
        num_training_steps = max_steps or 100
    from torch.optim.lr_scheduler import LambdaLR

    def _warmup_lr(step):
        if step < warmup_steps:
            return step / warmup_steps if warmup_steps else 1.0
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=_warmup_lr)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    run_id = run_dir.name
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("TRAINING START")
        print("=" * 60)
        print(f"  Run ID:           {run_id}")
        print(f"  Train samples:    {len(train_dataset)}")
        print(f"  Val samples:      {len(val_dataset)}")
        print(f"  Total steps:      {num_training_steps}")
        print(f"  LR:               {lr}")
        print(f"  Batch size:       {batch_size} (grad_accum: {grad_accum})")
        print(f"  Eval every:       {eval_every} steps")
        print(f"  Save every:       {save_every} steps")
        print("=" * 60 + "\n")

    metrics = {
        "run_id": run_id,
        "global_steps": 0,
        "train": [],
        "val": [],
        "final": {},
    }
    tokens_seen = 0
    global_step = 0
    best_val_loss = float("inf")

    def _write_metrics():
        write_json(run_dir / "train_metrics.json", metrics)

    def _eval():
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    out = model(**batch)
                    total_loss += out.loss.item()
                    n_batches += 1
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        raise
                    raise
        model.train()
        return total_loss / n_batches if n_batches else float("nan")

    try:
        model.train()
        for epoch in range(num_epochs):
            for step, batch in enumerate(train_loader):
                if max_steps is not None and global_step >= max_steps:
                    break
                with accelerator.accumulate(model):
                    out = model(**batch)
                    loss = out.loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    scheduler.step()
                    tokens_seen += batch["input_ids"].numel()
                    if global_step % log_every == 0:
                        metrics["train"].append({"step": global_step, "loss": loss.item()})
                        if accelerator.is_main_process:
                            pct = 100.0 * global_step / num_training_steps if num_training_steps else 0
                            print(
                                f"  [Epoch {epoch + 1}/{num_epochs}] step {global_step}/{num_training_steps} "
                                f"({pct:.1f}%) | train_loss: {loss.item():.4f}"
                            )
                    global_step += 1

                    if eval_every and global_step % eval_every == 0:
                        val_loss = _eval()
                        metrics["val"].append({"step": global_step, "loss": val_loss})
                        best_val_loss = min(best_val_loss, val_loss)
                        if accelerator.is_main_process:
                            print(
                                f"  >> Validation at step {global_step} | val_loss: {val_loss:.4f} "
                                f"(best so far: {best_val_loss:.4f})"
                            )

                    if save_every and global_step % save_every == 0 and accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(run_dir / "adapter")
                        _write_metrics()
                        print(f"  >> Checkpoint saved at step {global_step} -> {run_dir / 'adapter'}")

                if max_steps is not None and global_step >= max_steps:
                    break
            if max_steps is not None and global_step >= max_steps:
                break
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
            print(
                "OOM: Try lower batch_size, grad_accum_steps, or max_seq_len.",
                file=sys.stderr,
            )
        raise

    # Final eval and save
    val_loss = _eval()
    metrics["val"].append({"step": global_step, "loss": val_loss})
    metrics["global_steps"] = global_step
    metrics["final"] = {
        "val_loss": val_loss,
        "train_steps": global_step,
        "tokens_seen": tokens_seen,
    }
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        adapter_dir = run_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(adapter_dir)
        _write_metrics()
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Adapter:     {adapter_dir}")
        print(f"  Metrics:     {run_dir / 'train_metrics.json'}")
        print(f"  Final steps: {global_step}")
        print(f"  Tokens seen: {tokens_seen:,}")
        print(f"  Final val_loss: {val_loss:.4f}")
        print("=" * 60 + "\n")


# --- Main entry point --------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PocketGuide training entry point (config-driven, dry_run and LoRA training)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to train config YAML (e.g. configs/train_lora.yaml or configs/train_lora_v2.yaml)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate config and datasets, write artifacts only; no model load.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override config: stop training after this many steps.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Use this run ID instead of generating one (deterministic runs).",
    )
    parser.add_argument(
        "--v1_run_dir",
        type=str,
        default=None,
        help="Path to v1 run directory (runs/train/<v1_run_id>) for one-change guardrail; requires config with experiment block.",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    try:
        cfg = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    # Experiment block: require name and one_change; optional one-change guardrail vs v1
    exp = cfg.get("experiment")
    v1_run_ref: str | None = None
    if isinstance(exp, dict):
        if exp.get("name") is None or exp.get("name") == "":
            print("Config error: experiment block requires experiment.name", file=sys.stderr)
            sys.exit(1)
        if exp.get("one_change") is None or exp.get("one_change") == "":
            print("Config error: experiment block requires experiment.one_change", file=sys.stderr)
            sys.exit(1)
        one_change_key = exp.get("one_change_key")
        if args.v1_run_dir:
            v1_run_dir = Path(args.v1_run_dir)
            if not v1_run_dir.is_absolute():
                v1_run_dir = project_root / v1_run_dir
            v1_config_path = v1_run_dir / "config.yaml"
            if not v1_config_path.exists():
                print(f"Config error: v1 run config not found: {v1_config_path}", file=sys.stderr)
                sys.exit(1)
            if not one_change_key:
                print("Config error: experiment.one_change_key required when --v1_run_dir is set", file=sys.stderr)
                sys.exit(1)
            try:
                cfg_v1 = load_config(v1_config_path)
                validate_experiment_one_change(cfg, cfg_v1, one_change_key)
            except ValueError as e:
                print(f"Experiment guardrail: {e}", file=sys.stderr)
                sys.exit(1)
            v1_run_ref = str(v1_run_dir)

    try:
        train_records, val_records = validate_datasets(cfg, project_root)
    except (FileNotFoundError, ValueError) as e:
        print(f"Dataset error: {e}", file=sys.stderr)
        sys.exit(1)

    device = resolve_device(cfg.get("runtime", {}).get("device", "auto"))
    precision = resolve_precision(
        cfg.get("runtime", {}).get("precision", "auto"),
        device,
    )

    if args.run_id:
        run_id = args.run_id
    elif isinstance(exp, dict) and exp.get("name") == "v2":
        run_id = make_run_id() + "-v2"
    else:
        run_id = make_run_id()

    try:
        run_dir = create_run_dir(cfg, run_id, project_root)
    except FileExistsError as e:
        print(f"Run directory already exists: {e}", file=sys.stderr)
        sys.exit(1)

    # Snapshot config used (exact copy)
    config_snapshot_path = run_dir / "config.yaml"
    write_yaml(config_snapshot_path, cfg)

    meta = build_meta(cfg, run_id, run_dir, project_root, device, precision, v1_run_ref=v1_run_ref)
    meta_path = run_dir / "meta.json"
    write_json(meta_path, meta)

    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    print(f"Config snapshot: {config_snapshot_path}")
    print(f"Metadata: {meta_path}")
    print(f"Resolved device: {device}, precision: {precision}")
    print(f"Train records: {len(train_records)}, val records: {len(val_records)}")

    if args.dry_run:
        print("Dry run: skipping model load and training.")
        return

    run_training(
        cfg,
        run_dir,
        project_root,
        device,
        precision,
        max_steps_override=args.max_steps,
    )


if __name__ == "__main__":
    main()
