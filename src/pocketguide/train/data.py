"""
SFT dataset and collator for causal LM fine-tuning (Milestone 5 — Lesson 5.2).

Provides build_sft_dataset, SFTDataset, format_chat, and DataCollatorForPocketGuideSFT
with target-only loss masking. No model download required for tests.
"""

import json
from pathlib import Path
from typing import Any, Iterator

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object  # type: ignore[misc, assignment]

# Tokenizer type: any object with encode/decode and optional apply_chat_template
TOKENIZER_ANY = Any


class SFTDataset(Dataset):
    """Dataset that wraps a list of SFT records (id, messages, target, metadata). Picklable for DataLoader num_workers > 0."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.items[i]


def build_sft_dataset(jsonl_path: Path) -> Iterator[dict[str, Any]]:
    """Yield SFT examples from a JSONL file.

    Each line must be a JSON object with: id, messages, target, metadata.

    Args:
        jsonl_path: Path to train_sft.jsonl or val_sft.jsonl.

    Yields:
        Dicts with keys: id, messages, target, metadata.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"SFT JSONL not found: {path}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for key in ("id", "messages", "target", "metadata"):
                if key not in obj:
                    raise ValueError(f"SFT record missing required key: {key}")
            yield obj


def format_chat(messages: list[dict[str, str]], tokenizer: TOKENIZER_ANY) -> str:
    """Format chat messages into a single string for tokenization.

    Uses tokenizer.apply_chat_template if available (add_generation_prompt=False);
    otherwise falls back to a simple role/content concatenation.

    Args:
        messages: List of {"role": "system"|"user"|"assistant", "content": "..."}.
        tokenizer: Hugging Face tokenizer (or stub with encode/decode).

    Returns:
        Single string representing the prompt (no assistant response).
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.apply_chat_template is not None:
        try:
            # add_generation_prompt=False so we don't add assistant turn
            out = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if isinstance(out, str):
                return out
            # Some tokenizers return list of token ids
            if isinstance(out, list):
                return tokenizer.decode(out, skip_special_tokens=True)
        except Exception:
            pass
    # Fallback: simple concatenation
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
    return "\n".join(parts) if parts else ""


class DataCollatorForPocketGuideSFT:
    """Collator for SFT: concatenates formatted chat + target, masks labels on prompt.

    Produces batches of input_ids and labels with -100 on prompt positions
    so loss is computed only on target tokens. Supports variable-length
    sequences with padding.
    """

    def __init__(
        self,
        tokenizer: TOKENIZER_ANY,
        max_length: int = 2048,
        pad_to_multiple_of: int | None = 8,
        pad_token_id: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self._pad_id = pad_token_id
        if self._pad_id is None and hasattr(tokenizer, "pad_token_id"):
            self._pad_id = tokenizer.pad_token_id
        if self._pad_id is None and hasattr(tokenizer, "eos_token_id"):
            self._pad_id = tokenizer.eos_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of SFT examples into padded input_ids and labels.

        Each feature must have 'messages' and 'target'. Prompt (formatted messages)
        is masked with -100 in labels; target tokens get real token ids in labels.
        """
        if not features:
            return {"input_ids": [], "labels": [], "attention_mask": []}

        input_ids_list = []
        labels_list = []

        for feat in features:
            messages = feat["messages"]
            target = feat["target"]
            if not isinstance(target, str):
                target = json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

            prompt_str = format_chat(messages, self.tokenizer)
            full_text = prompt_str + target

            prompt_ids = self.tokenizer.encode(
                prompt_str,
                add_special_tokens=True,
                truncation=False,
            )
            if not isinstance(prompt_ids, list):
                prompt_ids = prompt_ids.tolist()

            full_ids = self.tokenizer.encode(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
            )
            if not isinstance(full_ids, list):
                full_ids = full_ids.tolist()

            # Labels: -100 for prompt span, token ids for target span
            num_prompt = len(prompt_ids)
            # If tokenizer adds special tokens differently for full text, align by length
            if len(full_ids) <= num_prompt:
                target_labels = full_ids
                prompt_len = 0
            else:
                prompt_len = num_prompt
                target_labels = full_ids.copy()
                for i in range(prompt_len):
                    target_labels[i] = -100

            labels_list.append(target_labels)
            input_ids_list.append(full_ids)

        # Pad to same length in batch
        max_len = max(len(ids) for ids in input_ids_list)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        max_len = min(max_len, self.max_length)
        pad_id = self._pad_id if self._pad_id is not None else 0

        padded_input_ids = []
        padded_labels = []
        attention_mask = []

        for ids, labs in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad_len)
            # -100 for padding in labels so they are ignored in loss
            padded_labels.append(labs + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        out = {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": attention_mask,
        }
        if torch is not None:
            out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
            out["labels"] = torch.tensor(out["labels"], dtype=torch.long)
            out["attention_mask"] = torch.tensor(out["attention_mask"], dtype=torch.long)
        return out
