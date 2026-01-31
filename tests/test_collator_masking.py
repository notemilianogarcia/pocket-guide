"""
Tests for SFT data collator (Milestone 5 — Lesson 5.2).

Uses a dummy tokenizer stub; no model download.
"""

import json
from typing import Any

import pytest

from pocketguide.train.data import DataCollatorForPocketGuideSFT, format_chat


class DummyTokenizer:
    """Stub tokenizer: each character maps to ord(c); space = 0; special tokens 999, 998."""

    def __init__(self, pad_token_id: int = 0, eos_token_id: int = 1):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self._vocab = {}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        ids = []
        if add_special_tokens:
            ids.append(999)
        for c in text:
            ids.append(ord(c) % 256)
        if add_special_tokens:
            ids.append(self.eos_token_id)
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        out = []
        for i in ids:
            if skip_special_tokens and i in (999, 998, self.eos_token_id, self.pad_token_id):
                continue
            if 0 <= i < 256:
                out.append(chr(i))
        return "".join(out)


def test_format_chat_fallback() -> None:
    """format_chat uses simple fallback when no apply_chat_template."""
    tok = DummyTokenizer()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    out = format_chat(messages, tok)
    assert "<|system|>" in out and "You are helpful." in out
    assert "<|user|>" in out and "Hello" in out


def test_collator_labels_masked_for_prompt() -> None:
    """Labels are -100 for prompt portion and not masked for target portion."""
    tok = DummyTokenizer(pad_token_id=0, eos_token_id=1)
    collator = DataCollatorForPocketGuideSFT(
        tokenizer=tok,
        max_length=512,
        pad_to_multiple_of=None,
        pad_token_id=0,
    )
    messages = [
        {"role": "system", "content": "JSON only."},
        {"role": "user", "content": "Say hi"},
    ]
    target = '{"summary":"Hi"}'
    features = [
        {"messages": messages, "target": target},
        {"messages": messages, "target": '{"summary":"Bye"}'},
    ]
    batch = collator(features)

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    assert input_ids is not None and labels is not None
    # Convert to list if tensor for indexing
    if hasattr(input_ids, "tolist"):
        labels_np = labels.tolist()
    else:
        labels_np = labels

    # First example: prompt tokens (system + user) should be -100; target tokens not
    first_labels = labels_np[0]
    prompt_str = format_chat(messages, tok)
    prompt_ids = tok.encode(prompt_str, add_special_tokens=True)
    num_prompt = len(prompt_ids)
    # Collator masks first num_prompt positions
    prompt_label_slice = first_labels[:num_prompt]
    assert all(x == -100 for x in prompt_label_slice), "Prompt positions should be -100"
    # At least one non-masked (target) position
    target_slice = first_labels[num_prompt:]
    non_masked = [x for x in target_slice if x != -100 and x != 0]
    assert len(non_masked) >= 1, "Target portion should have non-masked token ids"
