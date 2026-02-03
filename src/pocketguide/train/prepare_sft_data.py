"""
Prepare SFT datasets and fixed prompt suite (Milestone 5 — Lesson 5.2).

Reads split JSONL (train/val), converts to SFT format with deterministic
target serialization, writes train_sft.jsonl / val_sft.jsonl and
eval/suites/fixed20_v1.jsonl (deterministic sample of 20 prompts from val).

v5: Adds schema validation - drops examples that fail envelope+payload validation
to ensure model only sees schema-compliant outputs.
"""

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

from pocketguide.eval.parsing import parse_and_validate

SYSTEM_INSTRUCTION = (
    "Return JSON only. Output a single JSON object (not a list or array). "
    "Match the PocketGuide response envelope: summary, assumptions, uncertainty_notes, "
    "next_steps, verification_steps, payload_type, payload. "
    "Always include verification_steps (array of strings) and payload_type "
    "(one of: itinerary, checklist, decision_tree, procedure). "
    "For itinerary payloads, every activity item must have \"time_block\" (e.g. \"morning\", \"14:00-16:00\"). "
    "No markdown."
)

# Required envelope keys (v0/response_envelope.schema.json); model must see all in every example
ENVELOPE_KEYS = (
    "summary",
    "assumptions",
    "uncertainty_notes",
    "next_steps",
    "verification_steps",
    "payload_type",
    "payload",
)
VALID_PAYLOAD_TYPES = frozenset({"itinerary", "checklist", "decision_tree", "procedure"})


def _ensure_envelope(response: dict[str, Any], record_payload_type: str | None) -> dict[str, Any]:
    """Ensure response has all 7 envelope keys so the model always sees full structure in SFT.
    Fills missing keys with sensible defaults so we never train on incomplete examples.
    """
    out = dict(response)
    if not out.get("summary"):
        out["summary"] = out.get("summary") or "See payload."
    if "assumptions" not in out or not isinstance(out["assumptions"], list):
        out["assumptions"] = out.get("assumptions") if isinstance(out.get("assumptions"), list) else []
    if not out.get("uncertainty_notes"):
        out["uncertainty_notes"] = out.get("uncertainty_notes") or "Verify details with official sources."
    if "next_steps" not in out or not isinstance(out["next_steps"], list):
        out["next_steps"] = out.get("next_steps") if isinstance(out.get("next_steps"), list) else []
    if "verification_steps" not in out or not isinstance(out["verification_steps"], list):
        vs = out.get("verification_steps") if isinstance(out.get("verification_steps"), list) else []
        out["verification_steps"] = vs if vs else ["Check official sources for current information."]
    pt = out.get("payload_type") or record_payload_type
    if pt not in VALID_PAYLOAD_TYPES:
        pt = "checklist"
    out["payload_type"] = pt
    if "payload" not in out or not isinstance(out["payload"], dict):
        out["payload"] = out.get("payload") if isinstance(out.get("payload"), dict) else {}
    out["payload"] = _normalize_payload_for_sft(out["payload"], out["payload_type"])
    return out


def _normalize_payload_for_sft(payload: dict[str, Any], payload_type: str) -> dict[str, Any]:
    """Tighten SFT data: normalize payload so model only sees schema-consistent keys.
    - itinerary: ensure every activity item has time_block (copy from time_buffer if present, else default); remove time_buffer.
    - decision_tree: ensure title and nodes are present (required by schema).
    - checklist: ensure title and groups are present (required by schema).
    """
    if not isinstance(payload, dict):
        return payload
    out = copy.deepcopy(payload)
    
    if payload_type == "itinerary":
        trip_days = out.get("trip_days")
        if not isinstance(trip_days, list):
            return out
        for day in trip_days:
            if not isinstance(day, dict):
                continue
            items = day.get("items")
            if not isinstance(items, list):
                continue
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                # time_buffer -> time_block so model never sees wrong key
                if "time_buffer" in item and "time_block" not in item:
                    item = dict(item)
                    item["time_block"] = item.pop("time_buffer", "See summary")
                elif "time_block" not in item:
                    item = dict(item)
                    item["time_block"] = item.get("time_buffer", "See summary")
                    item.pop("time_buffer", None)
                else:
                    item = dict(item)
                    item.pop("time_buffer", None)
                items[i] = item
    
    elif payload_type == "decision_tree":
        # Ensure required fields: title and nodes
        if "title" not in out or not out.get("title"):
            out["title"] = out.get("title") or "Decision Tree"
        if "nodes" not in out or not isinstance(out.get("nodes"), list):
            out["nodes"] = out.get("nodes") if isinstance(out.get("nodes"), list) else []
    
    elif payload_type == "checklist":
        # Ensure required fields: title and groups
        if "title" not in out or not out.get("title"):
            out["title"] = out.get("title") or "Checklist"
        if "groups" not in out or not isinstance(out.get("groups"), list):
            out["groups"] = out.get("groups") if isinstance(out.get("groups"), list) else []
    
    return out


def _serialize_target(response: dict[str, Any]) -> str:
    """Deterministic JSON serialization for SFT target."""
    return json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_to_sft(
    record: dict[str, Any],
    source_split: str,
    validate: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert a split record to SFT format (id, messages, target, metadata).
    
    v5: If validate=True, validates the response against envelope+payload schemas.
    Returns (sft_record, error_message). If validation fails, returns (None, error_message).
    
    Args:
        record: Split record with prompt, response, payload_type, etc.
        source_split: "train" or "val"
        validate: If True, validate response against schemas and drop if invalid
        
    Returns:
        Tuple of (sft_record or None, error_message or None)
    """
    rid = record.get("id", "")
    prompt = record.get("prompt", "")
    response = record.get("response", {})
    if not isinstance(response, dict):
        response = {"summary": str(response)}
    response = _ensure_envelope(response, record.get("payload_type"))

    # v5: Validate before including in SFT
    if validate:
        response_str = _serialize_target(response)
        parse_result = parse_and_validate(response_str, strict_json=False, validate_payload=True)
        if not parse_result.success:
            error_msg = parse_result.error.message if parse_result.error else "Validation failed"
            return None, f"{rid}: {error_msg}"

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt},
    ]
    target = _serialize_target(response)
    metadata = {
        "payload_type": record.get("payload_type"),
        "category": record.get("category"),
        "difficulty": record.get("difficulty"),
        "region_tags": record.get("region_tags", []),
        "source_split": source_split,
    }
    return {
        "id": rid,
        "messages": messages,
        "target": target,
        "metadata": metadata,
    }, None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file; require prompt, response; id default to record_<line_num> if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for key in ("prompt", "response"):
                if key not in obj:
                    raise ValueError(f"Record at {path}:{line_num} missing key: {key}")
            if "id" not in obj:
                obj["id"] = f"record_{line_num}"
            records.append(obj)
    return records


def _deterministic_sample(
    records: list[dict[str, Any]],
    n: int,
    seed: int,
    prefer_diversity: bool = True,
) -> list[dict[str, Any]]:
    """Select up to n records deterministically. Light diversity by payload_type/difficulty."""
    if len(records) <= n:
        order = list(records)
        rng = random.Random(seed)
        rng.shuffle(order)
        return order
    rng = random.Random(seed)
    if prefer_diversity:
        # Group by (payload_type, difficulty) then sample from each
        groups: dict[tuple[Any, Any], list[dict]] = {}
        for r in records:
            key = (r.get("payload_type"), r.get("difficulty"))
            groups.setdefault(key, []).append(r)
        for key in groups:
            rng.shuffle(groups[key])
        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)
        out = []
        idx = 0
        while len(out) < n and idx < 1000:
            for key in group_keys:
                if groups[key] and len(out) < n:
                    out.append(groups[key].pop(0))
            idx += 1
        if len(out) < n:
            flat = [r for r in records if r not in out]
            rng.shuffle(flat)
            for r in flat:
                if len(out) >= n:
                    break
                out.append(r)
        return out[:n]
    indices = list(range(len(records)))
    rng.shuffle(indices)
    return [records[i] for i in indices[:n]]


def build_fixed_prompts(
    val_records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build fixed prompt suite: prefer val, then train; deterministic sample of n."""
    pool = list(val_records) if val_records else []
    if len(pool) < n and train_records:
        pool = list(val_records) + list(train_records)
    selected = _deterministic_sample(pool, n, seed, prefer_diversity=True)
    return [
        {
            "id": r.get("id", ""),
            "prompt": r.get("prompt", ""),
            "payload_type": r.get("payload_type"),
            "category": r.get("category"),
            "difficulty": r.get("difficulty"),
            "region_tags": r.get("region_tags", []),
        }
        for r in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SFT JSONL and fixed prompt suite (Lesson 5.2)."
    )
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=Path("data/processed/splits/v1"),
        help="Directory containing train.jsonl and val.jsonl",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/processed/sft/v1"),
        help="Output directory for train_sft.jsonl and val_sft.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle and sampling",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap train set size (deterministic selection)",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Cap val set size (deterministic selection)",
    )
    parser.add_argument(
        "--fixed_prompts_out",
        type=Path,
        default=Path("eval/suites/fixed20_v1.jsonl"),
        help="Output path for fixed prompt suite (20 prompts)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate all SFT examples against schemas and drop invalid ones (v5: default True)",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip validation (for backward compatibility)",
    )
    args = parser.parse_args()

    splits_dir = args.splits_dir
    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "val.jsonl"

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)

    rng = random.Random(args.seed)

    # Deterministic shuffle and optional cap
    train_order = list(train_records)
    rng.shuffle(train_order)
    if args.max_train_samples is not None:
        train_order = train_order[: args.max_train_samples]

    val_order = list(val_records)
    rng.shuffle(val_order)
    if args.max_val_samples is not None:
        val_order = val_order[: args.max_val_samples]

    # v5: Validate and drop invalid examples
    sft_train = []
    sft_val = []
    dropped_train = []
    dropped_val = []
    
    for r in train_order:
        sft_rec, error = record_to_sft(r, "train", validate=args.validate)
        if sft_rec is not None:
            sft_train.append(sft_rec)
        elif error:
            dropped_train.append(error)
    
    for r in val_order:
        sft_rec, error = record_to_sft(r, "val", validate=args.validate)
        if sft_rec is not None:
            sft_val.append(sft_rec)
        elif error:
            dropped_val.append(error)
    
    if dropped_train or dropped_val:
        print(f"v5 validation: Dropped {len(dropped_train)} train, {len(dropped_val)} val examples")
        if dropped_train:
            print(f"  Train drops (first 5): {dropped_train[:5]}")
        if dropped_val:
            print(f"  Val drops (first 5): {dropped_val[:5]}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_sft_path = out_dir / "train_sft.jsonl"
    val_sft_path = out_dir / "val_sft.jsonl"

    for path, items in [(train_sft_path, sft_train), (val_sft_path, sft_val)]:
        with open(path, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {train_sft_path} ({len(sft_train)} records)")
    print(f"Wrote {val_sft_path} ({len(sft_val)} records)")

    # Fixed prompt suite (20 from val, else val+train)
    fixed = build_fixed_prompts(val_records, train_records, n=20, seed=args.seed)
    fixed_path = args.fixed_prompts_out
    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fixed_path, "w", encoding="utf-8") as f:
        for obj in fixed:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if len(fixed) < 20:
        print(f"Fixed suite has {len(fixed)} prompts (dataset smaller than 20). Wrote {fixed_path}")
    else:
        print(f"Wrote {fixed_path} (20 prompts)")


if __name__ == "__main__":
    main()
