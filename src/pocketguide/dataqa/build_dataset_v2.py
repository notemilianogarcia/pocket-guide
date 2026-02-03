"""
Build dataset v2 from v1 with targeted interventions (Lesson 7.2).

Loads v1 (or v1_clean), applies optional stricter rejection, generates hard prompts,
optionally generates synthetic samples via teacher, oversamples v1 matches for target
failure modes, then writes dataset_v2.jsonl, manifest, and stats with full traceability.
"""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from pocketguide.dataqa.failure_tagging import tag_failure_modes
from pocketguide.dataqa.hard_prompt_generator import generate_hard_prompts


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file for reproducibility."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _stricter_rejection_rule(record: dict[str, Any], target_failure_modes: list[str]) -> tuple[bool, str | None]:
    """
    Apply stricter rejection aligned to target failure modes only.

    Returns (pass, rejection_reason). pass=True means keep the record.
    """
    response = record.get("response") or {}
    if not isinstance(response, dict):
        return False, "response_invalid"

    # Reject if verification_steps missing or empty (aligned to missing_verification_steps)
    if "missing_verification_steps" in target_failure_modes:
        steps = response.get("verification_steps")
        if not steps or (isinstance(steps, list) and len(steps) == 0):
            return False, "verification_steps_empty"
        # Reject if only generic (e.g. "check online")
        if isinstance(steps, list):
            steps_text = " ".join(s for s in steps if isinstance(s, str)).lower()
            if steps_text and ("check online" in steps_text or "verify online" in steps_text) and len(steps) <= 1:
                return False, "verification_steps_generic_only"

    # Reject if uncertainty_notes missing or empty (aligned to required-field / quality)
    if "missing_verification_steps" in target_failure_modes or "missing_next_steps" in target_failure_modes:
        unc = response.get("uncertainty_notes")
        if unc is None or (isinstance(unc, str) and not unc.strip()):
            return False, "uncertainty_notes_empty"

    return True, None


def apply_stricter_rejection(
    records: list[dict[str, Any]],
    target_failure_modes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split records into kept and removed; removed include reason.

    Returns (kept, removed_with_reasons).
    """
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for r in records:
        pass_ok, reason = _stricter_rejection_rule(r, target_failure_modes)
        if pass_ok:
            kept.append(r)
        else:
            removed.append({**r, "_rejection_reason": reason})
    return kept, removed


def oversample_v1_matches(
    records: list[dict[str, Any]],
    target_failure_modes: list[str],
    factor: float,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Add oversampled copies of v1 records that match target failure mode tags.

    factor is capped (e.g. 1.5 => add up to 50% more copies of matches).
    Each copy gets a new id and source_id pointing to original.
    """
    if factor <= 1.0:
        return []
    rng = random.Random(seed)
    matches: list[dict[str, Any]] = []
    for r in records:
        tags = tag_failure_modes(r)
        if tags & set(target_failure_modes):
            matches.append(r)
    if not matches:
        return []
    # Number of extra copies: (factor - 1) * len(matches), capped
    add_count = max(0, int((factor - 1.0) * len(matches)))
    if add_count == 0:
        return []
    out: list[dict[str, Any]] = []
    for i in range(add_count):
        orig = rng.choice(matches)
        copy = dict(orig)
        orig_id = orig.get("id", "unknown")
        new_id = f"v2_oversample_{orig_id}_{i}"
        copy["id"] = new_id
        copy["source"] = "v1"
        copy["source_id"] = orig_id
        out.append(copy)
    return out


def ensure_traceability_v1(record: dict[str, Any]) -> dict[str, Any]:
    """Ensure v1-origin record has source and source_id."""
    out = dict(record)
    if "source" not in out:
        out["source"] = "v1"
    if "source_id" not in out:
        out["source_id"] = out.get("id", "unknown")
    return out


def load_synthetic_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load cache of recipe_id -> {raw_text, teacher_model} from JSONL."""
    cache: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rid = entry.get("recipe_id")
            if rid:
                cache[rid] = {
                    "raw_text": entry.get("raw_text", ""),
                    "teacher_model": entry.get("teacher_model"),
                }
    return cache


def save_synthetic_cache(cache_path: Path, cache: dict[str, dict[str, Any]]) -> None:
    """Write cache to JSONL (one line per recipe_id)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for recipe_id, data in sorted(cache.items()):
            f.write(
                json.dumps(
                    {"recipe_id": recipe_id, **data},
                    ensure_ascii=False,
                )
                + "\n"
            )


# Allowed envelope payload_type enum (must match response_envelope.schema.json)
_ENVELOPE_PAYLOAD_TYPES = frozenset({"itinerary", "checklist", "decision_tree", "procedure"})

# Map teacher-style payload_type values to canonical enum
_PAYLOAD_TYPE_ALIASES: dict[str, str] = {
    "budget_checklist": "checklist",
    "budgeting_checklist": "checklist",
    "one_day_itinerary": "itinerary",
}


def normalize_synthetic_envelope(data: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize teacher-style envelope so it passes envelope schema validation.

    - Coerces assumptions, next_steps, verification_steps from string to array of strings.
    - Maps payload_type to canonical enum (e.g. budget_checklist -> checklist).
    - Wraps payload in an object if teacher returned an array (envelope requires object).
    """
    if not isinstance(data, dict):
        # Lenient parse can return a list (e.g. teacher wrapped envelope in an array); unwrap once.
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            data = data[0]
        else:
            return data
    if not isinstance(data, dict):
        return data
    out = dict(data)
    # Coerce string -> [string] for array fields
    for key in ("assumptions", "next_steps", "verification_steps"):
        val = out.get(key)
        if isinstance(val, str):
            val = val.strip()
            out[key] = [val] if val else []
        elif isinstance(val, list):
            out[key] = [str(x).strip() for x in val if str(x).strip()]
        elif val is None:
            out[key] = []
    # Map payload_type to enum
    pt = out.get("payload_type")
    if isinstance(pt, str):
        pt = pt.strip().lower()
        out["payload_type"] = _PAYLOAD_TYPE_ALIASES.get(pt, pt) if pt not in _ENVELOPE_PAYLOAD_TYPES else pt
        if out["payload_type"] not in _ENVELOPE_PAYLOAD_TYPES:
            out["payload_type"] = "checklist"
    # Envelope schema requires payload to be object; teacher sometimes returns array
    payload = out.get("payload")
    if isinstance(payload, list):
        out["payload"] = {"items": payload}
    elif payload is not None and not isinstance(payload, dict):
        out["payload"] = {"value": payload}
    return out


def build_synthetic_records(
    hard_prompts: list[dict[str, Any]],
    teacher_config_path: Path | None,
    project_root: Path,
    verbose: bool = False,
    cache_path: Path | None = None,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """
    Generate synthetic records for hard prompts via teacher (if configured).

    When cache_path is set, loads existing responses from cache and uses them instead of
    calling the teacher (saving API calls). New responses are written back to the cache.
    Re-running with the same cache re-parses cached raw text with current parse/validate logic.

    Returns (accepted_records, total_attempted, recipe_ids).
    When teacher_config_path is None or teacher not available, returns ([], 0, []).
    """
    if not hard_prompts:
        return [], 0, []
    if not teacher_config_path or not teacher_config_path.exists():
        return [], 0, []

    # Optional teacher integration: load config and router, call generate, parse_and_validate.
    # If teacher call fails or is not configured, return empty (no network in tests).
    try:
        from pocketguide.eval.parsing import parse_and_validate
        from pocketguide.data_generation.generate_critiques import (
            create_teacher_router,
            load_teacher_config,
        )
        from pocketguide.teachers.base import TeacherRequest
    except ImportError:
        return [], 0, []

    cache: dict[str, dict[str, Any]] = load_synthetic_cache(cache_path) if cache_path else {}
    cache_updated = False

    config = load_teacher_config(teacher_config_path)
    if not config:
        return [], 0, []

    # Only create router if we might need to call the API (some prompts not in cache)
    need_api = any(f"v2_synthetic_{hp.get('id', 'unknown')}" not in cache for hp in hard_prompts)
    router = create_teacher_router(config, override_dry_run=False) if need_api else None
    if need_api and router is None:
        return [], 0, []

    accepted: list[dict[str, Any]] = []
    recipe_ids: list[str] = []
    system_msg = "You are an expert travel assistant. Output valid JSON only with envelope fields: summary, assumptions, uncertainty_notes, next_steps, verification_steps, payload_type, payload. No prose. No markdown."
    n_total = len(hard_prompts)
    for i, hp in enumerate(hard_prompts):
        if verbose and (i + 1) % 10 == 0:
            print(f"[dataset-v2] Synthetic {i + 1}/{n_total}...")
        prompt = hp.get("prompt", "")
        recipe_id = f"v2_synthetic_{hp.get('id', 'unknown')}"
        recipe_ids.append(recipe_id)

        if recipe_id in cache:
            text = cache[recipe_id].get("raw_text", "")
            teacher_model = cache[recipe_id].get("teacher_model") or "cached"
        else:
            if router is None:
                continue
            req = TeacherRequest(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0.3,
            )
            try:
                raw = router.generate(req)
                text = raw.text
                teacher_model = raw.model
                cache[recipe_id] = {"raw_text": text, "teacher_model": teacher_model}
                cache_updated = True
            except Exception as e:
                if verbose and len(accepted) == 0 and i == 0:
                    print(f"[dataset-v2] First synthetic attempt failed (teacher error): {e}")
                continue

        # Parse with normalizer so teacher string/array conventions pass envelope schema;
        # skip payload schema so variable teacher payload shapes are accepted.
        result = parse_and_validate(
            text,
            strict_json=True,
            validate_payload=False,
            normalizer=normalize_synthetic_envelope,
        )
        if not result.success:
            result = parse_and_validate(
                text,
                strict_json=False,
                validate_payload=False,
                normalizer=normalize_synthetic_envelope,
            )
        if not result.success or not result.data:
            if verbose and len(accepted) == 0 and i == 0 and result.error:
                err = result.error
                print(f"[dataset-v2] First synthetic rejection: {err.error_type} — {err.message[:120]}...")
            continue
        envelope = result.data
        record = {
            "id": hp.get("id", recipe_id),
            "prompt": prompt,
            "response": envelope,
            "payload_type": hp.get("payload_type", "checklist"),
            "category": hp.get("category", "general"),
            "difficulty": hp.get("difficulty", "medium"),
            "region_tags": hp.get("region_tags", []),
            "source": "synthetic_v2",
            "recipe_id": recipe_id,
            "teacher_model": teacher_model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_failure_mode": hp.get("target_failure_mode", ""),
        }
        accepted.append(record)

    if cache_path and cache_updated:
        save_synthetic_cache(cache_path, cache)
        if verbose:
            print(f"[dataset-v2] Wrote synthetic cache ({len(cache)} entries) to {cache_path}")

    return accepted, len(hard_prompts), recipe_ids


def build_dataset_v2(
    base_dataset_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    project_root: Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Build dataset v2 from v1 with config-driven interventions.

    Returns manifest dict (also written to dataset_v2_manifest.json).
    """
    def log(msg: str) -> None:
        if verbose:
            print(f"[dataset-v2] {msg}")

    interventions = config.get("interventions", {})
    target_failure_modes = config.get("target_failure_modes", [])
    seeds = config.get("seeds", {})
    shuffle_seed = seeds.get("shuffle_seed", 42)
    sampling_seed = seeds.get("sampling_seed", 42)

    add_hard_prompts = interventions.get("add_hard_prompts", False)
    hard_prompt_count = interventions.get("hard_prompt_count", 0)
    add_targeted_synthetic = interventions.get("add_targeted_synthetic", False)
    synthetic_count = interventions.get("synthetic_count", 0)
    oversample_v1 = interventions.get("oversample_v1_matches", False)
    oversample_factor = float(interventions.get("oversample_factor", 1.0))
    stricter_rejection = interventions.get("stricter_rejection", False)

    log(f"Input: {base_dataset_path}")

    # 1) Load v1
    if not base_dataset_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_dataset_path}")
    v1_records = load_jsonl(base_dataset_path)
    v1_source_hash = hash_file(base_dataset_path)
    log(f"Loaded {len(v1_records)} records from v1 base.")

    # 2) Stricter rejection
    if stricter_rejection and target_failure_modes:
        log("Applying stricter rejection (verification_steps / uncertainty_notes)...")
        v1_kept, v1_removed = apply_stricter_rejection(v1_records, target_failure_modes)
        num_v1_removed = len(v1_removed)
        removal_reasons: dict[str, int] = {}
        for r in v1_removed:
            reason = r.get("_rejection_reason", "unknown")
            removal_reasons[reason] = removal_reasons.get(reason, 0) + 1
        log(f"  Kept {len(v1_kept)}, removed {num_v1_removed} ({removal_reasons})")
    else:
        v1_kept = list(v1_records)
        v1_removed = []
        num_v1_removed = 0
        removal_reasons = {}

    num_v1_kept = len(v1_kept)
    for r in v1_kept:
        r.setdefault("source", "v1")
        r.setdefault("source_id", r.get("id", "unknown"))

    # 3) Hard prompts
    hard_prompts: list[dict[str, Any]] = []
    if add_hard_prompts and hard_prompt_count > 0 and target_failure_modes:
        log(f"Generating {hard_prompt_count} hard prompts for target failure modes...")
        hard_prompts = generate_hard_prompts(target_failure_modes, hard_prompt_count, sampling_seed)
    num_hard_prompts_generated = len(hard_prompts)
    if hard_prompts:
        log(f"  Generated {num_hard_prompts_generated} hard prompts.")

    # 4) Synthetic (optional; requires teacher config). Optional cache saves API calls.
    teacher_config_path = project_root / Path(config.get("teacher_config_path", "configs/teacher.yaml"))
    synthetic_cache_path: Path | None = None
    if config.get("synthetic_cache_path"):
        p = Path(config["synthetic_cache_path"])
        synthetic_cache_path = p if p.is_absolute() else project_root / p
    synthetic_records: list[dict[str, Any]] = []
    synthetic_recipe_ids: list[str] = []
    synthetic_attempted = 0
    if add_targeted_synthetic and synthetic_count > 0 and hard_prompts:
        to_generate = hard_prompts[: min(synthetic_count, len(hard_prompts))]
        log(f"Generating synthetic samples via teacher ({len(to_generate)} prompts)...")
        synthetic_records, synthetic_attempted, synthetic_recipe_ids = build_synthetic_records(
            to_generate, teacher_config_path, project_root, verbose=verbose, cache_path=synthetic_cache_path
        )
        log(f"  Synthetic: {len(synthetic_records)} accepted of {synthetic_attempted} attempted.")
    synthetic_acceptance_rate = (len(synthetic_records) / synthetic_attempted * 100) if synthetic_attempted > 0 else 0.0

    # 5) Oversample v1 matches
    oversampled: list[dict[str, Any]] = []
    if oversample_v1 and oversample_factor > 1.0 and target_failure_modes:
        log("Oversampling v1 records matching target failure mode tags...")
        oversampled = oversample_v1_matches(v1_kept, target_failure_modes, oversample_factor, sampling_seed)
    num_oversampled = len(oversampled)
    if oversampled:
        log(f"  Added {num_oversampled} oversampled copies.")

    # 6) Combine and shuffle
    combined: list[dict[str, Any]] = []
    combined.extend(ensure_traceability_v1(r) for r in v1_kept)
    combined.extend(oversampled)
    combined.extend(synthetic_records)
    rng = random.Random(shuffle_seed)
    rng.shuffle(combined)
    log(f"Combined: {num_v1_kept} v1 + {num_oversampled} oversampled + {len(synthetic_records)} synthetic = {len(combined)} total.")

    # 7) Write dataset_v2.jsonl
    log(f"Writing {output_dir / 'dataset_v2.jsonl'}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset_v2.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for r in combined:
            # Strip internal keys if any
            out_r = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(out_r, ensure_ascii=False) + "\n")

    # Stats
    by_source: dict[str, int] = {}
    for r in combined:
        src = r.get("source", "v1")
        by_source[src] = by_source.get(src, 0) + 1
    by_category: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    by_payload_type: dict[str, int] = {}
    by_region: dict[str, int] = {}
    by_target_failure_mode: dict[str, int] = {}
    for r in combined:
        by_category[r.get("category", "unknown")] = by_category.get(r.get("category", "unknown"), 0) + 1
        by_difficulty[r.get("difficulty", "unknown")] = by_difficulty.get(r.get("difficulty", "unknown"), 0) + 1
        by_payload_type[r.get("payload_type", "unknown")] = by_payload_type.get(r.get("payload_type", "unknown"), 0) + 1
        for tag in r.get("region_tags", []):
            by_region[tag] = by_region.get(tag, 0) + 1
        if r.get("source") == "synthetic_v2":
            mode = r.get("target_failure_mode", "unknown")
            by_target_failure_mode[mode] = by_target_failure_mode.get(mode, 0) + 1

    stats = {
        "total_records": len(combined),
        "by_source": by_source,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "by_payload_type": by_payload_type,
        "by_region": by_region,
        "by_target_failure_mode": by_target_failure_mode,
    }
    stats_path = output_dir / "dataset_v2_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Manifest
    manifest = {
        "v1_source_path": str(base_dataset_path),
        "v1_source_sha256": v1_source_hash,
        "build_config": config,
        "summary": {
            "num_v1_kept": num_v1_kept,
            "num_v1_removed": num_v1_removed,
            "removal_reasons": removal_reasons,
            "num_oversampled": num_oversampled,
            "num_hard_prompts_generated": num_hard_prompts_generated,
            "num_synthetic_generated": len(synthetic_records),
            "synthetic_acceptance_rate_pct": round(synthetic_acceptance_rate, 2),
            "synthetic_attempted": synthetic_attempted,
        },
        "intervention_breakdown_by_target_failure_mode": {
            mode: {
                "hard_prompts": sum(1 for h in hard_prompts if h.get("target_failure_mode") == mode),
                "synthetic_accepted": by_target_failure_mode.get(mode, 0),
            }
            for mode in target_failure_modes
        },
        "synthetic_recipe_ids": synthetic_recipe_ids,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output_dir / "dataset_v2_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if verbose:
        print()
        print("[dataset-v2] Summary:")
        print(f"  v1 base (input):     {len(v1_records)} records")
        print(f"  stricter rejection: kept {num_v1_kept}, removed {num_v1_removed} ({removal_reasons})")
        print(f"  oversampled:         +{num_oversampled}")
        print(f"  synthetic:          +{len(synthetic_records)} (of {synthetic_attempted} attempted)")
        print(f"  total v2 output:    {len(combined)} records")
        print()

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset v2 from v1 with targeted interventions")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_v2_build.yaml"), help="Path to build config YAML")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    config_path = args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dataset_path = Path(config.get("base_dataset_path", "data/processed/dataset_v1_clean.jsonl"))
    output_dir = Path(config.get("output_dir", "data/processed"))
    project_root = Path(__file__).resolve().parents[3]
    if not base_dataset_path.is_absolute():
        base_dataset_path = project_root / base_dataset_path
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    build_dataset_v2(base_dataset_path, output_dir, config, project_root, verbose=not args.quiet)
    if not args.quiet:
        print(f"[INFO] Wrote {output_dir / 'dataset_v2.jsonl'}, manifest, stats.")


if __name__ == "__main__":
    main()
