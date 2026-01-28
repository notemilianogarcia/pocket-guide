"""Generate draft responses from teacher model with schema-first validation.

This module implements the draft generation pass (Lesson 3.3):
1. Load prompt plan (deterministic, from Lesson 3.1)
2. Call teacher model ONCE per prompt (OpenRouter with fallback, from Lesson 3.2)
3. Parse/validate response (strict JSON, lenient fallback, contract validation)
4. Write drafts with full provenance and validation results

Usage:
    python -m pocketguide.data_generation.generate_drafts \\
        --plan data/interim/prompt_plan_v1.jsonl \\
        --out_dir data/interim
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from pocketguide.eval.parsing import parse_and_validate
from pocketguide.teachers.base import TeacherRequest
from pocketguide.teachers.openrouter import OpenRouterTeacherClient
from pocketguide.teachers.router import TeacherRouterClient

logger = logging.getLogger(__name__)


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of file for reproducibility tracking."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_teacher_config(config_path: Path = None) -> dict:
    """Load teacher configuration from YAML.

    Args:
        config_path: Path to teacher.yaml (default: configs/teacher.yaml)

    Returns:
        Configuration dict
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "configs" / "teacher.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Teacher config not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def create_teacher_router(config: dict, override_dry_run: bool = None) -> TeacherRouterClient:
    """Create TeacherRouterClient from config.

    Args:
        config: Teacher config dict
        override_dry_run: If set, override config dry_run value

    Returns:
        Configured TeacherRouterClient
    """
    runtime_cfg = config.get("runtime", {})
    rate_limit_cfg = config.get("rate_limit", {})

    # Determine dry_run mode
    dry_run = runtime_cfg.get("dry_run", True)
    if override_dry_run is not None:
        dry_run = override_dry_run

    # Create OpenRouter backend
    backend = OpenRouterTeacherClient(
        model=config["models"][0],  # Will be overridden by router
        dry_run=dry_run,
        timeout_s=runtime_cfg.get("timeout_s", 60.0),
        rpm=rate_limit_cfg.get("rpm", 15),
        max_retries=rate_limit_cfg.get("max_retries_per_model", 6),
        backoff_base_s=rate_limit_cfg.get("backoff_base_s", 1.0),
        backoff_max_s=rate_limit_cfg.get("backoff_max_s", 30.0),
    )

    # Create router with fallback chain
    return TeacherRouterClient(
        backend=backend,
        models=config["models"],
        fallback_to_paid=runtime_cfg.get("fallback_to_paid", True),
        max_retries_per_model=rate_limit_cfg.get("max_retries_per_model", 6),
    )


def redact_config(config: dict) -> dict:
    """Create redacted copy of config (remove secrets)."""
    import copy

    redacted = copy.deepcopy(config)
    # Remove any potential secrets
    if "api_key" in redacted:
        redacted["api_key"] = "***REDACTED***"
    return redacted


def load_prompt_plan(plan_path: Path) -> list[dict]:
    """Load prompt plan from JSONL file.

    Args:
        plan_path: Path to prompt_plan_v1.jsonl

    Returns:
        List of prompt plan records
    """
    if not plan_path.exists():
        raise FileNotFoundError(f"Prompt plan not found: {plan_path}")

    records = []
    with open(plan_path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                # Validate required fields
                required = ["id", "payload_type", "prompt", "template_version"]
                for field in required:
                    if field not in record:
                        raise ValueError(f"Missing required field '{field}' in record {i}")
                records.append(record)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in prompt plan line {i}: {e}")

    return records


def build_teacher_request(prompt_record: dict, generation_config: dict) -> TeacherRequest:
    """Build TeacherRequest from prompt plan record.

    Args:
        prompt_record: Record from prompt plan
        generation_config: Generation parameters from teacher config

    Returns:
        TeacherRequest object
    """
    system_message = (
        "You are a helpful travel assistant. Provide structured, well-reasoned guidance. "
        "Always format your response as valid JSON with clear sections."
    )

    return TeacherRequest(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_record["prompt"]},
        ],
        temperature=generation_config.get("temperature", 0.2),
        top_p=generation_config.get("top_p", 0.9),
        max_tokens=generation_config.get("max_tokens", 900),
        seed=generation_config.get("seed", 42),
        metadata={
            "prompt_id": prompt_record["id"],
            "payload_type": prompt_record["payload_type"],
            "template_version": prompt_record.get("template_version"),
        },
    )


def build_draft_record_from_parse(
    prompt_record: dict,
    teacher_response: Any,
    parse_result: Any,
) -> dict:
    """Build draft record from parsed result.

    Args:
        prompt_record: Original prompt plan record
        teacher_response: Response from teacher model
        parse_result: Result from parse_and_validate (ParseResult object)

    Returns:
        Complete draft record
    """
    # Build contract dict from ParseResult
    contract = {
        "overall_ok": parse_result.success,
        "payload_type": parse_result.data.get("payload_type") if parse_result.data else None,
    }

    if parse_result.error:
        contract["error"] = {
            "code": parse_result.error.code,
            "error_type": parse_result.error.error_type,
            "failed_at": parse_result.error.failed_at,
            "message": parse_result.error.message,
        }

    return {
        "id": prompt_record["id"],
        "prompt_plan": {
            "payload_type": prompt_record.get("payload_type"),
            "category": prompt_record.get("category"),
            "difficulty": prompt_record.get("difficulty"),
            "region_tags": prompt_record.get("region_tags", []),
            "template_version": prompt_record.get("template_version"),
            "template_name": prompt_record.get("template_name"),
            "seed": prompt_record.get("seed"),
        },
        "prompt": prompt_record.get("prompt"),
        "teacher": {
            "provider": teacher_response.provider,
            "selected_model": teacher_response.raw.get("selected_model"),
            "attempted_models": teacher_response.raw.get("attempted_models", []),
            "request_id": teacher_response.request_id,
            "timing": teacher_response.timing,
            "usage": teacher_response.usage,
        },
        "output_text": teacher_response.text,
        "contract": contract,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_drafts(
    plan_path: Path,
    out_dir: Path,
    teacher: TeacherRouterClient,
    config: dict,
    limit: int = None,
    resume: bool = False,
) -> dict:
    """Generate drafts from prompt plan.

    Args:
        plan_path: Path to prompt_plan_v1.jsonl
        out_dir: Output directory
        teacher: TeacherRouterClient instance
        config: Teacher configuration
        limit: Max number of records to process (None = all)
        resume: Skip already-generated IDs

    Returns:
        Statistics dict
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    drafts_path = out_dir / "drafts_v1.jsonl"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Load existing IDs if resuming
    existing_ids = set()
    if resume and drafts_path.exists():
        logger.info(f"Resume mode: checking existing drafts in {drafts_path}")
        with open(drafts_path) as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        existing_ids.add(record["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        logger.info(f"Found {len(existing_ids)} existing records, will skip those")

    # Load prompt plan
    logger.info(f"Loading prompt plan from {plan_path}")
    prompts = load_prompt_plan(plan_path)
    logger.info(f"Loaded {len(prompts)} prompts")

    if limit:
        prompts = prompts[:limit]
        logger.info(f"Limited to {limit} prompts")

    # Statistics
    stats = {
        "total": 0,
        "processed": 0,
        "skipped": 0,
        "strict_json_pass": 0,
        "lenient_json_pass": 0,
        "envelope_pass": 0,
        "payload_pass": 0,
        "overall_pass": 0,
        "by_payload_type": {},
        "by_category": {},
        "by_difficulty": {},
        "by_error_code": {},
    }

    generation_cfg = config.get("generation", {})

    # Process prompts
    logger.info("Starting draft generation...")
    with open(drafts_path, "a") as out_f:
        for i, prompt_record in enumerate(prompts):
            stats["total"] += 1
            prompt_id = prompt_record["id"]

            # Skip if resuming and already processed
            if prompt_id in existing_ids:
                stats["skipped"] += 1
                continue

            stats["processed"] += 1

            try:
                # Build request
                teacher_request = build_teacher_request(prompt_record, generation_cfg)

                # Call teacher model once
                teacher_response = teacher.generate(teacher_request)

                # Parse and validate response
                parse_result = parse_and_validate(
                    teacher_response.text,
                    strict_json=True,
                )

                # Build draft record
                draft = build_draft_record_from_parse(prompt_record, teacher_response, parse_result)

                # Write to output
                out_f.write(json.dumps(draft) + "\n")

                # Update stats
                if draft["contract"].get("overall_ok"):
                    stats["overall_pass"] += 1

                # Track by type/category/difficulty
                payload_type = prompt_record.get("payload_type", "unknown")
                if payload_type not in stats["by_payload_type"]:
                    stats["by_payload_type"][payload_type] = 0
                stats["by_payload_type"][payload_type] += 1

                category = prompt_record.get("category", "unknown")
                if category not in stats["by_category"]:
                    stats["by_category"][category] = 0
                stats["by_category"][category] += 1

                difficulty = prompt_record.get("difficulty", "unknown")
                if difficulty not in stats["by_difficulty"]:
                    stats["by_difficulty"][difficulty] = 0
                stats["by_difficulty"][difficulty] += 1

                # Track errors
                if "error" in draft["contract"]:
                    error_code = draft["contract"]["error"].get("code", "unknown")
                    if error_code not in stats["by_error_code"]:
                        stats["by_error_code"][error_code] = 0
                    stats["by_error_code"][error_code] += 1

                if (stats["processed"] % 10) == 0:
                    logger.info(
                        f"Processed {stats['processed']}/{len(prompts)} "
                        f"(overall_ok: {stats['overall_pass']}/{stats['processed']})"
                    )

            except Exception as e:
                # Record failure but continue
                logger.warning(f"Error processing prompt {prompt_id}: {e}")

                # Create failure record
                failure_record = {
                    "id": prompt_id,
                    "prompt_plan": prompt_record,
                    "prompt": prompt_record.get("prompt"),
                    "teacher": None,
                    "output_text": None,
                    "contract": {
                        "strict_json_ok": False,
                        "lenient_json_ok": False,
                        "envelope_ok": False,
                        "payload_ok": False,
                        "overall_ok": False,
                        "error": {
                            "code": "GENERATION_ERROR",
                            "error_type": type(e).__name__,
                            "message": str(e),
                        },
                    },
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }

                out_f.write(json.dumps(failure_record) + "\n")

                # Update error stats
                stats["by_error_code"]["GENERATION_ERROR"] = (
                    stats["by_error_code"].get("GENERATION_ERROR", 0) + 1
                )

    logger.info(f"Draft generation complete. Output: {drafts_path}")
    
    # Write manifest and stats files
    write_manifest(out_dir, plan_path, config, stats, run_id)
    write_stats(out_dir, stats)
    
    return stats


def compute_pass_rates(stats: dict) -> dict:
    """Compute pass rate percentages."""
    total = stats["processed"]
    if total == 0:
        return {
            "strict_json_rate": 0,
            "lenient_json_rate": 0,
            "envelope_rate": 0,
            "payload_rate": 0,
            "overall_rate": 0,
        }

    return {
        "strict_json_rate": round(100 * stats["strict_json_pass"] / total, 2),
        "lenient_json_rate": round(100 * stats["lenient_json_pass"] / total, 2),
        "envelope_rate": round(100 * stats["envelope_pass"] / total, 2),
        "payload_rate": round(100 * stats["payload_pass"] / total, 2),
        "overall_rate": round(100 * stats["overall_pass"] / total, 2),
    }


def write_manifest(
    out_dir: Path,
    plan_path: Path,
    config: dict,
    stats: dict,
    run_id: str,
) -> Path:
    """Write generation manifest."""
    manifest_path = out_dir / "drafts_v1_manifest.json"

    manifest = {
        "version": "v1",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec": {
            "plan_path": str(plan_path),
            "plan_hash": hash_file(plan_path),
        },
        "teacher_config": redact_config(config),
        "schema_versions": {
            "envelope": "v0",
            "payload": "v1",
        },
        "counts": {
            "total_planned": stats["total"],
            "processed": stats["processed"],
            "skipped": stats["skipped"],
        },
        "outputs": {
            "drafts": "drafts_v1.jsonl",
            "manifest": "drafts_v1_manifest.json",
            "stats": "drafts_v1_stats.json",
        },
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest written to {manifest_path}")
    return manifest_path


def write_stats(out_dir: Path, stats: dict) -> Path:
    """Write generation statistics."""
    stats_path = out_dir / "drafts_v1_stats.json"

    pass_rates = compute_pass_rates(stats)

    output_stats = {
        "total": stats["total"],
        "processed": stats["processed"],
        "skipped": stats["skipped"],
        "pass_rates": pass_rates,
        "by_payload_type": stats["by_payload_type"],
        "by_category": stats["by_category"],
        "by_difficulty": stats["by_difficulty"],
        "top_errors": dict(
            sorted(stats["by_error_code"].items(), key=lambda x: x[1], reverse=True)[:10]
        ),
    }

    with open(stats_path, "w") as f:
        json.dump(output_stats, f, indent=2)

    logger.info(f"Stats written to {stats_path}")
    return stats_path


def main(args=None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate draft responses from teacher model with schema-first validation"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/interim/prompt_plan_v1.jsonl"),
        help="Path to prompt plan JSONL",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/interim"),
        help="Output directory for drafts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N prompts (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-generated IDs",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Override config to force dry-run mode (no API calls)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to teacher config YAML",
    )

    parsed_args = parser.parse_args(args)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Load config
        teacher_config = load_teacher_config(parsed_args.config)

        # Create teacher router
        teacher = create_teacher_router(teacher_config, override_dry_run=parsed_args.dry_run)

        # Generate drafts
        stats = generate_drafts(
            plan_path=parsed_args.plan,
            out_dir=parsed_args.out_dir,
            teacher=teacher,
            config=teacher_config,
            limit=parsed_args.limit,
            resume=parsed_args.resume,
        )

        logger.info("Draft generation successful")

        # Print summary
        print("\n" + "=" * 60)
        print("DRAFT GENERATION SUMMARY")
        print("=" * 60)
        print(f"Total planned: {stats['total']}")
        print(f"Processed: {stats['processed']}")
        print(f"Skipped: {stats['skipped']}")
        pass_rates = compute_pass_rates(stats)
        print(f"\nPass rates (processed samples):")
        print(f"  Strict JSON: {pass_rates['strict_json_rate']}%")
        print(f"  Lenient JSON: {pass_rates['lenient_json_rate']}%")
        print(f"  Envelope: {pass_rates['envelope_rate']}%")
        print(f"  Payload: {pass_rates['payload_rate']}%")
        print(f"  Overall: {pass_rates['overall_rate']}%")
        print(f"\nOutputs written to {parsed_args.out_dir}")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
