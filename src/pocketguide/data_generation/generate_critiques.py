"""Generate critiques for draft responses (Milestone 3, Lesson 3.4).

This module implements a quality gate pass that evaluates draft responses
for hallucination risk, schema compliance, actionability, and safety.
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import jsonschema
import yaml

from pocketguide.teachers.base import TeacherRequest
from pocketguide.teachers.openrouter import OpenRouterTeacherClient
from pocketguide.teachers.router import TeacherRouterClient

logger = logging.getLogger(__name__)

# Maximum raw critique text to store (prevent huge records)
MAX_CRITIQUE_TEXT_LENGTH = 20000


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of file for reproducibility."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_critique_schema(schema_path: Path = None) -> dict:
    """Load critique JSON schema.
    
    Args:
        schema_path: Path to schema file (defaults to package schema)
        
    Returns:
        Schema dict
    """
    if schema_path is None:
        # Default to package schema - use relative path from package root
        import pocketguide
        pkg_root = Path(pocketguide.__file__).parent
        schema_path = pkg_root / "data/schemas/v1/critique.schema.json"
    
    with open(schema_path) as f:
        return json.load(f)


def load_drafts(path: Path) -> list[dict]:
    """Load drafts JSONL file.
    
    Args:
        path: Path to drafts_v1.jsonl
        
    Returns:
        List of draft records
    """
    drafts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                draft = json.loads(line)
                drafts.append(draft)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
                continue
    return drafts


def load_teacher_config(config_path: Path = None) -> dict:
    """Load teacher configuration from YAML.
    
    Args:
        config_path: Path to teacher.yaml (defaults to configs/teacher.yaml)
        
    Returns:
        Configuration dict
    """
    if config_path is None:
        config_path = Path("configs/teacher.yaml")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_teacher_router(config: dict, override_dry_run: bool = False) -> TeacherRouterClient:
    """Create TeacherRouterClient from config.
    
    Args:
        config: Teacher configuration dict
        override_dry_run: Force dry_run mode
        
    Returns:
        TeacherRouterClient instance
    """
    models = config.get("models", [])
    runtime = config.get("runtime", {})
    rate_limit = config.get("rate_limit", {})
    
    dry_run = override_dry_run or runtime.get("dry_run", False)
    
    backend = OpenRouterTeacherClient(
        rpm=rate_limit.get("rpm", 15),
        dry_run=dry_run,
        timeout_s=runtime.get("timeout_s", 60),
    )
    
    router = TeacherRouterClient(
        backend=backend,
        models=models,
        fallback_to_paid=runtime.get("fallback_to_paid", True),
    )
    
    return router


def load_critique_prompt_template(template_path: Path = None) -> str:
    """Load critique prompt template.
    
    Args:
        template_path: Path to template file (defaults to data/prompts/teacher/v1/critique.txt)
        
    Returns:
        Template string
    """
    if template_path is None:
        template_path = Path("data/prompts/teacher/v1/critique.txt")
    
    with open(template_path) as f:
        return f.read()


def build_critique_prompt(draft: dict, template: str) -> str:
    """Build critique prompt from draft and template.
    
    Args:
        draft: Draft record with prompt, output_text, contract
        template: Prompt template string
        
    Returns:
        Formatted prompt
    """
    prompt_text = draft.get("prompt", "")
    output_text = draft.get("output_text", "")
    contract = draft.get("contract", {})
    draft_id = draft.get("id", "unknown")
    
    # Format contract as readable JSON
    contract_json = json.dumps(contract, indent=2)
    
    return template.format(
        prompt=prompt_text,
        output_text=output_text,
        contract_json=contract_json,
        draft_id=draft_id,
    )


def build_critique_request(critique_prompt: str, config: dict) -> TeacherRequest:
    """Build TeacherRequest for critique generation.
    
    Args:
        critique_prompt: Formatted critique prompt
        config: Teacher configuration
        
    Returns:
        TeacherRequest instance
    """
    gen_config = config.get("generation", {})
    
    return TeacherRequest(
        messages=[
            {
                "role": "system",
                "content": "You are an expert travel assistant quality reviewer. Output only valid JSON.",
            },
            {
                "role": "user",
                "content": critique_prompt,
            },
        ],
        temperature=gen_config.get("temperature", 0.2),
        top_p=gen_config.get("top_p", 0.9),
        max_tokens=gen_config.get("max_tokens", 1500),
        seed=config.get("seed", 42),
    )


def parse_critique_json(text: str) -> tuple[bool, dict | None, str | None]:
    """Parse critique response as JSON.
    
    Tries strict parsing first, then lenient extraction from markdown/prose.
    
    Args:
        text: Raw critique response text
        
    Returns:
        Tuple of (success, parsed_data, error_message)
    """
    # Try strict JSON parse
    try:
        data = json.loads(text.strip())
        return True, data, None
    except json.JSONDecodeError:
        pass
    
    # Try lenient extraction (markdown code fence)
    import re
    
    # Look for ```json or ``` blocks
    patterns = [
        r"```json\s*\n(.*?)\n```",
        r"```\s*\n(.*?)\n```",
        r"\{.*\}",  # Last resort: find JSON object
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_text = match.group(1) if match.lastindex else match.group(0)
            try:
                data = json.loads(json_text)
                return True, data, None
            except json.JSONDecodeError:
                continue
    
    return False, None, "Could not extract valid JSON from response"


def validate_critique_schema(data: dict, schema: dict) -> tuple[bool, str | None]:
    """Validate critique JSON against schema.
    
    Args:
        data: Parsed critique JSON
        schema: JSON schema dict
        
    Returns:
        Tuple of (valid, error_message)
    """
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)


def build_critique_record(
    draft: dict,
    teacher_response,
    critique_data: dict | None,
    critique_contract: dict,
    raw_critique_text: str,
) -> dict:
    """Build critique record for output.
    
    Args:
        draft: Original draft record
        teacher_response: TeacherResponse from critique generation
        critique_data: Validated critique JSON (None if validation failed)
        critique_contract: Validation results (strict_json_ok, schema_ok, error)
        raw_critique_text: Raw response text (truncated)
        
    Returns:
        Critique record dict
    """
    # Truncate raw text if too long
    if len(raw_critique_text) > MAX_CRITIQUE_TEXT_LENGTH:
        raw_critique_text = raw_critique_text[:MAX_CRITIQUE_TEXT_LENGTH] + "...[truncated]"
    
    record = {
        "id": draft.get("id"),
        "draft_id": draft.get("id"),  # Explicit reference
        "critique": critique_data,
        "raw_critique_text": raw_critique_text if not critique_data else None,
        "teacher": {
            "provider": teacher_response.provider,
            "selected_model": teacher_response.raw.get("selected_model"),
            "attempted_models": teacher_response.raw.get("attempted_models", []),
            "request_id": teacher_response.request_id,
            "timing": teacher_response.timing,
            "usage": teacher_response.usage,
        },
        "critique_contract": critique_contract,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    
    return record


def generate_critiques(
    drafts_path: Path,
    out_dir: Path,
    teacher: TeacherRouterClient,
    config: dict,
    schema: dict,
    template: str,
    limit: int = None,
    resume: bool = False,
) -> dict:
    """Generate critiques for draft responses.
    
    Args:
        drafts_path: Path to drafts_v1.jsonl
        out_dir: Output directory
        teacher: TeacherRouterClient instance
        config: Teacher configuration
        schema: Critique JSON schema
        template: Critique prompt template
        limit: Max number of drafts to process (None = all)
        resume: Skip already-critiqued IDs
        
    Returns:
        Statistics dict
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    critiques_path = out_dir / "critiques_v1.jsonl"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    # Load existing IDs if resuming
    existing_ids = set()
    if resume and critiques_path.exists():
        logger.info(f"Resume mode: checking existing critiques in {critiques_path}")
        with open(critiques_path) as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        existing_ids.add(record["draft_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        logger.info(f"Found {len(existing_ids)} existing critiques, will skip those")
    
    # Load drafts
    logger.info(f"Loading drafts from {drafts_path}")
    drafts = load_drafts(drafts_path)
    
    if limit:
        drafts = drafts[:limit]
    
    logger.info(f"Processing {len(drafts)} drafts")
    
    # Initialize stats
    stats = {
        "total": len(drafts),
        "processed": 0,
        "skipped": 0,
        "critique_parse_ok": 0,
        "critique_schema_ok": 0,
        "verdicts": {"pass": 0, "revise": 0, "reject": 0},
        "issue_types": {},
        "risk_levels": {"low": 0, "medium": 0, "high": 0},
        "score_sums": {
            "actionability": 0,
            "clarity": 0,
            "schema_compliance": 0,
            "safety_risk": 0,
        },
    }
    
    # Process drafts
    logger.info("Starting critique generation...")
    with open(critiques_path, "a") as out_f:
        for idx, draft in enumerate(drafts, 1):
            stats["total"] = idx  # Update as we go
            draft_id = draft.get("id", f"unknown_{idx}")
            
            # Skip if resuming and already processed
            if draft_id in existing_ids:
                stats["skipped"] += 1
                continue
            
            stats["processed"] += 1
            
            if idx % max(1, len(drafts) // 10) == 0:
                logger.info(f"  Processing {idx}/{len(drafts)}...")
            
            try:
                # Build critique prompt
                critique_prompt = build_critique_prompt(draft, template)
                
                # Build request
                request = build_critique_request(critique_prompt, config)
                
                # Call teacher
                teacher_response = teacher.generate(request)
                
                # Parse critique JSON
                parse_ok, critique_data, parse_error = parse_critique_json(teacher_response.text)
                
                # Validate schema if parse succeeded
                schema_ok = False
                schema_error = None
                if parse_ok and critique_data:
                    schema_ok, schema_error = validate_critique_schema(critique_data, schema)
                    stats["critique_parse_ok"] += 1
                    
                    if schema_ok:
                        stats["critique_schema_ok"] += 1
                        
                        # Update stats from critique
                        verdict = critique_data.get("verdict", "unknown")
                        if verdict in stats["verdicts"]:
                            stats["verdicts"][verdict] += 1
                        
                        # Issue types
                        for issue in critique_data.get("issues", []):
                            issue_type = issue.get("type", "unknown")
                            stats["issue_types"][issue_type] = stats["issue_types"].get(issue_type, 0) + 1
                        
                        # Risk levels
                        risk_level = critique_data.get("hallucination", {}).get("risk_level")
                        if risk_level in stats["risk_levels"]:
                            stats["risk_levels"][risk_level] += 1
                        
                        # Scores
                        scores = critique_data.get("scores", {})
                        for key in stats["score_sums"]:
                            if key in scores:
                                stats["score_sums"][key] += scores[key]
                
                # Build critique contract
                critique_contract = {
                    "strict_json_ok": parse_ok,
                    "schema_ok": schema_ok,
                }
                
                if not parse_ok:
                    critique_contract["error"] = {
                        "code": "JSON_PARSE_ERROR",
                        "message": parse_error or "Failed to parse JSON",
                    }
                elif not schema_ok:
                    critique_contract["error"] = {
                        "code": "SCHEMA_VALIDATION_ERROR",
                        "message": schema_error or "Schema validation failed",
                    }
                
                # Build critique record (store critique only if valid)
                critique_record = build_critique_record(
                    draft=draft,
                    teacher_response=teacher_response,
                    critique_data=critique_data if schema_ok else None,
                    critique_contract=critique_contract,
                    raw_critique_text=teacher_response.text,
                )
                
                # Write to output
                out_f.write(json.dumps(critique_record) + "\n")
                
            except Exception as e:
                logger.error(f"Error processing draft {draft_id}: {e}")
                
                # Write failure record
                failure_record = {
                    "id": draft_id,
                    "draft_id": draft_id,
                    "critique": None,
                    "raw_critique_text": None,
                    "teacher": None,
                    "critique_contract": {
                        "strict_json_ok": False,
                        "schema_ok": False,
                        "error": {
                            "code": "GENERATION_ERROR",
                            "message": str(e),
                        },
                    },
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                out_f.write(json.dumps(failure_record) + "\n")
    
    logger.info(f"Critique generation complete. Output: {critiques_path}")
    
    # Write manifest and stats
    write_manifest(out_dir, drafts_path, config, schema, stats, run_id)
    write_stats(out_dir, stats)
    
    return stats


def redact_config(config: dict) -> dict:
    """Redact sensitive information from config."""
    import copy
    
    redacted = copy.deepcopy(config)
    
    # Remove any keys that might contain secrets
    for key in ["api_key", "secret", "token", "password"]:
        if key in redacted:
            redacted[key] = "***REDACTED***"
    
    return redacted


def write_manifest(
    out_dir: Path,
    drafts_path: Path,
    config: dict,
    schema: dict,
    stats: dict,
    run_id: str,
) -> Path:
    """Write critique generation manifest.
    
    Args:
        out_dir: Output directory
        drafts_path: Input drafts path
        config: Teacher configuration
        schema: Critique schema
        stats: Generation statistics
        run_id: Run identifier
        
    Returns:
        Path to manifest file
    """
    manifest_path = out_dir / "critiques_v1_manifest.json"
    
    # Get schema path and hash
    schema_path = Path("src/pocketguide/data/schemas/v1/critique.schema.json")
    schema_hash = hash_file(schema_path) if schema_path.exists() else "unknown"
    
    manifest = {
        "version": "v1",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "drafts_path": str(drafts_path),
            "drafts_hash": hash_file(drafts_path),
        },
        "critique_schema": {
            "path": str(schema_path),
            "hash": schema_hash,
            "version": "v1",
        },
        "teacher_config": redact_config(config),
        "counts": {
            "total_drafts": stats["total"],
            "processed": stats["processed"],
            "skipped": stats["skipped"],
            "critique_parse_ok": stats["critique_parse_ok"],
            "critique_schema_ok": stats["critique_schema_ok"],
        },
        "outputs": {
            "critiques": "critiques_v1.jsonl",
            "manifest": "critiques_v1_manifest.json",
            "stats": "critiques_v1_stats.json",
        },
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Manifest written to {manifest_path}")
    return manifest_path


def write_stats(out_dir: Path, stats: dict) -> Path:
    """Write critique generation statistics.
    
    Args:
        out_dir: Output directory
        stats: Statistics dict
        
    Returns:
        Path to stats file
    """
    stats_path = out_dir / "critiques_v1_stats.json"
    
    # Compute averages
    processed_valid = stats["critique_schema_ok"]
    avg_scores = {}
    if processed_valid > 0:
        for key, total in stats["score_sums"].items():
            avg_scores[key] = round(total / processed_valid, 2)
    else:
        avg_scores = {key: 0 for key in stats["score_sums"]}
    
    # Compute rates
    total_processed = stats["processed"]
    critique_parse_rate = (
        round(100 * stats["critique_parse_ok"] / total_processed, 2)
        if total_processed > 0
        else 0
    )
    critique_schema_valid_rate = (
        round(100 * stats["critique_schema_ok"] / total_processed, 2)
        if total_processed > 0
        else 0
    )
    
    output_stats = {
        "total": stats["total"],
        "processed": stats["processed"],
        "skipped": stats["skipped"],
        "critique_parse_rate": critique_parse_rate,
        "critique_schema_valid_rate": critique_schema_valid_rate,
        "verdict_distribution": stats["verdicts"],
        "issue_type_counts": dict(
            sorted(stats["issue_types"].items(), key=lambda x: x[1], reverse=True)
        ),
        "risk_level_distribution": stats["risk_levels"],
        "avg_scores": avg_scores,
    }
    
    with open(stats_path, "w") as f:
        json.dump(output_stats, f, indent=2)
    
    logger.info(f"Stats written to {stats_path}")
    return stats_path


def main(args: list[str] = None) -> dict:
    """Main CLI entry point.
    
    Args:
        args: Command-line arguments (None = sys.argv)
        
    Returns:
        Statistics dict
    """
    parser = argparse.ArgumentParser(
        description="Generate critiques for draft responses (Lesson 3.4)"
    )
    parser.add_argument(
        "--drafts",
        type=Path,
        default=Path("data/interim/drafts_v1.jsonl"),
        help="Path to drafts JSONL file",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/interim"),
        help="Output directory for critiques",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to teacher config YAML (default: configs/teacher.yaml)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="Path to critique schema JSON (default: package schema)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Path to critique prompt template (default: data/prompts/teacher/v1/critique.txt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N drafts (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-critiqued IDs",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Override config to force dry-run mode (no API calls)",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run identifier (defaults to timestamp)",
    )
    
    parsed_args = parser.parse_args(args)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    try:
        # Load dependencies
        logger.info("Loading configuration and schema...")
        config = load_teacher_config(parsed_args.config)
        schema = load_critique_schema(parsed_args.schema)
        template = load_critique_prompt_template(parsed_args.template)
        
        # Create teacher router
        teacher = create_teacher_router(config, override_dry_run=parsed_args.dry_run)
        
        # Generate critiques
        stats = generate_critiques(
            drafts_path=parsed_args.drafts,
            out_dir=parsed_args.out_dir,
            teacher=teacher,
            config=config,
            schema=schema,
            template=template,
            limit=parsed_args.limit,
            resume=parsed_args.resume,
        )
        
        logger.info("Critique generation successful")
        logger.info(f"  Processed: {stats['processed']}")
        logger.info(f"  Parse rate: {stats['critique_parse_ok']}/{stats['processed']}")
        logger.info(f"  Schema valid rate: {stats['critique_schema_ok']}/{stats['processed']}")
        logger.info(f"  Verdicts: {stats['verdicts']}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Critique generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
