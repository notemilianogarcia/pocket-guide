"""
Generate final training dataset (v1) from prompt plans, drafts, and critiques.

This module joins prompt plans, draft outputs, and critique feedback to produce
a refined, quality-gated training dataset. The pipeline:

1. Loads prompt_plan + drafts + critiques by ID
2. For items with verdict != "reject", calls teacher to refine draft
3. Validates refined output (strict or lenient based on --gating_mode)
4. Applies quality gates (contract, verification, overconfidence)
5. Writes accepted samples to dataset_v1.jsonl
6. Writes rejected samples to dataset_v1_rejected.jsonl
7. Writes manifest and stats for reproducibility

Key features:
- Resume support: skips already-processed IDs
- Dry run: simulates without API calls
- Quality gates: deterministic, explainable rejection criteria
- Failure tolerance: records errors, continues processing
- Provenance tracking: captures teacher metadata, gate results, timestamps
"""

import argparse
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

import yaml

# Import teacher clients
from pocketguide.teachers.router import TeacherRouterClient
from pocketguide.teachers.openrouter import OpenRouterTeacherClient

# Import parser/validator
from pocketguide.eval.parsing import parse_and_validate

# Import quality gates
from pocketguide.data_generation.quality_gates import (
    apply_all_gates,
    compute_overall_quality,
    get_rejection_reasons,
)


class JoinedSample(NamedTuple):
    """A sample with prompt plan, draft, and critique joined."""
    
    id: str
    prompt_plan: Dict[str, Any]
    draft: Dict[str, Any]
    critique: Dict[str, Any] | None


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file for reproducibility tracking."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_mock_payload(payload_type: str) -> Dict[str, Any]:
    """Create a mock payload for dry-run mode."""
    if payload_type == "checklist":
        return {
            "title": "Travel Checklist",
            "groups": [{
                "name": "Preparation",
                "items": [{
                    "text": "Check visa requirements",
                    "priority": "must"
                }]
            }]
        }
    elif payload_type == "itinerary":
        return {
            "days": [{
                "day_number": 1,
                "activities": [{
                    "time_block": "morning",
                    "description": "Arrive at destination",
                    "location": "Airport"
                }]
            }]
        }
    elif payload_type == "decision_tree":
        return {
            "root_node": {
                "type": "decision",
                "question": "Do you have a visa?",
                "yes": {"type": "leaf", "outcome": "Proceed to travel"},
                "no": {"type": "leaf", "outcome": "Apply for visa"}
            }
        }
    elif payload_type == "procedure":
        return {
            "title": "Visa Application",
            "steps": [{
                "step_number": 1,
                "action": "Gather documents",
                "details": "Collect passport and application form"
            }]
        }
    else:
        # Fallback: checklist
        return create_mock_payload("checklist")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSON line in {path}: {e}", file=sys.stderr)
                continue
    return records


def load_prompt_template(template_name: str) -> str:
    """Load refinement prompt template from data directory."""
    template_path = Path("data") / "prompts" / "teacher" / "v1" / f"{template_name}.txt"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def join_inputs(
    prompt_plans: List[Dict[str, Any]],
    drafts: List[Dict[str, Any]],
    critiques: List[Dict[str, Any]]
) -> tuple[List[JoinedSample], List[Dict[str, Any]]]:
    """
    Join prompt plans, drafts, and critiques by ID.
    
    Returns:
        Tuple of (joined_samples, missing_input_rejections)
    """
    # Build indexes
    plan_index = {p["id"]: p for p in prompt_plans}
    draft_index = {d["id"]: d for d in drafts}
    critique_index = {c["draft_id"]: c for c in critiques}
    
    # All unique IDs from all sources
    all_ids = set(plan_index.keys()) | set(draft_index.keys()) | set(critique_index.keys())
    
    joined = []
    missing_rejections = []
    
    for sample_id in sorted(all_ids):
        plan = plan_index.get(sample_id)
        draft = draft_index.get(sample_id)
        critique = critique_index.get(sample_id)
        
        # Check if any component is missing
        missing_components = []
        if not plan:
            missing_components.append("prompt_plan")
        if not draft:
            missing_components.append("draft")
        # Note: critique is optional (we'll handle verdict="reject" separately)
        
        if missing_components:
            missing_rejections.append({
                "id": sample_id,
                "reason": "missing_inputs",
                "missing_components": missing_components,
                "created_at": datetime.now(timezone.utc).isoformat()
            })
            continue
        
        joined.append(JoinedSample(
            id=sample_id,
            prompt_plan=plan,
            draft=draft,
            critique=critique
        ))
    
    return joined, missing_rejections


def build_refinement_prompt(
    prompt_text: str,
    draft_output: str,
    critique_json: Dict[str, Any],
    rewrite_instructions: List[str],
    payload_type: str,
    template: str
) -> str:
    """
    Build refinement prompt from template.
    
    Args:
        prompt_text: Original user prompt
        draft_output: Draft response text
        critique_json: Full critique object
        rewrite_instructions: List of instructions from critique
        payload_type: Expected payload type
        template: Prompt template string
    
    Returns:
        Formatted prompt string
    """
    # Format rewrite instructions as numbered list
    instructions_text = "\n".join(
        f"{i+1}. {instr}"
        for i, instr in enumerate(rewrite_instructions)
    )
    
    # Serialize critique as formatted JSON
    critique_text = json.dumps(critique_json, indent=2)
    
    return template.format(
        prompt=prompt_text,
        draft_output=draft_output,
        critique_json=critique_text,
        rewrite_instructions=instructions_text,
        payload_type=payload_type
    )


def build_refinement_request(
    prompt: str,
    generation_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Build teacher request for refinement."""
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert travel advisor AI. Follow instructions exactly and output only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "generation_config": generation_config
    }


def parse_refined_output(
    raw_text: str,
    gating_mode: str
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    """
    Parse and validate refined output.
    
    Args:
        raw_text: Raw teacher response
        gating_mode: "strict" or "lenient"
    
    Returns:
        Tuple of (parsed_envelope, contract_dict)
        Contract dict contains: envelope_ok, payload_ok, parse_mode, errors
    """
    # Try strict first
    strict_result = parse_and_validate(raw_text, strict_json=True)
    
    if strict_result.success:
        # Strict parse succeeded
        return strict_result.data, {
            "envelope_ok": True,
            "payload_ok": True,
            "parse_mode": "strict",
            "errors": []
        }
    
    # If gating_mode is strict, don't try lenient
    if gating_mode == "strict":
        return None, {
            "envelope_ok": False,
            "payload_ok": False,
            "parse_mode": "strict",
            "errors": [strict_result.error.message if strict_result.error else "Parse failed"]
        }
    
    # Try lenient
    lenient_result = parse_and_validate(raw_text, strict_json=False)
    
    if lenient_result.success:
        return lenient_result.data, {
            "envelope_ok": True,
            "payload_ok": True,
            "parse_mode": "lenient",
            "errors": []
        }
    
    # Both failed
    return None, {
        "envelope_ok": False,
        "payload_ok": False,
        "parse_mode": "lenient",
        "errors": [lenient_result.error.message if lenient_result.error else "Parse failed"]
    }


def truncate_text(text: str, max_length: int = 20000) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "\n... [truncated]"


def build_accepted_record(
    sample: JoinedSample,
    refined_envelope: Dict[str, Any],
    teacher_metadata: Dict[str, Any],
    gates: Dict[str, Any],
    overall_ok: bool
) -> Dict[str, Any]:
    """Build record for accepted sample."""
    plan = sample.prompt_plan
    
    return {
        "id": sample.id,
        "category": plan.get("category"),
        "difficulty": plan.get("difficulty"),
        "region_tags": plan.get("region_tags", []),
        "payload_type": plan.get("payload_type"),
        "template_version": plan.get("template_version"),
        "template_name": plan.get("template_name"),
        "prompt": plan.get("prompt"),
        "response": refined_envelope,
        "teacher": teacher_metadata,
        "quality": {
            "overall_ok": overall_ok,
            "gates": {
                name: {
                    "passed": gate.passed,
                    "reason_code": gate.reason_code,
                    "details": gate.details
                }
                for name, gate in gates.items()
            }
        },
        "created_at": datetime.now(timezone.utc).isoformat()
    }


def build_rejected_record(
    sample: JoinedSample,
    reason_codes: List[str],
    contract: Dict[str, Any],
    gates: Dict[str, Any] | None,
    raw_text: str,
    teacher_metadata: Dict[str, Any] | None
) -> Dict[str, Any]:
    """Build record for rejected sample."""
    # Build critique summary
    critique_summary = None
    if sample.critique:
        critique_summary = {
            "verdict": sample.critique.get("verdict"),
            "top_issues": [
                {
                    "type": issue.get("type"),
                    "severity": issue.get("severity"),
                    "message": issue.get("message")
                }
                for issue in sample.critique.get("issues", [])[:3]  # Top 3
            ]
        }
    
    record = {
        "id": sample.id,
        "reason_codes": reason_codes,
        "contract": contract,
        "critique_summary": critique_summary,
        "raw_refined_text": truncate_text(raw_text),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    if gates:
        record["gates"] = {
            name: {
                "passed": gate.passed,
                "reason_code": gate.reason_code,
                "details": gate.details
            }
            for name, gate in gates.items()
        }
    
    if teacher_metadata:
        record["teacher"] = teacher_metadata
    
    return record


def load_existing_ids(dataset_path: Path, rejected_path: Path) -> set[str]:
    """Load IDs that have already been processed (accepted or rejected)."""
    existing_ids = set()
    
    # Load accepted IDs
    if dataset_path.exists():
        for record in load_jsonl(dataset_path):
            existing_ids.add(record["id"])
    
    # Load rejected IDs
    if rejected_path.exists():
        for record in load_jsonl(rejected_path):
            existing_ids.add(record["id"])
    
    return existing_ids


def generate_dataset(
    plan_path: Path,
    drafts_path: Path,
    critiques_path: Path,
    out_dir: Path,
    limit: int | None,
    resume: bool,
    dry_run: bool,
    gating_mode: str,
    run_id: str | None,
    teacher_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main dataset generation pipeline.
    
    Returns:
        Stats dict with counts and metrics
    """
    # Generate run ID if not provided
    if not run_id:
        run_id = f"dataset_v1_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    print(f"[INFO] Starting dataset generation: {run_id}")
    print(f"[INFO] Gating mode: {gating_mode}")
    print(f"[INFO] Dry run: {dry_run}")
    
    # Load inputs
    print(f"[INFO] Loading inputs...")
    prompt_plans = load_jsonl(plan_path)
    drafts = load_jsonl(drafts_path)
    critiques = load_jsonl(critiques_path)
    
    print(f"[INFO] Loaded {len(prompt_plans)} plans, {len(drafts)} drafts, {len(critiques)} critiques")
    
    # Join inputs
    joined_samples, missing_rejections = join_inputs(prompt_plans, drafts, critiques)
    print(f"[INFO] Joined {len(joined_samples)} samples")
    print(f"[INFO] Rejected {len(missing_rejections)} samples with missing inputs")
    
    # Load refinement template
    refine_template = load_prompt_template("refine")
    
    # Initialize teacher client
    teacher = None
    if not dry_run:
        # Get model configuration
        primary_model = teacher_config.get("primary_model", "meta-llama/llama-3.1-70b-instruct:free")
        
        teacher = TeacherRouterClient(
            primary_client=OpenRouterTeacherClient(model=primary_model),
            fallback_clients=[],
            rate_limit_rpm=teacher_config.get("rate_limit_rpm", 30)
        )
    
    # Setup output paths
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "dataset_v1.jsonl"
    rejected_path = out_dir / "dataset_v1_rejected.jsonl"
    manifest_path = out_dir / "dataset_v1_manifest.json"
    stats_path = out_dir / "dataset_v1_stats.json"
    
    # Resume support: load existing IDs
    existing_ids = set()
    if resume:
        existing_ids = load_existing_ids(dataset_path, rejected_path)
        print(f"[INFO] Resume mode: skipping {len(existing_ids)} already-processed IDs")
    
    # Write missing rejections immediately
    with open(rejected_path, "a", encoding="utf-8") as f_rej:
        for rejection in missing_rejections:
            f_rej.write(json.dumps(rejection, ensure_ascii=False) + "\n")
    
    # Track stats
    stats = {
        "attempted": 0,
        "accepted": 0,
        "rejected": 0,
        "skipped_critique_reject": 0,
        "skipped_resume": 0,
        "gate_failures": {},
        "verdict_distribution": {},
        "category_counts": {},
        "payload_type_counts": {},
        "difficulty_counts": {},
        "teacher_model_usage": {},
        "parse_mode_distribution": {}
    }
    
    # Open output files
    with open(dataset_path, "a", encoding="utf-8") as f_accept, \
         open(rejected_path, "a", encoding="utf-8") as f_reject:
        
        for i, sample in enumerate(joined_samples):
            # Apply limit
            if limit and stats["attempted"] >= limit:
                print(f"[INFO] Reached limit of {limit} attempts")
                break
            
            # Skip if already processed
            if sample.id in existing_ids:
                stats["skipped_resume"] += 1
                continue
            
            # Track verdict distribution
            if sample.critique:
                verdict = sample.critique.get("verdict", "unknown")
                stats["verdict_distribution"][verdict] = stats["verdict_distribution"].get(verdict, 0) + 1
            
            # Skip if critique verdict is "reject"
            if sample.critique and sample.critique.get("verdict") == "reject":
                stats["skipped_critique_reject"] += 1
                rejection = {
                    "id": sample.id,
                    "reason_codes": ["critique_reject"],
                    "critique_summary": {
                        "verdict": "reject",
                        "top_issues": [
                            {
                                "type": issue.get("type"),
                                "severity": issue.get("severity"),
                                "message": issue.get("message")
                            }
                            for issue in sample.critique.get("issues", [])[:3]
                        ]
                    },
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                f_reject.write(json.dumps(rejection, ensure_ascii=False) + "\n")
                continue
            
            stats["attempted"] += 1
            
            # Track category/payload_type/difficulty
            category = sample.prompt_plan.get("category", "unknown")
            payload_type = sample.prompt_plan.get("payload_type", "unknown")
            difficulty = sample.prompt_plan.get("difficulty", "unknown")
            
            stats["category_counts"][category] = stats["category_counts"].get(category, 0) + 1
            stats["payload_type_counts"][payload_type] = stats["payload_type_counts"].get(payload_type, 0) + 1
            stats["difficulty_counts"][difficulty] = stats["difficulty_counts"].get(difficulty, 0) + 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"[INFO] Processed {i + 1}/{len(joined_samples)} samples "
                      f"(accepted: {stats['accepted']}, rejected: {stats['rejected']})")
            
            # Build refinement prompt
            try:
                rewrite_instructions = sample.critique.get("rewrite_instructions", []) if sample.critique else []
                
                refinement_prompt = build_refinement_prompt(
                    prompt_text=sample.prompt_plan.get("prompt", ""),
                    draft_output=sample.draft.get("output_text", ""),
                    critique_json=sample.critique if sample.critique else {"verdict": "pass", "issues": []},
                    rewrite_instructions=rewrite_instructions,
                    payload_type=payload_type,
                    template=refine_template
                )
            except Exception as e:
                print(f"[ERROR] Failed to build prompt for {sample.id}: {e}", file=sys.stderr)
                rejection = build_rejected_record(
                    sample=sample,
                    reason_codes=["prompt_build_error"],
                    contract={"errors": [str(e)]},
                    gates=None,
                    raw_text="",
                    teacher_metadata=None
                )
                f_reject.write(json.dumps(rejection, ensure_ascii=False) + "\n")
                stats["rejected"] += 1
                continue
            
            # Call teacher (or simulate in dry run)
            if dry_run:
                # Simulate: first sample passes, second fails verification
                if stats["attempted"] == 1:
                    raw_response = json.dumps({
                        "summary": "Tourists from most countries can stay 90 days visa-free",
                        "assumptions": ["Tourist purpose", "Valid passport"],
                        "uncertainty_notes": "Visa policies may change",
                        "next_steps": ["Check embassy website before travel"],
                        "verification_steps": ["Visit embassy.gov for current requirements"],
                        "payload_type": payload_type,
                        "payload": create_mock_payload(payload_type)
                    })
                else:
                    raw_response = json.dumps({
                        "summary": "Visa requirements vary by country",
                        "assumptions": [],
                        "uncertainty_notes": "",
                        "next_steps": [],
                        "verification_steps": [],
                        "payload_type": payload_type,
                        "payload": create_mock_payload(payload_type)
                    })
                
                teacher_metadata = {
                    "provider": "dry_run",
                    "selected_model": "mock",
                    "attempted_models": ["mock"],
                    "request_id": f"dry_run_{sample.id}",
                    "timing": {"total_time_ms": 0},
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0}
                }
            else:
                request = build_refinement_request(
                    prompt=refinement_prompt,
                    generation_config=teacher_config.get("generation_config", {})
                )
                
                try:
                    response = teacher.generate(**request)
                    raw_response = response["content"]
                    teacher_metadata = {
                        "provider": response.get("provider"),
                        "selected_model": response.get("model"),
                        "attempted_models": response.get("attempted_models", []),
                        "request_id": response.get("request_id"),
                        "timing": response.get("timing", {}),
                        "usage": response.get("usage", {})
                    }
                    
                    # Track model usage
                    model = response.get("model", "unknown")
                    stats["teacher_model_usage"][model] = stats["teacher_model_usage"].get(model, 0) + 1
                    
                except Exception as e:
                    print(f"[ERROR] Teacher call failed for {sample.id}: {e}", file=sys.stderr)
                    rejection = build_rejected_record(
                        sample=sample,
                        reason_codes=["teacher_error"],
                        contract={"errors": [str(e)]},
                        gates=None,
                        raw_text="",
                        teacher_metadata=None
                    )
                    f_reject.write(json.dumps(rejection, ensure_ascii=False) + "\n")
                    stats["rejected"] += 1
                    continue
            
            # Parse and validate
            parsed_envelope, contract = parse_refined_output(raw_response, gating_mode)
            
            # Track parse mode
            parse_mode = contract.get("parse_mode", "unknown")
            stats["parse_mode_distribution"][parse_mode] = stats["parse_mode_distribution"].get(parse_mode, 0) + 1
            
            # Apply quality gates
            if parsed_envelope:
                gates = apply_all_gates(
                    envelope_ok=contract["envelope_ok"],
                    payload_ok=contract["payload_ok"],
                    parse_mode=parse_mode,
                    parsed_envelope=parsed_envelope,
                    full_response_text=raw_response
                )
                
                overall_ok = compute_overall_quality(gates)
                
                if overall_ok:
                    # Accept
                    accepted_record = build_accepted_record(
                        sample=sample,
                        refined_envelope=parsed_envelope,
                        teacher_metadata=teacher_metadata,
                        gates=gates,
                        overall_ok=True
                    )
                    f_accept.write(json.dumps(accepted_record, ensure_ascii=False) + "\n")
                    stats["accepted"] += 1
                else:
                    # Reject due to gates
                    reason_codes = get_rejection_reasons(gates)
                    for code in reason_codes:
                        stats["gate_failures"][code] = stats["gate_failures"].get(code, 0) + 1
                    
                    rejected_record = build_rejected_record(
                        sample=sample,
                        reason_codes=reason_codes,
                        contract=contract,
                        gates=gates,
                        raw_text=raw_response,
                        teacher_metadata=teacher_metadata
                    )
                    f_reject.write(json.dumps(rejected_record, ensure_ascii=False) + "\n")
                    stats["rejected"] += 1
            else:
                # Reject due to parse failure
                stats["gate_failures"]["parse_failed"] = stats["gate_failures"].get("parse_failed", 0) + 1
                
                rejected_record = build_rejected_record(
                    sample=sample,
                    reason_codes=["parse_failed"],
                    contract=contract,
                    gates=None,
                    raw_text=raw_response,
                    teacher_metadata=teacher_metadata
                )
                f_reject.write(json.dumps(rejected_record, ensure_ascii=False) + "\n")
                stats["rejected"] += 1
    
    # Final stats
    stats["missing_inputs"] = len(missing_rejections)
    stats["total_processed"] = stats["attempted"] + stats["skipped_critique_reject"] + stats["skipped_resume"]
    
    print(f"\n[INFO] Dataset generation complete")
    print(f"[INFO] Accepted: {stats['accepted']}")
    print(f"[INFO] Rejected: {stats['rejected']}")
    print(f"[INFO] Skipped (critique reject): {stats['skipped_critique_reject']}")
    print(f"[INFO] Skipped (resume): {stats['skipped_resume']}")
    print(f"[INFO] Missing inputs: {stats['missing_inputs']}")
    
    # Write manifest
    write_manifest(
        manifest_path=manifest_path,
        run_id=run_id,
        plan_path=plan_path,
        drafts_path=drafts_path,
        critiques_path=critiques_path,
        dataset_path=dataset_path,
        rejected_path=rejected_path,
        teacher_config=teacher_config,
        gating_mode=gating_mode,
        stats=stats
    )
    
    # Write detailed stats
    write_stats(stats_path, stats)
    
    return stats


def write_manifest(
    manifest_path: Path,
    run_id: str,
    plan_path: Path,
    drafts_path: Path,
    critiques_path: Path,
    dataset_path: Path,
    rejected_path: Path,
    teacher_config: Dict[str, Any],
    gating_mode: str,
    stats: Dict[str, Any]
) -> None:
    """Write manifest file with reproducibility metadata."""
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "prompt_plan": {
                "path": str(plan_path),
                "sha256": hash_file(plan_path)
            },
            "drafts": {
                "path": str(drafts_path),
                "sha256": hash_file(drafts_path)
            },
            "critiques": {
                "path": str(critiques_path),
                "sha256": hash_file(critiques_path)
            }
        },
        "outputs": {
            "dataset": str(dataset_path),
            "rejected": str(rejected_path),
            "stats": str(manifest_path.parent / "dataset_v1_stats.json")
        },
        "config": {
            "gating_mode": gating_mode,
            "teacher": {
                k: v for k, v in teacher_config.items()
                if k != "api_key"  # Redact sensitive info
            }
        },
        "schema_versions": {
            "envelope": "v0",
            "payload": "v1"
        },
        "counts": {
            "attempted": stats["attempted"],
            "accepted": stats["accepted"],
            "rejected": stats["rejected"],
            "skipped_critique_reject": stats["skipped_critique_reject"],
            "missing_inputs": stats["missing_inputs"]
        }
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Wrote manifest: {manifest_path}")


def write_stats(stats_path: Path, stats: Dict[str, Any]) -> None:
    """Write detailed statistics file."""
    # Compute rates
    total_attempts = stats["attempted"]
    acceptance_rate = (stats["accepted"] / total_attempts * 100) if total_attempts > 0 else 0
    
    # Compute gate pass rates
    gate_pass_rates = {}
    for gate_name in ["contract_ok", "verification_when_needed", "overconfidence_guard"]:
        failures = stats["gate_failures"].get(gate_name, 0)
        passes = total_attempts - failures
        rate = (passes / total_attempts * 100) if total_attempts > 0 else 0
        gate_pass_rates[gate_name] = round(rate, 2)
    
    detailed_stats = {
        "summary": {
            "total_attempted": total_attempts,
            "accepted": stats["accepted"],
            "rejected": stats["rejected"],
            "acceptance_rate_pct": round(acceptance_rate, 2)
        },
        "gate_pass_rates": gate_pass_rates,
        "rejection_reasons": stats["gate_failures"],
        "verdict_distribution": stats["verdict_distribution"],
        "category_counts": stats["category_counts"],
        "payload_type_counts": stats["payload_type_counts"],
        "difficulty_counts": stats["difficulty_counts"],
        "teacher_model_usage": stats["teacher_model_usage"],
        "parse_mode_distribution": stats["parse_mode_distribution"],
        "skipped": {
            "critique_reject": stats["skipped_critique_reject"],
            "missing_inputs": stats["missing_inputs"],
            "resume": stats["skipped_resume"]
        }
    }
    
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(detailed_stats, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Wrote stats: {stats_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate final training dataset from plans, drafts, and critiques"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/interim/prompt_plan_v1.jsonl"),
        help="Path to prompt plan JSONL"
    )
    parser.add_argument(
        "--drafts",
        type=Path,
        default=Path("data/interim/drafts_v1.jsonl"),
        help="Path to drafts JSONL"
    )
    parser.add_argument(
        "--critiques",
        type=Path,
        default=Path("data/interim/critiques_v1.jsonl"),
        help="Path to critiques JSONL"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Max number of samples to attempt (for testing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-processed IDs"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Simulate without calling teacher API"
    )
    parser.add_argument(
        "--gating_mode",
        choices=["strict", "lenient"],
        default="lenient",
        help="Parsing mode: strict (JSON only) or lenient (allow markdown)"
    )
    parser.add_argument(
        "--run_id",
        help="Custom run ID for reproducibility tracking"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to teacher config YAML (default: use sensible defaults)"
    )
    
    args = parser.parse_args()
    
    # Load teacher config
    if args.config and args.config.exists():
        with open(args.config, "r") as f:
            teacher_config = yaml.safe_load(f)
    else:
        teacher_config = {
            "rate_limit_rpm": 30,
            "generation_config": {
                "temperature": 0.3,
                "max_tokens": 2048,
                "top_p": 0.95
            }
        }
    
    # Run pipeline
    stats = generate_dataset(
        plan_path=args.plan,
        drafts_path=args.drafts,
        critiques_path=args.critiques,
        out_dir=args.out_dir,
        limit=args.limit,
        resume=args.resume,
        dry_run=args.dry_run,
        gating_mode=args.gating_mode,
        run_id=args.run_id,
        teacher_config=teacher_config
    )
    
    # Exit with appropriate code
    if stats["accepted"] == 0:
        print("[WARN] No samples accepted - check rejection reasons", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
