"""Prompt planning CLI for synthetic dataset generation.

Generates a deterministic prompt plan from a spec file without calling any teacher model.
"""

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_spec(spec_path: Path) -> dict[str, Any]:
    """Load dataset generation spec from YAML file.

    Args:
        spec_path: Path to spec YAML file

    Returns:
        Parsed spec dictionary

    Raises:
        ValueError: If spec is missing required keys
        FileNotFoundError: If spec file doesn't exist
    """
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    with open(spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    # Validate required keys
    required_keys = [
        "version",
        "seed",
        "total_examples",
        "payload_type_distribution",
        "category_distribution",
        "difficulty_levels",
        "region_tags",
        "locations",
        "user_profiles",
    ]
    missing = [k for k in required_keys if k not in spec]
    if missing:
        raise ValueError(f"Spec missing required keys: {missing}")

    return spec


def load_template(template_path: Path) -> str:
    """Load a prompt template file.

    Args:
        template_path: Path to template file

    Returns:
        Template content as string

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    with open(template_path, encoding="utf-8") as f:
        return f.read()


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def sample_distribution(
    distribution: dict[str, float], rng: random.Random, count: int
) -> list[str]:
    """Sample from a distribution deterministically.

    Args:
        distribution: Dict mapping keys to probabilities (must sum to 1.0)
        rng: Random number generator
        count: Number of samples to generate

    Returns:
        List of sampled keys
    """
    keys = list(distribution.keys())
    weights = [distribution[k] for k in keys]

    # Normalize weights to sum to 1.0
    total = sum(weights)
    if abs(total - 1.0) > 0.01:
        weights = [w / total for w in weights]

    return rng.choices(keys, weights=weights, k=count)


def generate_trip_context(
    rng: random.Random, include_constraints: bool, constraint_types: list[str]
) -> str:
    """Generate trip context with optional constraints.

    Args:
        rng: Random number generator
        include_constraints: Whether to include constraints
        constraint_types: Types of constraints to potentially include

    Returns:
        Trip context string
    """
    contexts = []

    if include_constraints:
        # Budget constraint
        if "budget_range" in constraint_types and rng.random() < 0.7:
            budgets = [
                "budget-friendly (under $1500)",
                "moderate budget ($2000-3500)",
                "comfortable budget ($4000-6000)",
            ]
            contexts.append(rng.choice(budgets))

        # Date constraint
        if "date_constraints" in constraint_types and rng.random() < 0.6:
            dates = [
                "traveling in March 2026",
                "summer vacation (June-August)",
                "winter holiday season",
                "spring break",
            ]
            contexts.append(rng.choice(dates))

        # Preferences
        if "preferences" in constraint_types and rng.random() < 0.5:
            prefs = [
                "family-friendly activities",
                "adventure and outdoor focus",
                "cultural and historical sites",
                "food and local cuisine focus",
                "relaxation and wellness",
            ]
            contexts.append(rng.choice(prefs))

        # Duration
        if "duration" in constraint_types and rng.random() < 0.7:
            durations = ["3-day weekend", "5 days", "1 week", "10-14 days", "2 weeks"]
            contexts.append(rng.choice(durations))

    if not contexts:
        return "Planning a trip with flexible dates and budget"

    return ", ".join(contexts)


def generate_prompt_plan(spec: dict[str, Any], templates_dir: Path) -> list[dict[str, Any]]:
    """Generate deterministic prompt plan from spec.

    Args:
        spec: Dataset generation spec
        templates_dir: Path to templates directory

    Returns:
        List of prompt plan records
    """
    # Initialize RNG with seed for determinism
    rng = random.Random(spec["seed"])

    total = spec["total_examples"]

    # Sample distributions
    payload_types = sample_distribution(spec["payload_type_distribution"], rng, total)
    categories = sample_distribution(spec["category_distribution"], rng, total)
    difficulties = sample_distribution(spec["difficulty_levels"], rng, total)
    regions = sample_distribution(spec["region_tags"], rng, total)

    # Determine which prompts should have ambiguity or constraints
    constraint_rate = spec.get("prompt_constraints", {}).get("constraint_rate", 0.5)
    constraint_flags = [rng.random() < constraint_rate for _ in range(total)]

    constraint_types = spec.get("prompt_constraints", {}).get("include_constraints", [])

    # Load templates
    templates = {}
    for payload_type in spec["payload_type_distribution"].keys():
        template_name = f"{payload_type}_draft.txt"
        template_path = templates_dir / template_name
        templates[payload_type] = load_template(template_path)

    # Generate prompt plan
    plan = []
    user_profiles = spec["user_profiles"]

    # Count by payload type for ID generation
    payload_counters = {pt: 0 for pt in spec["payload_type_distribution"].keys()}

    for i in range(total):
        payload_type = payload_types[i]
        category = categories[i]
        difficulty = difficulties[i]
        region = regions[i]

        # Generate stable ID
        payload_counters[payload_type] += 1
        record_id = f"v1_{payload_type}_{category}_{payload_counters[payload_type]:04d}"

        # Select location for this region
        region_locations = spec["locations"].get(region, [])
        if not region_locations:
            raise ValueError(f"No locations defined for region: {region}")

        location_data = rng.choice(region_locations)
        country = location_data["country"]
        city = rng.choice(location_data["cities"])

        # Generate context
        user_profile = rng.choice(user_profiles)
        trip_context = generate_trip_context(rng, constraint_flags[i], constraint_types)

        # Render prompt
        template = templates[payload_type]
        prompt = template.format(
            user_profile=user_profile,
            trip_context=trip_context,
            country=country,
            city=city,
            category=category,
            difficulty=difficulty,
        )

        # Build record
        record = {
            "id": record_id,
            "payload_type": payload_type,
            "category": category,
            "difficulty": difficulty,
            "region_tags": [region],
            "template_version": "teacher/v1",
            "template_name": f"{payload_type}_draft.txt",
            "prompt": prompt,
            "location": {"country": country, "city": city, "region": region},
            "created_by": "plan_prompts",
            "seed": spec["seed"],
        }

        plan.append(record)

    return plan


def write_jsonl(data: list[dict[str, Any]], output_path: Path) -> None:
    """Write data to JSONL file.

    Args:
        data: List of dictionaries to write
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")


def write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write data to formatted JSON file.

    Args:
        data: Dictionary to write
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def compute_stats(plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute statistics from prompt plan.

    Args:
        plan: List of prompt plan records

    Returns:
        Statistics dictionary
    """
    stats = {
        "total_examples": len(plan),
        "by_payload_type": {},
        "by_category": {},
        "by_difficulty": {},
        "by_region": {},
    }

    # Count by dimensions
    for record in plan:
        payload_type = record["payload_type"]
        category = record["category"]
        difficulty = record["difficulty"]
        region = record["region_tags"][0]  # First region tag

        stats["by_payload_type"][payload_type] = stats["by_payload_type"].get(payload_type, 0) + 1
        stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
        stats["by_difficulty"][difficulty] = stats["by_difficulty"].get(difficulty, 0) + 1
        stats["by_region"][region] = stats["by_region"].get(region, 0) + 1

    return stats


def create_manifest(
    spec_path: Path, templates_dir: Path, output_files: dict[str, Path], stats: dict[str, Any]
) -> dict[str, Any]:
    """Create manifest for prompt plan generation.

    Args:
        spec_path: Path to spec file
        templates_dir: Path to templates directory
        output_files: Dict of output file paths
        stats: Statistics dictionary

    Returns:
        Manifest dictionary
    """
    # Compute spec hash
    spec_hash = compute_file_hash(spec_path)

    # Compute template hashes
    template_hashes = {}
    for template_file in templates_dir.glob("*.txt"):
        template_hashes[template_file.name] = compute_file_hash(template_file)

    manifest = {
        "version": "v1",
        "generated_at": datetime.now().isoformat(),
        "spec_file": str(spec_path),
        "spec_hash": spec_hash,
        "templates_dir": str(templates_dir),
        "template_hashes": template_hashes,
        "output_files": {k: str(v) for k, v in output_files.items()},
        "counts": {
            "total_examples": stats["total_examples"],
            "payload_types": len(stats["by_payload_type"]),
            "categories": len(stats["by_category"]),
            "difficulties": len(stats["by_difficulty"]),
            "regions": len(stats["by_region"]),
        },
    }

    return manifest


def main(spec_path: str | None = None, out_dir: str | None = None) -> None:
    """Main entry point for prompt planner.

    Args:
        spec_path: Path to spec YAML file (default: data/specs/dataset_v1_spec.yaml)
        out_dir: Output directory (default: data/interim)
    """
    # Set defaults
    if spec_path is None:
        spec_path = "data/specs/dataset_v1_spec.yaml"
    if out_dir is None:
        out_dir = "data/interim"

    spec_path = Path(spec_path)
    out_dir = Path(out_dir)
    templates_dir = Path("data/prompts/teacher/v1")

    print(f"Loading spec from: {spec_path}")
    spec = load_spec(spec_path)

    print(f"Generating prompt plan with seed={spec['seed']}, total={spec['total_examples']}")
    plan = generate_prompt_plan(spec, templates_dir)

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_path = out_dir / "prompt_plan_v1.jsonl"
    print(f"Writing prompt plan to: {plan_path}")
    write_jsonl(plan, plan_path)

    # Compute stats
    stats = compute_stats(plan)
    stats_path = out_dir / "prompt_plan_v1_stats.json"
    print(f"Writing stats to: {stats_path}")
    write_json(stats, stats_path)

    # Create manifest
    output_files = {"plan": plan_path, "stats": stats_path}
    manifest = create_manifest(spec_path, templates_dir, output_files, stats)
    manifest_path = out_dir / "prompt_plan_v1_manifest.json"
    print(f"Writing manifest to: {manifest_path}")
    write_json(manifest, manifest_path)

    print("\nPrompt plan generation complete!")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Payload types: {list(stats['by_payload_type'].keys())}")
    print(f"  Categories: {len(stats['by_category'])}")
    print(f"  Difficulties: {list(stats['by_difficulty'].keys())}")
    print(f"  Regions: {list(stats['by_region'].keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate deterministic prompt plan from dataset spec"
    )
    parser.add_argument(
        "--spec",
        type=str,
        default="data/specs/dataset_v1_spec.yaml",
        help="Path to dataset spec YAML file",
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/interim", help="Output directory for generated files"
    )

    args = parser.parse_args()
    main(spec_path=args.spec, out_dir=args.out_dir)
