"""
Base Model Evaluation Report Generator v0

Generates a human-readable markdown report from evaluation artifacts:
- metrics.json
- base_model_outputs.jsonl
- meta.json

The report includes:
- Experimental setup
- Metrics summary table
- Failure taxonomy v0
- 10 curated failure examples
- Summary and next steps

This module does NOT re-run evaluation or compute new metrics.
It only consumes existing outputs.
"""

import argparse
import json
import sys
from pathlib import Path


def generate_report(
    run_dir: Path,
    output_path: Path,
    overrides_path: Path | None = None,
) -> None:
    """
    Generate evaluation report from existing artifacts.

    Args:
        run_dir: Path to runs/eval/<timestamp> directory
        output_path: Path where markdown report will be written
        overrides_path: Optional path to config with curated_failure_ids

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data is malformed
    """
    # Load required files
    metrics_path = run_dir / "metrics.json"
    outputs_path = run_dir / "base_model_outputs.jsonl"
    meta_path = run_dir / "meta.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {run_dir}")
    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing base_model_outputs.jsonl in {run_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {run_dir}")

    with open(metrics_path) as f:
        metrics = json.load(f)

    with open(meta_path) as f:
        meta = json.load(f)

    # Load all outputs
    outputs = []
    with open(outputs_path) as f:
        for line in f:
            if line.strip():
                outputs.append(json.loads(line))

    # Load manual overrides if provided
    manual_ids = []
    if overrides_path and overrides_path.exists():
        with open(overrides_path) as f:
            import yaml

            config = yaml.safe_load(f)
            manual_ids = config.get("curated_failure_ids", [])

    # Select curated failures
    curated = _select_curated_failures(outputs, manual_ids, target_count=10)

    # Generate report
    report = _build_report(metrics, meta, curated, run_dir)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report generated: {output_path}")


def _select_curated_failures(
    outputs: list[dict],
    manual_ids: list[str],
    target_count: int = 10,
) -> list[dict]:
    """
    Select up to target_count failure examples.

    Priority:
    1. Manual IDs from overrides
    2. Auto-selected by severity score

    Auto-selection criteria:
    - Failed strict JSON = +3
    - Failed required fields = +2
    - Missing verification marker in uncertainty/safety = +1
    - Latency outlier (>p90) = +1

    Ensures diversity across suites.
    """
    # Build lookup by id (or example_id for backward compat)
    by_id = {o.get("id", o.get("example_id")): o for o in outputs}

    selected = []

    # Add manual IDs first
    for example_id in manual_ids:
        if example_id in by_id:
            selected.append(by_id[example_id])
        if len(selected) >= target_count:
            return selected

    # Compute severity scores for remaining
    scored = []
    for output in outputs:
        output_id = output.get("id", output.get("example_id"))
        selected_ids = [s.get("id", s.get("example_id")) for s in selected]
        if output_id in selected_ids:
            continue  # Already selected manually

        checks = output.get("checks", {})
        score = 0

        # Strict JSON failure
        if not checks.get("strict_json_ok", True):
            score += 3

        # Required fields failure
        if checks.get("required_fields_ok") is False:
            score += 2

        # Missing verification marker in uncertainty/safety suites
        suite = output.get("suite", "")
        uncertainty = checks.get("uncertainty", {})
        if suite in ["uncertainty", "safety"]:
            if not uncertainty.get("has_verification_marker", False):
                score += 1

        # Latency outlier (simple heuristic: >5s)
        timing = output.get("timing", {})
        latency = timing.get("latency_s")
        if latency and latency > 5.0:
            score += 1

        scored.append((score, output_id, output))

    # Sort by score (descending), then by ID (for determinism)
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Ensure diversity: try to get at least one from each suite
    suites_seen = {s["suite"] for s in selected}

    # First pass: add one from each unseen suite
    for _score, _output_id, output in scored:
        if len(selected) >= target_count:
            break
        if output["suite"] not in suites_seen:
            selected.append(output)
            suites_seen.add(output["suite"])

    # Second pass: fill remaining slots by highest score
    for _score, output_id, output in scored:
        if len(selected) >= target_count:
            break
        selected_ids = [s.get("id", s.get("example_id")) for s in selected]
        if output_id not in selected_ids:
            selected.append(output)

    return selected


def _build_report(
    metrics: dict,
    meta: dict,
    curated: list[dict],
    run_dir: Path,
) -> str:
    """Build the markdown report."""
    lines = []

    # 1) Title + Overview
    model_id = meta.get("model_id", "unknown")
    lines.append("# Base Model Evaluation Report v0\n")
    lines.append(
        f"This report evaluates **{model_id}** to establish a "
        "pre-adaptation baseline. It summarizes automated metrics, "
        "defines a failure taxonomy, and presents curated failure examples.\n"
    )

    # 2) Experimental Setup
    lines.append("## Experimental Setup\n")
    lines.append(f"- **Model ID**: {model_id}\n")
    lines.append(f"- **Revision**: {meta.get('revision', 'N/A')}\n")
    lines.append(f"- **Device**: {meta.get('device', 'N/A')}\n")
    lines.append(f"- **Dtype**: {meta.get('dtype', 'N/A')}\n")
    lines.append(f"- **Seed**: {meta.get('seed', 'N/A')}\n")

    # Generation config
    gen_config = meta.get("generation_config", {})
    if gen_config:
        lines.append("\n**Generation Config**:\n")
        for key, value in sorted(gen_config.items()):
            lines.append(f"- {key}: {value}\n")

    # Benchmark suites
    suite_counts = meta.get("suite_counts", {})
    lines.append("\n**Benchmark Suites**:\n")
    for suite, count in sorted(suite_counts.items()):
        lines.append(f"- {suite}: {count} examples\n")
    lines.append("")

    # 3) Metrics Summary
    lines.append("## Metrics Summary\n")
    lines.append(_build_metrics_table(metrics))
    lines.append("")

    # 4) Failure Taxonomy v0
    lines.append("## Failure Taxonomy v0\n")
    lines.append(_build_taxonomy())
    lines.append("")

    # 5) Curated Failure Examples
    lines.append("## Curated Failure Examples\n")
    lines.append(
        f"The following {len(curated)} examples illustrate key weaknesses " "of the base model:\n"
    )
    for i, output in enumerate(curated, start=1):
        lines.append(_build_failure_example(i, output))
    lines.append("")

    # 6) Summary & Next Steps
    lines.append("## Summary & Next Steps\n")
    lines.append(_build_summary(metrics, curated))
    lines.append("")

    return "".join(lines)


def _build_metrics_table(metrics: dict) -> str:
    """Build markdown table from metrics.json."""
    overall = metrics.get("overall", {})
    by_suite = metrics.get("by_suite", {})

    lines = []
    lines.append(
        "| Suite | N | Strict JSON % | Lenient JSON % | Required Fields % | Assumptions % | Verification % | Clarifying % | Avg Latency (s) | p90 Latency (s) |\n"
    )
    lines.append(
        "|-------|---|---------------|----------------|-------------------|---------------|----------------|--------------|-----------------|------------------|\n"
    )

    # Overall row
    lines.append(_format_metric_row("**Overall**", overall))

    # Suite rows (sorted)
    for suite in sorted(by_suite.keys()):
        suite_metrics = by_suite[suite]
        lines.append(_format_metric_row(suite, suite_metrics))

    return "".join(lines)


def _format_metric_row(label: str, data: dict) -> str:
    """Format one row of the metrics table."""

    def pct(value):
        if value is None:
            return "—"
        return f"{value * 100:.1f}"

    def num(value):
        if value is None:
            return "—"
        return f"{value:.2f}"

    n = data.get("n", 0)
    strict = pct(data.get("strict_json_parse_rate"))
    lenient = pct(data.get("lenient_json_parse_rate"))
    required = pct(data.get("required_fields_rate"))
    assumptions = pct(data.get("assumptions_marker_rate"))
    verification = pct(data.get("verification_marker_rate"))
    clarifying = pct(data.get("clarifying_questions_rate"))
    avg_latency = num(data.get("avg_latency_s"))
    p90_latency = num(data.get("p90_latency_s"))

    return f"| {label} | {n} | {strict} | {lenient} | {required} | {assumptions} | {verification} | {clarifying} | {avg_latency} | {p90_latency} |\n"


def _build_taxonomy() -> str:
    """Define failure taxonomy v0 (static)."""
    return """
This taxonomy classifies common failure modes observed in the base model:

1. **JSON Format Violations**: Output is not valid JSON or cannot be extracted from markdown code fences.

2. **Schema / Required-Field Failures**: Output is valid JSON but missing required fields specified in the task.

3. **Hallucinated Specificity**: Model invents concrete details (dates, names, numbers) without sufficient input context.

4. **Missed Constraints**: Model ignores explicit constraints in the prompt (e.g., budget limits, date ranges).

5. **Overconfident Answers Under Uncertainty**: Model provides definitive answers when input is ambiguous or incomplete, without expressing uncertainty.

6. **Weak Verification Guidance**: Model fails to suggest verification steps or alternative approaches when appropriate.

7. **Safety Overreach or Vagueness**: Model either refuses benign requests or provides overly generic safety warnings.
"""


def _build_failure_example(index: int, output: dict) -> str:
    """Format one curated failure example."""
    lines = []
    output_id = output.get("id", output.get("example_id", "unknown"))
    lines.append(f"### Example {index}: {output_id}\n")
    lines.append(f"**Suite**: {output.get('suite', 'unknown')}\n\n")

    # Prompt (truncated)
    prompt = output.get("prompt", "")
    prompt_display = _truncate(prompt, 400)
    lines.append("**Prompt**:\n```\n")
    lines.append(prompt_display)
    lines.append("\n```\n\n")

    # Model output (truncated)
    model_output = output.get("output_text", output.get("output", ""))
    output_display = _truncate(model_output, 600)
    lines.append("**Model Output**:\n```\n")
    lines.append(output_display)
    lines.append("\n```\n\n")

    # Failed checks
    checks = output.get("checks", {})
    lines.append("**Failed Checks**:\n")

    if not checks.get("strict_json_ok", True):
        lines.append("- ❌ Strict JSON parse failed\n")
    if not checks.get("lenient_json_ok", True):
        lines.append("- ❌ Lenient JSON parse failed\n")
    if checks.get("required_fields_ok") is False:
        missing = checks.get("missing_fields", [])
        lines.append(f"- ❌ Missing required fields: {', '.join(missing)}\n")

    uncertainty = checks.get("uncertainty", {})
    if not uncertainty.get("has_verification_marker", False):
        suite = output.get("suite", "")
        if suite in ["uncertainty", "safety"]:
            lines.append("- ⚠️  No verification marker in uncertainty/safety suite\n")
    if not uncertainty.get("has_assumptions_marker", False):
        suite = output.get("suite", "")
        if suite in ["uncertainty", "safety"]:
            lines.append("- ⚠️  No assumptions marker in uncertainty/safety suite\n")

    # Taxonomy category
    category = _assign_taxonomy_category(output)
    lines.append(f"\n**Taxonomy Category**: {category}\n\n")

    return "".join(lines)


def _assign_taxonomy_category(output: dict) -> str:
    """Assign a taxonomy category based on failed checks."""
    checks = output.get("checks", {})
    suite = output.get("suite", "")

    # First matching category
    if not checks.get("strict_json_ok", True):
        return "JSON Format Violations"
    if checks.get("required_fields_ok") is False:
        return "Schema / Required-Field Failures"

    uncertainty = checks.get("uncertainty", {})
    if suite in ["uncertainty", "safety"]:
        if not uncertainty.get("has_verification_marker", False):
            return "Weak Verification Guidance"
        if not uncertainty.get("has_assumptions_marker", False):
            return "Overconfident Answers Under Uncertainty"

    if suite == "safety":
        return "Safety Overreach or Vagueness"

    # Default
    return "Missed Constraints"


def _build_summary(metrics: dict, curated: list[dict]) -> str:
    """Build summary and next steps section."""
    overall = metrics.get("overall", {})

    strict_rate = overall.get("strict_json_parse_rate", 0) or 0
    lenient_rate = overall.get("lenient_json_parse_rate", 0) or 0
    required_rate = overall.get("required_fields_rate", 0) or 0
    verification_rate = overall.get("verification_marker_rate", 0) or 0

    lines = []
    lines.append("**Key Findings**:\n\n")

    # JSON parsing
    if strict_rate < 0.5:
        lines.append(
            f"- The base model struggles with strict JSON formatting "
            f"({strict_rate * 100:.1f}% success rate), though lenient parsing "
            f"recovers some outputs ({lenient_rate * 100:.1f}%).\n"
        )

    # Required fields
    if required_rate is not None and required_rate < 0.8:
        lines.append(
            f"- Schema compliance is weak: only {required_rate * 100:.1f}% of outputs "
            "include all required fields.\n"
        )

    # Verification markers
    if verification_rate < 0.5:
        lines.append(
            f"- The model rarely provides verification guidance "
            f"({verification_rate * 100:.1f}% of outputs).\n"
        )

    # Diversity
    suites_with_failures = {ex["suite"] for ex in curated}
    lines.append(
        f"- Failures span {len(suites_with_failures)} suites, indicating "
        "systemic rather than task-specific issues.\n"
    )

    lines.append("\n**Next Steps**:\n\n")
    lines.append(
        "- **Milestone 2**: Select and fine-tune a base model using LoRA/QLoRA "
        "to improve JSON adherence and schema compliance.\n"
    )
    lines.append(
        "- **Milestone 3**: Develop synthetic data generation for underrepresented "
        "failure modes (e.g., missed constraints, verification guidance).\n"
    )
    lines.append(
        "- **Milestone 4**: Implement iterative refinement to address "
        "hallucinated specificity and overconfidence.\n"
    )
    lines.append(
        "- **Milestone 5**: Evaluate adapted model and measure improvement " "across all metrics.\n"
    )

    return "".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate Base Model Evaluation Report v0")
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to runs/eval/<timestamp> directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evaluation_report_base_v0.md"),
        help="Output path for markdown report (default: docs/evaluation_report_base_v0.md)",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        help="Optional path to config YAML with curated_failure_ids",
    )

    args = parser.parse_args()

    try:
        generate_report(args.run_dir, args.output, args.overrides)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
