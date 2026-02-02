"""
Lesson 5.5: Training report v1 generator.

Generates a concise, reproducible markdown report from a single training run's
artifacts (config, meta, train_metrics, samples). No new computation; fully
from existing files.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Error codes from parsing.py for failure taxonomy
JSON_STRICT_PARSE_FAILED = "JSON_STRICT_PARSE_FAILED"
JSON_LENIENT_PARSE_FAILED = "JSON_LENIENT_PARSE_FAILED"
ENVELOPE_SCHEMA_FAILED = "ENVELOPE_SCHEMA_FAILED"
PAYLOAD_SCHEMA_FAILED = "PAYLOAD_SCHEMA_FAILED"

REQUIRED_ARTIFACTS = [
    "config.yaml",
    "meta.json",
    "train_metrics.json",
    "samples/base_outputs.jsonl",
    "samples/finetuned_outputs.jsonl",
    "samples/comparison_metrics.json",
]


def _require_run_dir(run_dir: Path) -> None:
    """Validate run_dir and required artifacts; raise with clear message if missing."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    missing = []
    for rel in REQUIRED_ARTIFACTS:
        if not (run_dir / rel).exists():
            missing.append(rel)
    if missing:
        raise FileNotFoundError(
            f"Missing required artifacts in {run_dir}: {', '.join(missing)}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _escape_md(s: str, max_len: int = 300) -> str:
    """Shorten and escape markdown-sensitive characters for safe inclusion in report."""
    s = s.replace("\\", "\\\\").replace("`", "\\`").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _build_overview(run_dir: Path, meta: dict, cfg: dict, train_metrics: dict) -> str:
    run_id = meta.get("run_id") or run_dir.name
    timestamp = meta.get("timestamp", "")
    base_model_id = meta.get("base_model_id") or cfg.get("base_model_id", "")
    final = train_metrics.get("final", {})
    total_steps = final.get("train_steps") or train_metrics.get("global_steps", "")
    return f"""## 1. Overview

| Field | Value |
|-------|-------|
| run_id | {run_id} |
| date/time | {timestamp} |
| base_model_id | {base_model_id} |
| fine-tuning method | LoRA |
| dataset version | dataset_v1_clean |
| total training steps | {total_steps} |
"""


def _build_config_table(cfg: dict, meta: dict) -> str:
    lora = cfg.get("lora", {})
    training = cfg.get("training", {})
    data = cfg.get("data", {})
    runtime = cfg.get("runtime", {})
    resolved = meta.get("resolved", {})
    device = resolved.get("device") or runtime.get("device", "")
    precision = resolved.get("precision") or runtime.get("precision", "")
    return f"""## 2. Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | {lora.get('r', '')} |
| LoRA alpha | {lora.get('alpha', '')} |
| LoRA dropout | {lora.get('dropout', '')} |
| batch_size | {training.get('batch_size', '')} |
| grad_accum_steps | {training.get('grad_accum_steps', '')} |
| learning_rate | {training.get('lr', '')} |
| max_seq_len | {data.get('max_seq_len', '')} |
| device | {device} |
| precision | {precision} |
| seed | {training.get('seed', '')} |
"""


def _build_quantitative_metrics(train_metrics: dict, comparison: dict) -> str:
    final = train_metrics.get("final", {})
    train_list = train_metrics.get("train", [])
    val_list = train_metrics.get("val", [])
    final_train_loss = train_list[-1]["loss"] if train_list else None
    final_val_loss = final.get("val_loss")
    best_val_loss = min(v["loss"] for v in val_list) if val_list else None
    tokens_seen = final.get("tokens_seen")

    s31 = """### 3.1 Training & Validation

| Metric | Value |
|--------|-------|
| final training loss | """
    s31 += str(round(final_train_loss, 4)) if final_train_loss is not None else "—"
    s31 += " |\n| final validation loss | "
    s31 += str(round(final_val_loss, 4)) if final_val_loss is not None else "—"
    s31 += " |\n| best validation loss | "
    s31 += str(round(best_val_loss, 4)) if best_val_loss is not None else "—"
    s31 += " |\n| total tokens seen | "
    s31 += str(tokens_seen) if tokens_seen is not None else "—"
    s31 += " |\n\n"

    base_m = comparison.get("base", {})
    finetuned_m = comparison.get("finetuned", {})
    delta_m = comparison.get("delta", {})

    def _rate(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, (int, float)):
            return str(round(v, 4))
        return str(v)

    s32 = """### 3.2 Structured Output Metrics (Fixed Prompt Suite)

| Metric | Base | Finetuned | Delta |
|--------|------|-----------|-------|
| parse_success_rate | """
    s32 += f"{_rate(base_m.get('parse_success_rate'))} | {_rate(finetuned_m.get('parse_success_rate'))} | {_rate(delta_m.get('parse_success_rate'))} |\n"
    s32 += f"| schema_valid_rate | {_rate(base_m.get('schema_valid_rate'))} | {_rate(finetuned_m.get('schema_valid_rate'))} | {_rate(delta_m.get('schema_valid_rate'))} |\n"
    s32 += f"| required_field_presence_rate | {_rate(base_m.get('required_field_presence_rate'))} | {_rate(finetuned_m.get('required_field_presence_rate'))} | {_rate(delta_m.get('required_field_presence_rate'))} |\n"
    s32 += f"| uncertainty_marker_presence_rate | {_rate(base_m.get('uncertainty_marker_presence_rate'))} | {_rate(finetuned_m.get('uncertainty_marker_presence_rate'))} | {_rate(delta_m.get('uncertainty_marker_presence_rate'))} |\n"
    s32 += f"| avg_latency_ms | {_rate(base_m.get('avg_latency_ms'))} | {_rate(finetuned_m.get('avg_latency_ms'))} | {_rate(delta_m.get('avg_latency_ms'))} |\n\n"

    return "## 3. Quantitative Metrics\n\n" + s31 + s32


def _build_qualitative_examples(
    base_records: list[dict],
    finetuned_records: list[dict],
) -> str:
    # Index by prompt_id for stable ordering
    base_by_id = {r["prompt_id"]: r for r in base_records}
    finetuned_by_id = {r["prompt_id"]: r for r in finetuned_records}
    all_ids = sorted(set(base_by_id) & set(finetuned_by_id))

    finetuned_ok_base_fail = [
        pid
        for pid in all_ids
        if finetuned_by_id[pid].get("schema_valid") or finetuned_by_id[pid].get("parse_success")
        and not (base_by_id[pid].get("schema_valid") or base_by_id[pid].get("parse_success"))
    ]
    both_fail = [
        pid
        for pid in all_ids
        if not (base_by_id[pid].get("parse_success")) and not (finetuned_by_id[pid].get("parse_success"))
    ]

    # Deterministic: take first up to 5 and first up to 3
    examples_improved = finetuned_ok_base_fail[:5]
    examples_hard = both_fail[:3]

    lines = ["## 4. Qualitative Examples\n"]

    def _result_line(rec: dict) -> str:
        if rec.get("schema_valid"):
            return "success"
        code = rec.get("error_code") or "—"
        return f"failure ({code})"

    for pid in examples_improved:
        br = base_by_id[pid]
        fr = finetuned_by_id[pid]
        prompt_short = _escape_md(br.get("prompt", ""), 200)
        lines.append(f"### Improved: {pid}")
        lines.append(f"- **Prompt (shortened):** {prompt_short}")
        lines.append(f"- **Base:** {_result_line(br)}")
        lines.append(f"- **Finetuned:** {_result_line(fr)}")
        lines.append("")

    for pid in examples_hard:
        br = base_by_id[pid]
        fr = finetuned_by_id[pid]
        prompt_short = _escape_md(br.get("prompt", ""), 200)
        lines.append(f"### Hard case: {pid}")
        lines.append(f"- **Prompt (shortened):** {prompt_short}")
        lines.append(f"- **Base:** {_result_line(br)}")
        lines.append(f"- **Finetuned:** {_result_line(fr)}")
        lines.append("")

    return "\n".join(lines)


def _build_failure_taxonomy(finetuned_records: list[dict]) -> str:
    invalid_json = 0
    schema_mismatch = 0
    missing_verification_steps = 0
    overconfident_language = 0

    for r in finetuned_records:
        ec = r.get("error_code") or ""
        if ec in (JSON_STRICT_PARSE_FAILED, JSON_LENIENT_PARSE_FAILED):
            invalid_json += 1
        elif ec in (ENVELOPE_SCHEMA_FAILED, PAYLOAD_SCHEMA_FAILED):
            schema_mismatch += 1
        missing = r.get("missing_fields") or []
        if "verification_steps" in missing:
            missing_verification_steps += 1
        # Overconfident: not inferred from current artifacts; leave as 0

    lines = [
        "## 5. Failure Taxonomy (v1)",
        "",
        "| Failure type | Count |",
        "|--------------|-------|",
        f"| invalid JSON | {invalid_json} |",
        f"| schema mismatch | {schema_mismatch} |",
        f"| missing verification steps | {missing_verification_steps} |",
        f"| overconfident language | {overconfident_language} |",
        "",
    ]
    return "\n".join(lines)


def _build_notes(cfg: dict, meta: dict, comparison: dict) -> str:
    data = cfg.get("data", {})
    train_path = data.get("train_path", "")
    val_path = data.get("val_path", "")
    num_prompts = comparison.get("num_prompts", 0)
    return f"""## 6. Notes & Limitations

- **Dataset:** train path `{train_path}`, val path `{val_path}` (dataset_v1_clean).
- **Evaluation scope:** Fixed prompt suite only ({num_prompts} prompts).
- **Known limitations:** Usefulness and factual quality not fully evaluated; report is derived only from parse/schema and heuristic metrics.
"""


def generate_report(run_dir: Path, out_path: Path) -> None:
    """Load all artifacts and write training_report_v1.md."""
    _require_run_dir(run_dir)

    cfg = _load_yaml(run_dir / "config.yaml")
    meta = _load_json(run_dir / "meta.json")
    train_metrics = _load_json(run_dir / "train_metrics.json")
    base_records = _load_jsonl(run_dir / "samples/base_outputs.jsonl")
    finetuned_records = _load_jsonl(run_dir / "samples/finetuned_outputs.jsonl")
    comparison = _load_json(run_dir / "samples/comparison_metrics.json")

    sections = [
        "# PocketGuide — Training Report v1\n",
        _build_overview(run_dir, meta, cfg, train_metrics),
        _build_config_table(cfg, meta),
        _build_quantitative_metrics(train_metrics, comparison),
        _build_qualitative_examples(base_records, finetuned_records),
        _build_failure_taxonomy(finetuned_records),
        _build_notes(cfg, meta, comparison),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training report v1 from a run directory (Lesson 5.5)."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Training run directory (e.g. runs/train/<run_id>)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="docs/training_report_v1.md",
        help="Output markdown path (default: docs/training_report_v1.md)",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_path = Path(args.out).resolve()

    try:
        generate_report(run_dir, out_path)
        print(f"Wrote {out_path}")
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
