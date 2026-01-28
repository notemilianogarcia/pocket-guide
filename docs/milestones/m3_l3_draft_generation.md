# Draft Generation Pass (Milestone 3, Lesson 3.3)

**Status**: ✅ COMPLETED

## Overview

This lesson implements the batch draft generation pass, which converts prompt plans into synthetic training data by calling the teacher model with schema-first validation. The system ensures every generated response conforms to the expected schema before storing, and tracks provenance and validation metadata for full transparency.

## Architecture

### Design Pattern: Schema-First Validation

The draft generation pipeline follows a **deterministic, schema-first** approach:

1. **Load prompt plan** from JSONL with required fields validation
2. **Build teacher request** from prompt record with generation config
3. **Call teacher model** (once per prompt, no retries on generation failure)
4. **Parse & validate response** against envelope + payload schema
5. **Build draft record** with prompt metadata, teacher provenance, validation results
6. **Write to JSONL** (append-only, one record per line)
7. **Generate manifest & stats** for reproducibility and quality tracking

### Key Design Decisions

- **Append-only JSONL output**: Enables resuming from failure mid-run
- **No generation retries**: Teacher is called exactly once per prompt; failures are recorded
- **Per-sample provenance**: Every draft records which teacher model generated it, with timing and token usage
- **Pass-rate tracking**: Statistics computed across all samples by type, category, difficulty, and error code
- **Deterministic hashing**: Prompt plan hash in manifest enables verifying data reproducibility

## Implementation

### Core Components

#### 1. Main Module ([generate_drafts.py](../../src/pocketguide/data_generation/generate_drafts.py))

**Helper Functions:**

- `hash_file(path)`: SHA256 hash of file for reproducibility verification
- `load_prompt_plan(path)`: Parse JSONL, validate required fields
- `load_teacher_config(path)`: Load teacher.yaml with defaults
- `create_teacher_router(config)`: Instantiate `TeacherRouterClient` from config
- `build_teacher_request(record, config)`: Create `TeacherRequest` with system + user messages
- `build_draft_record_from_parse(prompt_record, response, parse_result)`: Convert `ParseResult` to output record with validation metadata
- `compute_pass_rates(stats)`: Calculate percentage pass rates
- `write_manifest(out_dir, plan_path, config, stats, run_id)`: Write reproducibility metadata
- `write_stats(out_dir, stats)`: Write aggregated statistics

**Main Function:**

```python
def generate_drafts(
    plan_path: Path,
    out_dir: Path,
    teacher: TeacherRouterClient,
    config: dict,
    limit: int = None,
    resume: bool = False,
) -> dict
```

- Loads prompt plan and existing IDs (for resume support)
- Processes up to `limit` prompts (None = all)
- Skips already-generated IDs when resuming
- Calls teacher exactly once per prompt
- Validates with `parse_and_validate()` from `pocketguide.eval.parsing`
- Writes records to `drafts_v1.jsonl` in output directory
- Writes manifest and stats files at completion
- Returns aggregated statistics dict

#### 2. Output Schema

**Draft Record (JSONL line)**:
```json
{
  "id": "prompt_0",
  "prompt_plan": {
    "payload_type": "itinerary",
    "category": "planning",
    "difficulty": "medium",
    "region_tags": ["europe", "france"],
    "template_version": "v1",
    "template_name": "itinerary_draft",
    "seed": 42
  },
  "prompt": "Plan a 3-day trip to Paris...",
  "teacher": {
    "provider": "openrouter",
    "selected_model": "meta-llama/llama-3.3-70b-instruct:free",
    "attempted_models": ["meta-llama/llama-3.3-70b-instruct:free"],
    "request_id": "req-abc123",
    "timing": {"latency_s": 2.34},
    "usage": {
      "prompt_tokens": 150,
      "completion_tokens": 280,
      "total_tokens": 430
    }
  },
  "output_text": "{\"summary\": \"...\"...}",
  "contract": {
    "overall_ok": true,
    "strict_json_ok": true,
    "lenient_json_ok": true,
    "envelope_ok": true,
    "payload_ok": true,
    "payload_type": "itinerary"
  },
  "created_at": "2025-04-15T10:23:45.123456+00:00"
}
```

**Manifest** (`drafts_v1_manifest.json`):
```json
{
  "version": "v1",
  "run_id": "20250415_102345",
  "generated_at": "2025-04-15T10:23:45.123456+00:00",
  "spec": {
    "plan_path": "data/prompt_plan_v1.jsonl",
    "plan_hash": "abc123def456..."
  },
  "teacher_config": {
    "models": ["meta-llama/llama-3.3-70b-instruct:free", "..."],
    "generation": {"temperature": 0.2, "max_tokens": 900, "...": "..."},
    "rate_limit": {"rpm": 15}
  },
  "schema_versions": {
    "envelope": "v0",
    "payload": "v1"
  },
  "counts": {
    "total_planned": 100,
    "processed": 100,
    "skipped": 0
  },
  "outputs": {
    "drafts": "drafts_v1.jsonl",
    "manifest": "drafts_v1_manifest.json",
    "stats": "drafts_v1_stats.json"
  }
}
```

**Statistics** (`drafts_v1_stats.json`):
```json
{
  "total": 100,
  "processed": 100,
  "skipped": 0,
  "pass_rates": {
    "strict_json_rate": 98.5,
    "lenient_json_rate": 99.2,
    "envelope_rate": 97.8,
    "payload_rate": 96.3,
    "overall_rate": 96.3
  },
  "by_payload_type": {
    "itinerary": 35,
    "review": 42,
    "summary": 23
  },
  "by_category": {
    "planning": 50,
    "evaluation": 30,
    "creation": 20
  },
  "by_difficulty": {
    "easy": 30,
    "medium": 45,
    "hard": 25
  },
  "top_errors": {
    "JSON_PARSE_ERROR": 3,
    "ENVELOPE_VALIDATION_ERROR": 1
  }
}
```

## Usage

### CLI

**Generate drafts from prompt plan:**
```bash
python -m pocketguide.data_generation.generate_drafts \
  --plan data/prompt_plan_v1.jsonl \
  --out_dir data/drafts \
  --config configs/teacher.yaml
```

**Dry-run mode (no API calls):**
```bash
python -m pocketguide.data_generation.generate_drafts \
  --plan data/prompt_plan_v1.jsonl \
  --out_dir data/drafts \
  --config configs/teacher.yaml \
  --dry_run
```

**With options:**
```bash
python -m pocketguide.data_generation.generate_drafts \
  --plan data/prompt_plan_v1.jsonl \
  --out_dir data/drafts \
  --config configs/teacher.yaml \
  --limit 10           # Process only first 10 prompts
  --resume             # Skip already-generated IDs
  --dry_run            # Override config dry_run setting
```

### Makefile Targets

```bash
# Generate drafts (requires teacher.yaml config)
make drafts

# Dry-run without API calls
make drafts-dry-run
```

### Python API

```python
from pathlib import Path
from pocketguide.data_generation.generate_drafts import (
    create_teacher_router,
    generate_drafts,
    load_teacher_config,
)

# Load config and create teacher
config = load_teacher_config(Path("configs/teacher.yaml"))
teacher = create_teacher_router(config)

# Generate drafts
stats = generate_drafts(
    plan_path=Path("data/prompt_plan_v1.jsonl"),
    out_dir=Path("data/drafts"),
    teacher=teacher,
    config=config,
    limit=100,      # Optional: process only first 100
    resume=False,   # Optional: continue from last processed
)

# Check results
print(f"Processed: {stats['processed']}")
print(f"Overall pass rate: {stats['overall_pass']} / {stats['processed']}")
```

## Testing

**Test Coverage** (13 tests in [test_generate_drafts.py](../../tests/test_generate_drafts.py)):

- `TestHashFile`: File hashing determinism
- `TestLoadPromptPlan`: JSONL validation (valid plan, missing file, missing field, blank lines)
- `TestBuildTeacherRequest`: Request construction with generation config
- `TestBuildDraftRecord`: Record building with success and error cases
- `TestComputePassRates`: Pass rate percentage calculations
- `TestGenerateDrafts`: Integration tests (dry-run, limit, resume)

**Run tests:**
```bash
pytest tests/test_generate_drafts.py -v
```

**All tests use mocks** (no network calls or real API usage):
- `TeacherRouterClient` mocked to return controlled responses
- ParseResult objects constructed directly
- Temporary files used for I/O testing

## Integration Points

### Dependencies

- **[pocketguide.eval.parsing](../../src/pocketguide/eval/parsing.py)**: `parse_and_validate()` and `ParseResult`
- **[pocketguide.teachers.base](../../src/pocketguide/teachers/base.py)**: `TeacherRouterClient`, `TeacherRequest`, `TeacherResponse`
- **[configs/teacher.yaml](../../configs/teacher.yaml)**: Teacher model and generation configuration

### Outputs Consumed By

- Evaluation benchmarks: Use `drafts_v1.jsonl` as synthetic training data
- Quality analysis: Analyze `drafts_v1_stats.json` for generation quality
- Reproducibility: Use `drafts_v1_manifest.json` to verify data generation

## Key Features

### Reproducibility
- Deterministic prompt plan hashing enables verifying unchanged inputs
- Full config snapshot in manifest for auditing parameter choices
- Run ID and timestamp for tracing generation sessions

### Resumable Generation
- Append-only JSONL output with optional `--resume` flag
- Existing IDs tracked in memory during run
- Safe to interrupt and restart without data loss

### Comprehensive Validation
- Schema-first approach: validates against envelope + payload
- Per-sample validation results stored in contract section
- Pass-rate statistics at multiple granularities (type, category, difficulty, error)

### Transparent Provenance
- Teacher metadata recorded: provider, model, request ID, timing, token usage
- Validation outcomes stored with each sample
- Error tracking by code and type for quality analysis

## Files

| File | Purpose |
|------|---------|
| [src/pocketguide/data_generation/generate_drafts.py](../../src/pocketguide/data_generation/generate_drafts.py) | Main implementation (583 lines) |
| [tests/test_generate_drafts.py](../../tests/test_generate_drafts.py) | Test suite (469 lines, 13 tests) |
| [configs/teacher.yaml](../../configs/teacher.yaml) | Teacher config (referenced, not created by this lesson) |

## Test Results

✅ All 169 tests passing:
- 13 tests in `test_generate_drafts.py`
- 156+ tests in other modules (no regressions)

## Next Steps

With Lesson 3.3 complete, the synthetic data generation pipeline is fully operational. Next lessons can focus on:

1. **Lesson 3.4**: Evaluation metrics and quality analysis on generated drafts
2. **Lesson 3.5**: Fine-tuning workflows using generated data
3. **Advanced**: Iterative refinement loops with teacher model feedback

## Summary

Lesson 3.3 implements a production-ready draft generation system that:
- **Respects schema constraints** at every step (schema-first validation)
- **Tracks provenance** completely (teacher metadata + validation results)
- **Enables reproducibility** (deterministic hashing, config snapshots)
- **Supports workflows** (dry-run, resumable, limiting for testing)
- **Provides transparency** (detailed statistics and error tracking)

The system bridges the gap between prompt planning and evaluation, converting validated prompts into high-quality synthetic training data ready for downstream tasks.
