# Critique Pass - Quality Gate (Milestone 3, Lesson 3.4)

**Status**: ✅ COMPLETED

## Overview

This lesson implements a quality gate pass that evaluates draft responses for training data quality. Each draft is critiqued by the teacher model against structured criteria including hallucination risk, schema compliance, actionability, and safety. The critique system provides structured feedback that feeds into the refinement pass (Lesson 3.5).

## Architecture

### Design Pattern: Structured Quality Assessment

The critique pipeline follows a **structured, schema-validated** approach:

1. **Load drafts** from JSONL with all metadata
2. **Build critique prompt** including original prompt, draft output, and contract validation results
3. **Call teacher model** to generate structured critique JSON
4. **Parse & validate critique** against critique schema
5. **Store critique record** with teacher provenance and validation results
6. **Aggregate statistics** on verdict distribution, issue types, and quality scores

### Key Design Decisions

- **Schema-validated critiques**: All valid critiques conform to critique.schema.json
- **Verdict-driven workflow**: Teacher decides pass/revise/reject (not hardcoded logic)
- **Failure tolerance**: Invalid critiques stored with error details, don't crash pipeline
- **Structured feedback**: Issues, scores, and rewrite instructions enable automated refinement
- **Full provenance**: Teacher metadata tracked for audit trail

## Implementation

### Core Components

#### 1. Critique Schema ([critique.schema.json](../../src/pocketguide/data/schemas/v1/critique.schema.json))

JSON Schema with strict validation for critique structure:

**Required Top-Level Fields:**
- `id`: Draft ID being critiqued
- `verdict`: "pass", "revise", or "reject"
- `issues`: Array of issue objects (can be empty)
- `scores`: Object with 1-5 ratings for actionability, clarity, schema_compliance, safety_risk
- `hallucination`: Risk assessment with level, risky claims, rationale
- `verification`: Missing verification steps assessment
- `schema`: Schema compliance evaluation
- `rewrite_instructions`: Actionable imperatives for refinement

**Issue Object:**
- `type`: Category (hallucination, overconfidence, schema, verification, actionability, clarity, safety)
- `severity`: "low", "medium", or "high"
- `message`: Human-readable description
- `evidence_path`: Optional JSON path to problematic content

#### 2. Critique Prompt Template ([critique.txt](../../data/prompts/teacher/v1/critique.txt))

Versioned prompt template that instructs the teacher to:
- Focus on hallucination risk (time-sensitive claims, unverifiable facts)
- Check schema compliance against contract results
- Evaluate actionability and clarity
- Identify missing verification steps
- Provide structured JSON output only (no prose)
- Use imperative rewrite instructions

#### 3. Critique Runner ([generate_critiques.py](../../src/pocketguide/data_generation/generate_critiques.py))

**Main Function:**
```python
def generate_critiques(
    drafts_path: Path,
    out_dir: Path,
    teacher: TeacherRouterClient,
    config: dict,
    schema: dict,
    template: str,
    limit: int = None,
    resume: bool = False,
) -> dict
```

**Features:**
- Streams drafts from JSONL (memory efficient)
- Resume support (skips already-critiqued IDs)
- Lenient parsing (tries markdown code fence extraction)
- Schema validation with jsonschema library
- Failure tolerance (stores invalid critiques with error details)
- Truncates huge raw text (max 20k chars)
- Writes manifest and stats files

**Helper Functions:**
- `load_critique_schema()`: Load critique JSON schema
- `load_drafts()`: Stream JSONL drafts
- `build_critique_prompt()`: Format prompt with draft data
- `build_critique_request()`: Create TeacherRequest
- `parse_critique_json()`: Strict + lenient JSON parsing
- `validate_critique_schema()`: Schema validation with jsonschema
- `build_critique_record()`: Create output record
- `write_manifest()`: Reproducibility metadata
- `write_stats()`: Aggregated quality metrics

#### 4. Output Schema

**Critique Record (JSONL line):**
```json
{
  "id": "draft_0",
  "draft_id": "draft_0",
  "critique": {
    "id": "draft_0",
    "verdict": "revise",
    "issues": [
      {
        "type": "hallucination",
        "severity": "medium",
        "message": "Claims specific opening hours without verification guidance",
        "evidence_path": "payload.trip_days[0].activities[1]"
      }
    ],
    "scores": {
      "actionability": 4,
      "clarity": 5,
      "schema_compliance": 5,
      "safety_risk": 4
    },
    "hallucination": {
      "risk_level": "medium",
      "risky_claims": ["Museum opens at 9:00 AM"],
      "rationale": "Time-sensitive information without verification step"
    },
    "verification": {
      "missing_when_needed": true,
      "suggested_steps": ["Verify museum hours before visit"]
    },
    "schema": {
      "envelope_ok": true,
      "payload_ok": true,
      "notes": "Schema compliant",
      "missing_fields": []
    },
    "rewrite_instructions": [
      "Add verification step for museum hours",
      "Qualify time-sensitive claims with 'typically' or 'as of [date]'"
    ]
  },
  "raw_critique_text": null,
  "teacher": {
    "provider": "openrouter",
    "selected_model": "meta-llama/llama-3.3-70b-instruct:free",
    "attempted_models": ["meta-llama/llama-3.3-70b-instruct:free"],
    "request_id": "req-abc123",
    "timing": {"latency_s": 2.1},
    "usage": {"total_tokens": 450}
  },
  "critique_contract": {
    "strict_json_ok": true,
    "schema_ok": true
  },
  "created_at": "2026-01-28T14:23:45.123456+00:00"
}
```

**For Invalid Critiques:**
```json
{
  "id": "draft_1",
  "draft_id": "draft_1",
  "critique": null,
  "raw_critique_text": "This is not valid JSON...",
  "teacher": {...},
  "critique_contract": {
    "strict_json_ok": false,
    "schema_ok": false,
    "error": {
      "code": "JSON_PARSE_ERROR",
      "message": "Could not extract valid JSON from response"
    }
  },
  "created_at": "2026-01-28T14:23:47.123456+00:00"
}
```

**Manifest (critiques_v1_manifest.json):**
```json
{
  "version": "v1",
  "run_id": "20260128_142345",
  "generated_at": "2026-01-28T14:23:45.123456+00:00",
  "inputs": {
    "drafts_path": "data/interim/drafts_v1.jsonl",
    "drafts_hash": "abc123..."
  },
  "critique_schema": {
    "path": "src/pocketguide/data/schemas/v1/critique.schema.json",
    "hash": "def456...",
    "version": "v1"
  },
  "teacher_config": {
    "models": ["meta-llama/llama-3.3-70b-instruct:free", "..."],
    "generation": {"temperature": 0.2, "max_tokens": 1500}
  },
  "counts": {
    "total_drafts": 100,
    "processed": 100,
    "skipped": 0,
    "critique_parse_ok": 98,
    "critique_schema_ok": 97
  },
  "outputs": {
    "critiques": "critiques_v1.jsonl",
    "manifest": "critiques_v1_manifest.json",
    "stats": "critiques_v1_stats.json"
  }
}
```

**Statistics (critiques_v1_stats.json):**
```json
{
  "total": 100,
  "processed": 100,
  "skipped": 0,
  "critique_parse_rate": 98.0,
  "critique_schema_valid_rate": 97.0,
  "verdict_distribution": {
    "pass": 45,
    "revise": 42,
    "reject": 10
  },
  "issue_type_counts": {
    "hallucination": 23,
    "verification": 18,
    "overconfidence": 12,
    "actionability": 8,
    "clarity": 5,
    "schema": 3,
    "safety": 1
  },
  "risk_level_distribution": {
    "low": 65,
    "medium": 28,
    "high": 4
  },
  "avg_scores": {
    "actionability": 4.2,
    "clarity": 4.5,
    "schema_compliance": 4.8,
    "safety_risk": 4.6
  }
}
```

## Usage

### CLI

**Generate critiques:**
```bash
python -m pocketguide.data_generation.generate_critiques \
  --drafts data/interim/drafts_v1.jsonl \
  --out_dir data/interim
```

**With options:**
```bash
python -m pocketguide.data_generation.generate_critiques \
  --drafts data/interim/drafts_v1.jsonl \
  --out_dir data/interim \
  --limit 10 \
  --resume \
  --dry_run
```

**Custom schema/template:**
```bash
python -m pocketguide.data_generation.generate_critiques \
  --drafts data/interim/drafts_v1.jsonl \
  --out_dir data/interim \
  --schema path/to/custom_critique.schema.json \
  --template path/to/custom_template.txt
```

### Makefile Targets

```bash
# Generate critiques (requires teacher.yaml config + drafts)
make critiques

# Dry-run without API calls
make critiques-dry-run
```

### Python API

```python
from pathlib import Path
from pocketguide.data_generation.generate_critiques import (
    create_teacher_router,
    generate_critiques,
    load_critique_schema,
    load_critique_prompt_template,
    load_teacher_config,
)

# Load dependencies
config = load_teacher_config(Path("configs/teacher.yaml"))
schema = load_critique_schema()
template = load_critique_prompt_template()
teacher = create_teacher_router(config)

# Generate critiques
stats = generate_critiques(
    drafts_path=Path("data/interim/drafts_v1.jsonl"),
    out_dir=Path("data/interim"),
    teacher=teacher,
    config=config,
    schema=schema,
    template=template,
    limit=50,
    resume=False,
)

print(f"Processed: {stats['processed']}")
print(f"Parse rate: {stats['critique_parse_rate']}%")
print(f"Verdicts: {stats['verdict_distribution']}")
```

## Testing

**Test Coverage** (16 tests in [test_generate_critiques.py](../../tests/test_generate_critiques.py)):

- `TestHashFile`: File hashing determinism
- `TestCritiqueSchema`: Schema loading, validation (valid/invalid cases, enum validation)
- `TestParseCritiqueJson`: JSON parsing (strict, markdown-wrapped, invalid)
- `TestBuildCritiquePrompt`: Prompt formatting
- `TestBuildCritiqueRequest`: Request construction
- `TestBuildCritiqueRecord`: Record building (valid/invalid critiques, text truncation)
- `TestGenerateCritiques`: Integration tests (mocked teacher, limit, resume)

**Run tests:**
```bash
pytest tests/test_generate_critiques.py -v
```

**All tests use mocks** (no network calls):
- TeacherRouterClient mocked to return controlled critiques
- First returns valid JSON, second returns invalid
- Validates record structure and stats computation

## Integration Points

### Dependencies

- **[pocketguide.teachers](../../src/pocketguide/teachers/)**: TeacherRouterClient for API calls
- **[jsonschema](https://pypi.org/project/jsonschema/)**: Schema validation library
- **[configs/teacher.yaml](../../configs/teacher.yaml)**: Teacher model configuration

### Inputs

- **data/interim/drafts_v1.jsonl**: Generated by Lesson 3.3

### Outputs Consumed By

- **Lesson 3.5 (Refinement pass)**: Uses critiques with verdict="revise" and rewrite_instructions
- **Quality analysis**: Stats file enables monitoring critique quality over time
- **Dataset filtering**: Can reject drafts with verdict="reject" or high-risk hallucination

## Key Features

### Schema-Validated Critiques
- Strict JSON schema with required fields and enum validation
- Validates all successful critiques before storage
- Provides actionable error messages on schema failures

### Failure Tolerance
- Invalid critiques stored with error details (don't crash pipeline)
- Raw critique text preserved for debugging
- Per-record contract tracking (similar to drafts)

### Structured Feedback
- Issues categorized by type and severity
- Numeric scores enable quantitative filtering
- Rewrite instructions in imperative mood for automation

### Quality Metrics
- Parse rate and schema validation rate
- Verdict distribution (pass/revise/reject ratios)
- Issue type frequency analysis
- Average quality scores across dimensions

### Reproducibility
- Deterministic hashing of inputs and schema
- Full config snapshot in manifest
- Run ID and timestamps for tracing

## Critique Prompt Design

The critique prompt template emphasizes:

1. **Hallucination Detection**: Focus on time-sensitive claims, specific prices/dates/schedules that can't be verified
2. **Verification Gaps**: Identify when verification guidance is needed but missing
3. **Schema Compliance**: Reference draft contract results to identify structural issues
4. **Actionable Feedback**: Rewrite instructions use imperative verbs ("Remove", "Add", "Qualify")
5. **Structured Output**: JSON-only response (no prose) for reliable parsing

## How This Feeds Lesson 3.5 (Refinement)

The critique pass enables automated refinement by:

1. **Filtering**: Only drafts with verdict="revise" need refinement
2. **Prioritization**: High-severity issues or low scores indicate urgent fixes
3. **Guided Rewriting**: Rewrite instructions provide specific, actionable changes
4. **Quality Gates**: drafts with verdict="pass" can skip refinement
5. **Rejection**: drafts with verdict="reject" can be filtered from final dataset

**Refinement Workflow (Lesson 3.5):**
```
critiques_v1.jsonl (verdict="revise")
  → Load draft + critique
  → Build refinement prompt with rewrite_instructions
  → Call teacher to rewrite
  → Validate refined output
  → Store as refined_v1.jsonl
```

## Files

| File | Purpose | Lines |
|------|---------|-------|
| [critique.schema.json](../../src/pocketguide/data/schemas/v1/critique.schema.json) | Critique JSON schema | 130 |
| [critique.txt](../../data/prompts/teacher/v1/critique.txt) | Prompt template | 75 |
| [generate_critiques.py](../../src/pocketguide/data_generation/generate_critiques.py) | Main implementation | 710 |
| [test_generate_critiques.py](../../tests/test_generate_critiques.py) | Test suite | 556 |

## Test Results

✅ All 185 tests passing:
- 16 tests in `test_generate_critiques.py` (new)
- 169 tests in other modules (no regressions)

## Next Steps

With Lesson 3.4 complete, the quality gate is operational. Next lesson:

**Lesson 3.5: Refinement Pass**
- Load critiques with verdict="revise"
- Build refinement prompts with rewrite_instructions
- Call teacher to rewrite drafts
- Validate refined outputs
- Track refinement success rate

## Summary

Lesson 3.4 implements a production-ready critique system that:
- **Evaluates quality** with structured, schema-validated critiques
- **Identifies issues** across hallucination, verification, schema, actionability, clarity, safety
- **Provides feedback** with numeric scores and actionable rewrite instructions
- **Tolerates failures** without crashing (stores error details)
- **Tracks metrics** for quality monitoring
- **Enables refinement** with verdict-driven workflow and structured feedback

The system bridges draft generation and refinement, providing the structured feedback needed to improve low-quality drafts systematically.
