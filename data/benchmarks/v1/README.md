# Held-Out Benchmark v1

This directory contains held-out test prompts for evaluating travel guide generation models.

## Overview

- **Total prompts:** 23
- **Generation date:** 2026-01-31T16:10:39.158979
- **Random seed:** 42
- **Train/Val/Test fractions:** 80.0% / 10.0% / 10.0%
- **Similarity threshold:** 0.75

## Files

- **prompts_test.jsonl**: Test prompts without responses (for evaluation)

## Leakage Guardrails

All prompts in this benchmark are:
- Held out from training data (test split only)
- Protected from near-duplicate leakage via group-based splitting
- Validated to have no exact duplicates in train/val splits

## Usage

To evaluate a model on this benchmark:

```bash
python -m pocketguide.eval.run_benchmark \
  --prompts data/benchmarks/v1/prompts_test.jsonl \
  --output results/benchmark_v1_results.jsonl
```

Each prompt includes:
- `id`: Unique identifier
- `payload_type`: Expected response format (checklist, guide, etc.)
- `category`: Topic category (visa, passport, etc.)
- `difficulty`: Complexity level (easy, medium, hard)
- `region_tags`: Relevant geographic regions
- `prompt`: The question to answer

Responses should be generated in the format specified by `payload_type`.
