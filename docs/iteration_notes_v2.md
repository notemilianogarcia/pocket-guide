# Iteration Notes v2 — Lesson 7.1: Failure Analysis & Hypothesis Selection

## Data Audit Summary

**Local eval runs inspected**

- **Runs under `runs/eval/` with `local_metrics.json`:** `20260201_221911`, `20260201_221952`.
- **Most recent run:** `20260201_221952`.

**Metrics (run 20260201_221952)**

- **Suites:** `fixed20_v1` (n=20), `local_regression_v1` (n=15); overall n=35.
- **Parse / schema rates:** `strict_parse_rate` 1.0, `lenient_parse_rate` 1.0, `envelope_valid_rate` 1.0, `payload_valid_rate` 1.0.
- **Latency / throughput:** `avg_latency_ms` 0.0, `avg_tokens_per_sec_approx` 0.0.

**Per-sample outputs (`local_outputs.jsonl`)**

- All 35 records have `error_type: null`, `strict_parse_ok: true`, `lenient_parse_ok: true`, `envelope_valid: true`, `payload_valid: true`.
- No `json_parse`, `envelope_schema`, or `payload_schema` failures recorded.

**Conclusion**

Current local eval data reflects **stub runtime only** (no real model inference). There are no observed failure modes in the metrics or per-sample outputs. Failure-mode selection for v2 is therefore based on:

1. The **failure taxonomy** already used in Milestone 6 (`report_base_v0`: JSON format, schema/required-field, hallucination, constraints, overconfidence, verification guidance, safety).
2. The **error types** implemented in the pipeline: `json_parse`, `envelope_schema`, `payload_schema` (and `runtime_error`).
3. **Known failure patterns** for quantized and local LLMs: truncated or malformed JSON; missing “optional-looking” but required envelope/payload fields.

---

## Selected Failure Modes

**Failure #1 — JSON format violations (truncation / malformed JSON)**

- **Description:** Output is not valid JSON or cannot be recovered by lenient parsing (e.g., truncated before closing braces, stray markdown, trailing commas).
- **Evidence:** Taxonomy category 1 in M6; pipeline error type `json_parse` (strict and lenient parse failure). Quantized and smaller local models frequently truncate or emit invalid JSON under context or token limits.
- **Impact:** High severity (downstream parsing and schema checks never run); frequency is typically high when running real quantized/local inference.

**Failure #2 — Schema / required-field failures (missing envelope or payload fields)**

- **Description:** Output is valid JSON but fails envelope or payload schema due to missing required fields (e.g., `verification_steps`, `summary`, or payload-type–specific keys).
- **Evidence:** Taxonomy category 2 in M6; pipeline error types `envelope_schema` and `payload_schema`. Smaller or quantized models often omit fields that look optional but are required by our schemas.
- **Impact:** High severity (envelope/payload validity fails; structured use of the response is blocked); commonly observed when required-field lists are long or nested.

---

## Hypotheses

**Failure #1 (JSON format)**

- **Why it happens:** Quantized models have reduced capacity and are more prone to stopping mid-generation or emitting invalid syntax (e.g., unterminated strings, missing brackets). Context limits and sampling can increase truncation.
- **Why it appears after quantization / local runtime:** Quantization and smaller context budgets amplify truncation and syntax errors; the same model in full-precision HF form typically has higher valid-JSON rate.
- **Why it was not fully fixed by training v1:** v1 training targeted structure and content (envelope, payload types); it did not explicitly target robust JSON boundaries, length control, or “always close JSON” behavior under token limits.
- **Proposed intervention (one, data-centric):** **Dataset augmentation for v2:** Add or oversample examples that (a) end with well-formed JSON (no trailing content), and (b) include short, strict-JSON-only responses near the intended max length, so the model learns to produce complete, valid JSON under constraints. Optionally tighten QC to reject training examples that are truncated or fail strict parse.

**Failure #2 (Schema / required-field)**

- **Why it happens:** The model omits fields it treats as optional (e.g., `verification_steps`, or payload fields like `priority`). Training data may under-represent examples where every required field is present.
- **Why it appears after quantization / local runtime:** Capacity reduction makes it harder to satisfy all required keys; local/quantized models tend to drop “secondary” fields first.
- **Why it was not fully fixed by training v1:** v1 data and balancing may not have emphasized full envelope + payload completeness; some required fields may be rare in the current dataset.
- **Proposed intervention (one, data-centric):** **Stricter QC and optional oversampling for v2:** (a) In dataset_v2, enforce a QC rule that rejects any example missing a required envelope or payload field. (b) Optionally oversample or augment with examples that explicitly include all required fields (especially `verification_steps` and payload-required keys) so the model sees them consistently.

---

## What Stays Fixed

To isolate the effect of v2 interventions, the following are held constant:

- **Benchmark:** Same suites — `eval/suites/fixed20_v1.jsonl` and `eval/suites/local_regression_v1.jsonl`.
- **Runtime:** Same local runtime (llama.cpp, same config and binary path).
- **Evaluation scripts and metrics:** Same `local_eval.py` and aggregation; same metrics (`strict_parse_rate`, `lenient_parse_rate`, `envelope_valid_rate`, `payload_valid_rate`, latency, tokens/sec approx).
- **Parsing and schemas:** Same `parse_and_validate` and schema set (envelope + itinerary, checklist, decision_tree, procedure).

No changes to benchmark prompts, runtime choice, or evaluation logic are introduced in this iteration. Lesson 7.2 will implement the chosen interventions (dataset augmentation and/or QC for v2) and retrain; re-evaluation will use the same setup above to measure delta.
