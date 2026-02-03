# Schema failures and v1 vs v2 comparison

## Is it our validation or the model?

Validation is **correct**: the envelope schema (`v0/response_envelope.schema.json`) requires a single JSON **object** with exactly these root-level fields:

- `summary`, `assumptions`, `uncertainty_notes`, `next_steps`, `verification_steps`, `payload_type`, `payload`

When a sample is rejected, it’s because the **model output** doesn’t match this contract.

### Two common failure modes

1. **Missing envelope fields (e.g. `verification_steps`, `payload_type`)**
   - **Example:** `parse_ok=True`, `schema_ok=False`, `missing_fields: ["verification_steps", "payload_type"]`, `parsed_top_level_keys: ["assumptions", "next_steps", "payload", "summary", "uncertainty_notes"]`
   - **Meaning:** The model produced valid JSON and included most envelope keys but **omitted** required fields. The prompts explicitly ask for all envelope fields (including `verification_steps` and `payload_type`). This is **model behavior**, not a validation bug.

2. **Root is an array, not an object**
   - **Example:** `parsed_top_level_keys: null`, `schema_error_message: "['Flexible itinerary...'] is not of type 'object'"`
   - **Meaning:** The first (or only) JSON we could extract was an **array** (e.g. a one-element list). The envelope must be a single **object**. This can happen when the model outputs a list/summary first and the real envelope object appears later or not at all. Lenient parsing now **prefers a JSON object over an array** when both exist, so this case should be rarer when the model does output the envelope. If you still see it, the model is either not outputting the envelope object or putting it in a form we don’t parse (e.g. truncated, malformed).

### Summary

- **Validation:** Correct and aligned with the schema and prompts.
- **Rejections:** Due to model output (missing fields or wrong root type), not validator errors.

---

## Comparing v1 base, v1 finetuned, v2 base, v2 finetuned

`run_samples` always runs **base** (no adapter) and **finetuned** (with adapter) for the **same** run dir. So:

- For **v1:** use `--run_dir runs/train/<v1_run_id>` (e.g. `20260131_224510`).  
  Outputs: base vs finetuned for that v1 run.
- For **v2:** use `--run_dir runs/train/<v2_run_id>` (e.g. `20260203_013228-v2`).  
  Outputs: base vs finetuned for that v2 run.

**v1 base vs v1 finetuned:**  
Compare `runs/train/<v1_run_id>/samples/comparison_metrics.json`: `base` vs `finetuned` (and `delta`). Same prompts; base = no adapter, finetuned = v1 adapter.

**v2 base vs v2 finetuned:**  
Same idea: `runs/train/<v2_run_id>/samples/comparison_metrics.json`. Base = same base model, no adapter; finetuned = v2 adapter.

**v2 finetuned vs v1 finetuned:**  
Compare the **finetuned** sections (and optionally `finetuned_outputs.jsonl`) from the two run dirs. Same prompts; different adapters (v1 vs v2). No single script; just diff the two `comparison_metrics.json` files and/or the two `finetuned_outputs.jsonl` files.

**v2 base vs v1 base:**  
Base model is the same for both runs; so v1 base and v2 base outputs should be identical for the same prompts (same seed). No need to compare unless you changed base model between runs.

### Suggested workflow

1. Run `run_samples` for v1 run dir → get v1 base + v1 finetuned metrics.
2. Run `run_samples` for v2 run dir → get v2 base + v2 finetuned metrics.
3. Compare v1 finetuned vs v2 finetuned (e.g. `schema_valid_rate`, `parse_success_rate`) to see the effect of the v2 training change (e.g. LR tweak / dataset v2).

Debug failures using `samples/schema_failures_debug.jsonl` and the `[schema fail]` lines in the run_samples log.
