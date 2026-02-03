# Why schema_valid_rate Is 0%: Evidence-Based Analysis

This document explains **why every finetuned run (v1–v4) got 0% schema validity**, using the actual validation logic, schemas, and sample outputs. It is not guesswork: it ties each failure mode to code and to concrete examples from `runs/train/20260203_053120-v4/samples/`.

---

## 1. Did our validation fail? Is it too strict?

**No.** Validation is correct and aligned with the intended contract.

- **Envelope** (`v0/response_envelope.schema.json`): The root must be a **single JSON object** with exactly seven required keys: `summary`, `assumptions`, `uncertainty_notes`, `next_steps`, `verification_steps`, `payload_type`, `payload`. This is the design.
- **Payloads**: Each `payload_type` (itinerary, checklist, decision_tree, procedure) has its own schema. For example:
  - **Itinerary**: `payload` must have `title`, `trip_days`; each activity item must have `time_block` and `activity` (see `v1/itinerary.payload.schema.json`).
  - **Checklist**: `payload` must have `title` and `groups` (see `v1/checklist.payload.schema.json`).
  - **Decision tree**: `payload` must have `title` and `nodes` (see `v1/decision_tree.payload.schema.json`).

The evaluator uses `parse_and_validate()` in `src/pocketguide/eval/parsing.py`: strict parse first, then lenient parse; then envelope schema; then payload schema. No bug was found in this pipeline. Failures are due to **model output** not satisfying these contracts, not to overly strict or wrong validation.

---

## 2. What actually fails: two failure modes

From `runs/train/20260203_053120-v4/samples/schema_failures_debug.jsonl`, every finetuned sample fails in one of two ways:

| Error code               | schema_error_message                                      | Count (v4, n=20) |
|--------------------------|-----------------------------------------------------------|-------------------|
| `ENVELOPE_SCHEMA_FAILED`  | Top-level JSON must be an object (envelope), got list    | **15**            |
| `PAYLOAD_SCHEMA_FAILED`  | 'time_block' is a required property / 'title' is required | **5**             |

So the two concrete failure modes are:

1. **“Got list”** — the parser returns a **list/array** as the top-level JSON instead of an object.
2. **Payload schema** — the top-level is a valid envelope object, but the **payload** fails its type-specific schema (missing `time_block` in itinerary items, or missing `title` in checklist/decision_tree payloads).

Below we explain each with **evidence from real outputs and code**.

---

## 3. Failure mode 1: “Top-level JSON must be an object (envelope), got list”

### What the parser does

Lenient parsing in `parsing.py` (`_parse_json_lenient`) does the following:

1. Tries to find JSON inside markdown code fences; if it finds a **dict**, it returns it immediately.
2. If no dict was found in fences, it scans the text for the **first** `{` and the **first** `[`.
3. For `{`: it bracket-matches from that `{` to the matching `}`. If that substring is valid JSON and is a dict, it returns it.
4. If the object is **not closed** (e.g. output truncated before the final `}`), bracket-matching never gets a complete object, so **no dict is added**.
5. For `[`: it bracket-matches from the first `[` to the matching `]`. That often corresponds to the **first array inside the object** (e.g. `"assumptions": [...]`). That array **is** complete, so it is added to candidates.
6. If the only complete JSON structure found is that inner array, the function returns it. So the “top-level” parsed value is a **list**, and envelope validation correctly fails with “got list”.

So **“got list” does not mean the model output a root-level array**. It means the **only complete JSON** the parser could extract was an **inner array**, because the **envelope object was never closed** (truncation).

### Evidence from a real sample

From `finetuned_outputs.jsonl` (v4), prompt_id `v1_itinerary_entry_requirements_0051`:

- **raw_text_output** starts with: `-{"assumptions":["Travel during typical Italian spring break..."], ...` and contains a long envelope (summary, payload with trip_days, etc.). The stored output **ends mid-string**: `"time_block\":\"09:0` — i.e. the model’s completion was **cut off** before the closing `}` of the envelope.
- **parsed_json** recorded for this sample is:  
  `["Travel during typical Italian spring break season (March-April) with mild weather.", "Frequent flyer with US passport...", ...]`  
  i.e. the **assumptions array** (the first complete `[...]` in the text).
- So: the model **did** output an envelope object, but the response was **truncated**. The parser found the first complete JSON = the inner `assumptions` array → returns list → envelope check fails with “got list”.

Another example: `v1_itinerary_local_customs_0011`. The raw output starts with `JSON\n{"assumptions":[...]` and is again long; the parser’s result is a single-element array (one string). So again the first complete JSON is an inner array, and the envelope object is incomplete (truncated).

### Conclusion for “got list”

- **Validation is correct**: we require a root-level object; we got a list.
- **Root cause**: **Truncation**. The model often produces the envelope object but the completion is cut before the final `}`. The only well-formed JSON we can extract is an inner array (e.g. `assumptions` or `next_steps`), so we report “got list”.
- **Not**: “model chose to output a list” or “parser is wrong.”

---

## 4. Failure mode 2: Payload schema (time_block, title)

When the envelope **is** fully parsed (object with all seven keys), we then validate `payload` against the schema for `payload_type`. Here we see two kinds of payload failures.

### 4a. Itinerary: `'time_block' is a required property`

Schema: each activity in `payload.trip_days[].items[]` must have **`time_block`** and **`activity`** (see `itinerary.payload.schema.json`).

From `finetuned_outputs.jsonl`, prompt_id `v1_itinerary_entry_requirements_0025` (Bangkok itinerary):

- Envelope parses correctly (all seven keys present).
- In **payload**, some activity items use **`"time": "12:00-13:00"`** instead of **`"time_block"`**. For example:
  - `{"activity":"Light lunch of Thai classics at hotel's rooftop bar.","time":"12:00-13:00"}`
  - `{"activity":"Siesta and optional spa treatment...","time":"13:00-16:00"}`
- The schema requires `time_block`; the model emitted `time`. So validation fails with `'time_block' is a required property`.

So we **did** tell the model the right format (system instruction and SFT say “every activity item must have \"time_block\""), and we normalized training data to use `time_block` only. At **inference** the model still sometimes uses `time`. So the model did not fully learn to use only the `time_block` key.

### 4b. Checklist / decision_tree: `'title' is a required property`

- **Checklist** payload must have `title` and `groups`.
- **Decision tree** payload must have `title` and `nodes`.

From the same v4 samples, prompt_id `v2_hard_missing_verification_steps_0031` (decision tree for Brazil safety):

- Envelope parses correctly.
- **Payload** looks like:  
  `{"common":["Level of risk by destination..."],"specific":["Verify current conditions..."], ...}`  
  i.e. the model used a **different shape** (`common`, `specific`, `root`, etc.), not the schema’s **`title` + `nodes`**.
- So validation fails with `'title' is a required property` (and the structure is wrong for `nodes`).

So for **decision_tree**, the model is emitting a structure that **does not match** our `decision_tree.payload.schema.json`. That can mean either:

- Training/teacher data for decision_tree used a different format (e.g. `common`/`specific`/`root`) that was never aligned to the strict `title` + `nodes` schema, or  
- The model generalized to that format and we never trained it on enough schema-compliant decision_tree examples.

Either way, the **schema is not too strict**; the **payload shape** in the output is wrong.

---

## 5. Did we correctly tell the finetuned model the format?

**Yes, for the envelope and for itinerary.**

- **System instruction** (in `prepare_sft_data.py`): We say “Output a single JSON object (not a list or array)”, list all envelope keys, and “For itinerary payloads, every activity item must have \"time_block\"”.
- **SFT data**: Envelope is normalized (all seven keys); itinerary payloads are normalized so activity items have `time_block` and not `time_buffer`.

So the model was **instructed** and **trained** on the right envelope and on itinerary with `time_block`. Despite that:

- **Truncation** still prevents the envelope from being parsed as an object in many cases (“got list”).
- At inference the model **still sometimes uses `time` instead of `time_block`** in itinerary items.
- For **decision_tree** (and possibly checklist), the **payload shape** in training or in model output does not match the strict schema (`title` + `nodes` for decision_tree; `title` + `groups` for checklist).

---

## 6. Are training prompts / prompt QA too relaxed?

- **Envelope and itinerary**: Prompts and SFT are explicit and normalized. The main issues are **truncation** and **key choice** (`time` vs `time_block`), not “we didn’t ask for the format.”
- **Decision_tree (and possibly checklist)**: The gap is **payload structure**. If teacher-generated decision_tree data used something like `common`/`specific`/`root` instead of `title` + `nodes`, then prompt QA and data QA allowed that shape into training, so the model was never trained on schema-compliant decision_tree payloads. So for those types, **prompt QA / data QA could be tightened** so that:
  - Only payloads that validate against the current schemas are accepted, and  
  - Teacher prompts explicitly request the schema shape (e.g. `title` + `nodes` for decision_tree).

---

## 7. Summary and what would need to improve

| Question | Answer |
|----------|--------|
| Did validation fail? | No. Validation correctly enforces envelope + payload schemas. |
| Is validation too strict? | No. The contract (single root object, required keys, payload shapes) is intentional. |
| Did we tell the model the format? | Yes for envelope and itinerary (system instruction + SFT normalization). For decision_tree, training data may not match schema. |
| Why 0% schema_valid? | (1) **Truncation** → only inner array parseable → “got list”. (2) **Payload**: wrong key (`time` vs `time_block`) or wrong shape (no `title`/`nodes` for decision_tree). |
| Are prompts / QA too relaxed? | For envelope/itinerary, mostly no. For decision_tree (and similar), yes: payload shape should be enforced in data and prompts. |

**Concrete improvements for the future:**

1. **Truncation (main driver of “got list”)**  
   - Ensure **no truncation** during evaluation (e.g. `max_new_tokens` high enough, or stop at last complete `}`).  
   - Optionally, in lenient parsing, prefer the **longest** complete JSON object that starts at the first `{` (e.g. try to close unclosed braces heuristically) so that a truncated envelope might still count as an object. (Current design correctly refuses to invent a closing `}`.)

2. **Itinerary: time_block vs time**  
   - Keep SFT normalization. Consider **reinforcing in inference**: e.g. system message again “use only time_block, not time, for activity times”.  
   - Or **post-process**: map `time` → `time_block` for itinerary activity items before validation (if product accepts that).

3. **Decision_tree (and checklist) payload shape**  
   - **Audit training data**: ensure every decision_tree (and checklist) example validates against the current payload schema.  
   - **Teacher / data pipeline**: require schema validation before accepting a record; add explicit prompts for `title` + `nodes` (and `title` + `groups`).  
   - **Prompt QA**: reject or fix any example that doesn’t match the schema so the model never sees the wrong shape.

4. **Stricter data QA**  
   - Run payload validation (envelope + payload) on every training example; drop or fix any that fail. That way the model only sees schema-compliant outputs.

This gives a **fact-based** picture of why schema_valid_rate is 0% and what to change, without guessing.

---

## 8. Where is truncation happening: training data vs inference?

**Investigation:** Checked SFT training data (`data/processed/sft/v2/train_sft.jsonl`) and inference generation config.

**Finding 1: Training data is NOT truncated**  
- SFT examples are **complete** JSON objects with closing braces. The `target` field in each example contains full, valid JSON. The model was trained on complete envelopes.

**Finding 2: Inference output IS truncated**  
- From `finetuned_outputs.jsonl` (v4), raw outputs end mid-string (e.g. `"time_block\":\"09:0"`).  
- `tokens_generated` shows 2052–2502 tokens, but outputs are cut before the closing `}`.  
- `run_samples.py` sets `max_new_tokens = max(max_seq_len, 2048)` (line 308), so we're using at least 2048 tokens.  
- **Root cause**: The model is generating **multiple JSON objects** (repetition) or **very long single responses** that exceed even 2048 tokens. For example:
  - Sample `v2_hard_missing_verification_steps_0031`: Outputs prose, then JSON, then **repeats the same JSON**, then outputs a different payload type (checklist), then starts another that gets cut off.
  - Sample `v1_itinerary_entry_requirements_0051`: Generates a complete envelope, then **duplicates** `trip_days` array, then gets cut mid-string.

**Conclusion:**  
- **Training data is fine** — no need to redo the dataset.  
- **Inference truncation** is due to:
  1. **Repetition**: Model generates multiple JSON objects (learned behavior or lack of stop tokens).
  2. **Length**: Some responses exceed 2048 tokens even for a single envelope.
  3. **No explicit stop tokens**: Generation doesn't stop at EOS or a JSON-closing pattern.

**Do we need to redo the dataset?**  
**No.** The dataset has complete examples. The issue is **inference generation** (repetition, length, stop tokens), not training data quality.

---

## 9. Targeted improvements for v5 (if iterating further)

If you decide to do a v5 iteration, here are **specific, actionable** improvements based on the evidence above:

### 9.1 Fix inference truncation (addresses 15/20 "got list" failures)

**Problem:** Model generates multiple JSON objects or very long responses that exceed `max_new_tokens`, causing truncation before the closing `}`.

**Solutions:**

1. **Increase `max_new_tokens` for evaluation**  
   - Current: `max(max_seq_len, 2048)` (often 2048).  
   - Change: Use **4096** or **5120** for evaluation to ensure even long responses complete.  
   - Location: `src/pocketguide/train/run_samples.py` line 308.

2. **Add explicit stop tokens**  
   - Add `eos_token_id` and potentially a custom stop sequence (e.g. `"\n}\n"` after a complete JSON) to `GenSpec` and `model.generate()` call.  
   - This prevents the model from generating multiple JSON objects.  
   - Location: `src/pocketguide/inference/base_model.py` `generate_one()` and `generate_batch()`.

3. **Post-process to extract first complete JSON**  
   - After generation, scan the output for the **first complete** `{...}` object (bracket-matched) and use only that.  
   - This handles cases where the model outputs prose + JSON or multiple JSONs.  
   - Location: Add a helper in `run_samples.py` before `parse_and_validate()`.

4. **Reduce repetition penalty or adjust sampling**  
   - Current: `repetition_penalty=1.1`. Try **1.2–1.3** to reduce duplicate JSON generation.  
   - Or use **nucleus sampling** (`top_p=0.95`) with lower temperature to reduce repetition.

### 9.2 Fix payload schema failures (addresses 5/20 failures)

**Problem:** Model uses wrong keys (`time` instead of `time_block`) or wrong payload shape (decision_tree without `title` + `nodes`).

**Solutions:**

1. **Stricter SFT data validation**  
   - Before writing `train_sft.jsonl`, validate **every** example with `parse_and_validate()` (envelope + payload).  
   - Drop or fix any example that fails. This ensures the model **never** sees schema-invalid examples.  
   - Location: `src/pocketguide/train/prepare_sft_data.py` — add validation step before `record_to_sft()`.

2. **Fix decision_tree training data**  
   - Audit `dataset_v2.jsonl` for decision_tree examples. Check if they have `payload.title` and `payload.nodes`.  
   - If not, either:
     - **Regenerate** those examples with teacher prompts that explicitly request `title` + `nodes` structure.
     - **Transform** existing decision_tree payloads to match the schema (if possible).
   - Location: Add a data QA step in `src/pocketguide/data_generation/qa_dataset.py` or create a migration script.

3. **Reinforce in system instruction**  
   - Add explicit examples: "For decision_tree payloads, include 'title' (string) and 'nodes' (array of node objects with 'id', 'type', 'text')."  
   - Location: `src/pocketguide/train/prepare_sft_data.py` `SYSTEM_INSTRUCTION`.

4. **Post-process itinerary payloads**  
   - If acceptable for the product, add a normalization step: for itinerary activity items, if `time` exists and `time_block` doesn't, copy `time` → `time_block` and remove `time`.  
   - Location: Add helper in `run_samples.py` or `parsing.py` before payload validation.

### 9.3 Training improvements (if re-training)

**If you re-train v5:**

1. **More training steps**  
   - v4 had 8 epochs × ~40 steps/epoch = ~320 steps. Consider **10–12 epochs** or **grad_accum_steps=2** (more optimizer steps) if compute allows.

2. **Loss weighting on envelope keys**  
   - Weight loss higher for tokens corresponding to envelope keys (`summary`, `verification_steps`, `payload_type`, etc.) so the model prioritizes learning the structure.  
   - This is more complex but could help.

3. **Curriculum learning**  
   - Train first on **shorter examples** (envelope-only or simple payloads), then gradually add longer, more complex payloads.  
   - This helps the model learn the envelope structure before tackling long payloads.

### 9.4 Evaluation improvements

1. **Better truncation handling in parser**  
   - In `_parse_json_lenient()`, if we find an unclosed `{` but a complete inner array, try to **heuristically close** the object (e.g. add `}` at the end if the last character is `"` or `]`).  
   - This is a **band-aid** but could recover some truncated envelopes.  
   - Location: `src/pocketguide/eval/parsing.py` `_parse_json_lenient()`.

2. **Separate metrics for truncation vs schema**  
   - Track `truncation_rate` (output ends without closing `}`) separately from `schema_valid_rate`.  
   - This clarifies whether failures are due to truncation or wrong structure.

### 9.5 Priority order for v5

If doing v5, prioritize:

1. **High impact, low effort:**
   - Increase `max_new_tokens` to 4096 for eval (5 min change).
   - Add stop tokens to generation (15 min change).
   - Post-process to extract first complete JSON (30 min change).

2. **High impact, medium effort:**
   - Validate all SFT examples before training (1–2 hours).
   - Fix decision_tree training data (2–4 hours if regenerating, less if transforming).

3. **Medium impact, higher effort:**
   - Re-train with more epochs/steps (hours of compute).
   - Loss weighting or curriculum learning (requires code changes + re-train).

**Expected outcome:** If you fix inference truncation (stop tokens + higher max_new_tokens + post-process), you should see **"got list" failures drop from 15/20 to ~5–8/20**. If you also fix payload schema in training data, you should see **schema_valid_rate jump from 0% to 10–30%** (depending on how many examples were schema-invalid in training).
