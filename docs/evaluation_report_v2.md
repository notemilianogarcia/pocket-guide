# Evaluation Report v2 — Targeted Iteration Results

## 1. Summary

This iteration attempted to fix two failure modes identified in Lesson 7.1: (1) JSON format violations (truncation and malformed JSON) and (2) schema and required-field failures (missing envelope or payload fields such as `verification_steps`). The approach was data-centric (dataset v2 with targeted interventions) plus a single training change for v2, followed by further training iterations (v3, v4) to increase learning signal and tighten SFT data.

What changed: dataset v2 added hard prompts and synthetic examples aligned to the target failure modes, stricter rejection in QC, and oversampling; v2 training used a single hyperparameter change (reduced learning rate) on the same base model; v3 increased epochs, sequence length, warmup, and LoRA capacity on the same v2 data; v4 added prompt and payload normalization in SFT preparation and further training steps. All runs were evaluated under the same adapter-based evaluation (run_samples on a fixed prompt suite) with the same parsing and validation logic.

The goal of achieving a non-zero schema validity rate (full envelope and payload compliance) was not achieved: finetuned schema_valid_rate remained 0% across v1, v2, v3, and v4. Parse success rate improved from 0.80 (v1) to 1.00 (v3, v4), and required_field_presence_rate improved from 0% (v1, v2) to 0.10 (v3) and 0.20 (v4), suggesting partial learning of envelope structure without full schema compliance.

---

## 2. Targeted Failure Modes

**Failure mode 1 — JSON format violations (truncation / malformed JSON)**

- **Description:** Output is not valid JSON or cannot be recovered by lenient parsing (e.g., truncated before closing braces, stray markdown, trailing commas).
- **Evidence:** Taxonomy category in Milestone 6; pipeline error type `json_parse` (strict and lenient parse failure). Quantized and smaller models frequently truncate or emit invalid JSON under context or token limits.
- **Impact:** Downstream parsing and schema checks do not run; high severity when inference is token- or context-limited.

**Failure mode 2 — Schema / required-field failures**

- **Description:** Output is valid JSON but fails envelope or payload schema due to missing required fields (e.g., `verification_steps`, `payload_type`, or payload-specific keys such as `time_block` in itinerary activities).
- **Evidence:** Pipeline error types `envelope_schema` and `payload_schema`; debug artifacts showed missing `verification_steps` and `payload_type`, or root-level array instead of object, or payload keys such as `time_buffer` instead of `time_block`.
- **Impact:** Envelope or payload validation fails; structured use of the response is blocked.

---

## 3. Interventions Applied

### 3.1 Dataset Intervention

Dataset v2 was built from the same v1 clean source with targeted interventions (see `data/processed/dataset_v2_manifest.json`):

- **Target failure modes:** `missing_verification_steps`, `invalid_json_truncation`.
- **Interventions:** Hard prompts added (50), targeted synthetic generation (50 attempted, 40 accepted, 80% acceptance rate), oversampling of v1 matches (factor 1.5), stricter rejection (verification_steps and uncertainty_notes).
- **Counts:** 274 v1 records kept; 50 hard prompts generated; 33 hard prompts for missing_verification_steps, 17 for invalid_json_truncation; 25 synthetic accepted for missing_verification_steps, 15 for invalid_json_truncation.
- **Rationale:** Hard prompts and synthetic examples were designed to elicit and reinforce complete envelope output and verification steps; stricter rejection ensures training examples have required fields. This aligns with the hypothesis that the model needs more exposure to complete, schema-consistent examples.

The resulting dataset (dataset_v2) was split into train/val/test; SFT preparation (prepare_sft_data) was run for v2 with envelope normalization (all seven envelope keys present). For v4, SFT was regenerated with strengthened system instruction (“Output a single JSON object (not a list or array)”, explicit “time_block” requirement for itineraries) and payload normalization (itinerary activity items: ensure `time_block`, remove `time_buffer`).

### 3.2 Training Adjustments

**v2 (run 20260203_013228-v2)**

- **Single change:** Learning rate reduced from 2.0e-4 (v1) to 1.5e-4. All other training and data paths switched to v2 splits and SFT; max_seq_len 1024; num_epochs 1; LoRA r=16, alpha=32.
- **Source:** `runs/train/20260203_013228-v2/config.yaml`, `meta.json`. Experiment one_change: “training.lr reduced to 1.5e-4 (v1 baseline 2.0e-4)”.
- **Rationale:** A single hyperparameter change isolates the effect of the dataset intervention; lower LR may improve stability on the new data.

**v3 (run 20260203_041745-v3)**

- **Changes (relative to v2):** Same v2 data; num_epochs 5 (v2 had 1); max_seq_len 2048 (v2 had 1024); warmup_steps 20; grad_accum_steps 8 (v2 had 16); LoRA r=32, alpha=64 (v2 had r=16, alpha=32).
- **Rationale:** More epochs and longer sequence length allow the model to see full envelope and payload in the loss; larger LoRA capacity and warmup support learning the full structure.

**v4 (run 20260203_053120-v4)**

- **Changes (relative to v3):** Same v2 data; SFT regenerated with strengthened prompt and itinerary payload normalization; num_epochs 8 (v3 had 5); grad_accum_steps 4 (v3 had 8); warmup_steps 30.
- **Rationale:** More optimizer steps and consistent SFT data (object-first prompt, time_block-only payloads) to reduce “root is list” and payload key errors.

---

## 4. Evaluation Setup (Controlled)

- **Benchmark suite:** Same fixed prompt set for all runs: `eval/suites/fixed20_v1.jsonl` (20 prompts). No change to prompts between v1, v2, v3, and v4 evaluation.
- **Evaluation script:** Adapter-based `run_samples` (pocketguide.train.run_samples): loads base model and LoRA adapter from each run directory, generates completions with the same generation config (max_new_tokens at least 2048, do_sample, temperature, top_p, repetition_penalty), and runs the same parsing and validation pipeline on each completion.
- **Runtime:** All evaluation runs used the same Hugging Face + PEFT runtime (no GGUF/llama.cpp for this comparison). Device and precision resolved from run config (e.g., cuda, bf16).
- **Parsing and validation:** Same `parse_and_validate` (strict then lenient JSON extraction; envelope schema v0; payload schemas v1). No changes to schemas or validation logic between runs.
- **Metrics:** comparison_metrics.json produced per run: parse_success_rate (lenient parse recovers valid JSON), schema_valid_rate (envelope and payload both pass), required_field_presence_rate (envelope required fields present when parse succeeds), uncertainty_marker_presence_rate, avg_latency_ms. schema_valid_rate implies both envelope and payload validation pass.

**Baseline and comparison runs**

- **v1 baseline (adapter):** run 20260131_224510 (v1 finetuned). Samples: `runs/train/20260131_224510/samples/comparison_metrics.json`.
- **v2:** run 20260203_013228-v2. Samples: `runs/train/20260203_013228-v2/samples/comparison_metrics.json`.
- **v3:** run 20260203_041745-v3. Samples: `runs/train/20260203_041745-v3/samples/comparison_metrics.json`.
- **v4:** run 20260203_053120-v4. Samples: `runs/train/20260203_053120-v4/samples/comparison_metrics.json`.

Note: runs under `runs/eval/` with `local_metrics.json` (e.g., 20260201_221952) used a stub/local runtime and do not reflect real model inference; they are not used as the baseline or comparison for this report.

---

## 5. Before / After Metrics

All metrics below are **finetuned** (adapter) results from comparison_metrics.json for each run (n=20 prompts). Base model metrics are omitted for brevity; base schema_valid_rate was 0% for v1 and 5% for v2/v3/v4 (one sample passed in base for the latter runs).

| Metric | v1 (224510) | v2 (013228-v2) | v3 (041745-v3) | v4 (053120-v4) |
|--------|-------------|----------------|----------------|----------------|
| parse_success_rate | 0.80 | 0.85 | 1.00 | 1.00 |
| schema_valid_rate | 0.00 | 0.00 | 0.00 | 0.00 |
| required_field_presence_rate | 0.00 | 0.00 | 0.10 | 0.20 |
| uncertainty_marker_presence_rate | 0.85 | 0.95 | 1.00 | 1.00 |
| avg_latency_ms | 42228.57 | 19738.82 | 19780.39 | 19280.70 |

**Deltas (v4 vs v1)**

- parse_success_rate: +0.20 (absolute).
- schema_valid_rate: 0.00 (no change).
- required_field_presence_rate: +0.20 (absolute).
- uncertainty_marker_presence_rate: +0.15 (absolute).
- avg_latency_ms: lower in v2/v3/v4 than v1 (different hardware or batching; not interpreted as a model quality delta).

Strict_parse_rate, lenient_parse_rate, envelope_valid_rate, and payload_valid_rate are not reported separately in comparison_metrics; schema_valid_rate is the joint envelope+payload pass rate. Latency and tokens/sec are not compared across runs for causal interpretation due to possible environment and batching differences.

---

## 6. Analysis: What Improved and Why

**Parse success rate**

- Increased from 0.80 (v1) to 0.85 (v2) to 1.00 (v3, v4). This is consistent with the hypothesis that more training (v3, v4), longer max_seq_len (v3, v4), and dataset v2 (v2 onward) improve the likelihood of emitting parseable JSON. Lenient parsing prefers a JSON object over an array when both exist; combined with “object-first” prompt in v4 SFT, the reduction in “root is list” failures in debug artifacts is consistent with this improvement.

**Required field presence rate**

- Increased from 0.00 (v1, v2) to 0.10 (v3) and 0.20 (v4). This metric is computed only over parse-successful samples and measures presence of required envelope fields. The improvement suggests that v3 and v4 training (more epochs, longer sequences, larger LoRA, and in v4 the explicit prompt and payload normalization) led to more often including some of the required envelope fields when the output is at least parseable. It does not imply full envelope+payload validity, which remained 0%.

**Uncertainty marker presence**

- Already high in v1 (0.85) and at or near 1.00 in v2–v4. No targeted intervention was applied for this; the result is consistent with the model retaining or slightly improving uncertainty phrasing.

**Schema validity (envelope + payload)**

- No improvement: 0% across all finetuned runs. Debug artifacts (e.g., schema_failures_debug.jsonl) show that failures are either (1) top-level JSON is a list instead of an object, or (2) envelope is correct but payload fails (e.g., missing `time_block`, or disallowed key `time_buffer`). The interventions (dataset v2, v3/v4 training, v4 prompt and payload QA) did not yield a non-zero schema_valid_rate under the same evaluation setup.

---

## 7. What Did Not Improve (or Regressed)

**Schema validity rate**

- Remained 0% for all finetuned runs (v1–v4). The goal of achieving at least some fully schema-compliant outputs was not met. Plausible reasons: (1) the fixed evaluation suite may include prompts that are difficult for the current model size and data mix; (2) training may need even more steps or different objectives (e.g., loss weighting on envelope keys); (3) some failure modes (e.g., emitting a list before the object) may require different prompting or decoding strategies rather than more of the same training.

**Required field presence rate**

- Still low in absolute terms (0.20 in v4). Many parse-successful samples still lack at least one required envelope field, so envelope validity alone is not achieved for most samples.

**v2 finetuned vs v2 base**

- In the v2 run, base schema_valid_rate was 0.05 (1/20) and finetuned was 0.00. So the v2 adapter did not preserve or improve the single base pass; one sample that passed at base failed under the v2 adapter. This is a small sample effect but worth noting as a lack of improvement rather than a gain.

No other metrics regressed; parse_success and required_field_presence improved or stayed flat from v1 to v4.

---

## 8. Conclusions and Next Steps

The Milestone 7 iteration was only partly successful. Targeted failure modes (JSON format and schema/required-field) were addressed with dataset v2 (hard prompts, synthetic examples, stricter QC) and a single v2 training change (LR), then with stronger training in v3 (epochs, sequence length, LoRA, warmup) and v4 (more steps, prompt and payload normalization). Parse success rate reached 100% (v3, v4) and required_field_presence_rate increased to 0.20 (v4), but schema_valid_rate remained 0% across all runs. The link from hypothesis to intervention to metric delta is clear for parse success and required-field presence; for full schema validity, the interventions did not produce a measurable improvement under the same evaluation conditions.

If continuing iteration, next steps could include: (1) expanding the training set or adding curriculum (e.g., envelope-only then full payload); (2) loss masking or weighting to emphasize envelope and payload required fields; (3) inference-time or decoding changes (e.g., object-first sampling or post-processing) to reduce list-first outputs; (4) re-evaluating on a larger or different prompt set to confirm whether the 0% schema_valid rate is stable or suite-dependent. This experiment demonstrates that improving parse success and partial structure (required fields) is achievable with data and training iteration, while full envelope+payload validity for this model and setup remains unsolved under the current evaluation.
