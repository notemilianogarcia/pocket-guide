# v4 + faster evaluation

## 1. Faster evaluation (batched inference)

Evaluation was taking 30–45 minutes for 20 prompts because generation ran **one prompt at a time** (low GPU utilization). Batched inference is now the default.

**Change:** `run_samples` uses **batched generation** (default `--batch_size 8`). With 20 prompts you run 3 base batches + 3 finetuned batches instead of 40 single-prompt runs. Expect **roughly 4–6× faster** wall time (e.g. ~8–12 min instead of 45 min on an A100).

**Usage (Lightning, no make):**

```bash
# Default batch_size=8 (recommended on A100)
python3 -m pocketguide.train.run_samples \
  --run_dir runs/train/<RUN_ID> \
  --prompts eval/suites/fixed20_v1.jsonl

# If you hit OOM, reduce batch size
python3 -m pocketguide.train.run_samples \
  --run_dir runs/train/<RUN_ID> \
  --prompts eval/suites/fixed20_v1.jsonl \
  --batch_size 4
```

Same outputs: `comparison_metrics.json`, `schema_failures_debug.jsonl`, base/finetuned JSONL.

---

## 2. v4: last iteration (prompt + data QA + more training)

Your debug showed: **most failures = "got list"**; **2 had full envelope** but failed on payload (`time_block` missing, `time_buffer` unexpected). v4 improves that without adding new data.

**v4 scope:**

- **Prompt:** System instruction now says "Output a single JSON object (not a list or array)" and "For itinerary payloads, every activity item must have \"time_block\"". Reduces list-first and payload key mistakes.
- **Data QA (SFT):** `prepare_sft_data` normalizes itinerary payloads: every activity item gets `time_block` (from `time_buffer` if present, else default); `time_buffer` is removed so the model never sees the wrong key.
- **More training:** 8 epochs, grad_accum 4 → ~320 optimizer steps (v3 had 5 epochs, grad_accum 8 → ~100 steps). Same max_seq_len 2048, LoRA r=32, warmup 30.
- **Same data:** v2 splits; no new data. **Re-run prepare_sft_data for v2** before training v4 so SFT gets the new prompt and payload normalization.

**Config:** `configs/train_lora_v4.yaml` (already added). Run ID will be `<timestamp>-v4`.

**Commands (Lightning):**

```bash
# 1. Re-run prepare_sft_data so SFT has new prompt + itinerary time_block normalization
python3 -m pocketguide.train.prepare_sft_data \
  --splits_dir data/processed/splits/v2 \
  --out_dir data/processed/sft/v2 \
  --seed 42 \
  --fixed_prompts_out eval/suites/fixed20_v1.jsonl

# 2. Train v4
python3 -m pocketguide.train.train --config configs/train_lora_v4.yaml

# 3. Eval v4 (with batching; replace RUN_ID with your v4 run folder)
python3 -m pocketguide.train.run_samples \
  --run_dir runs/train/<RUN_ID> \
  --prompts eval/suites/fixed20_v1.jsonl
```

**When to do v4:** If you want one more push for a non-zero schema_valid rate before M8 (quantization) and M9 (portfolio). If you’re fine with the current v3 story (“envelope learned for some; list vs object + payload details remain”), you can skip v4 and go straight to M8 + M9.

---

## 3. Recommendation

1. **Re-run v3 eval with batching** (or run v4 and eval v4) so it finishes in ~10 min and doesn’t get disconnected.
2. **Option A:** Do **v4** (one training run + batched eval), then **M8 + M9** and move on.
3. **Option B:** Skip v4; **finalize v3 results** (use partial debug + comparison_metrics if you have them, or re-run v3 eval with batching); then **M8 + M9** and move on.

Either way, the project is **not** a waste: you have evaluation-driven iteration, clear metrics, and a documented path from 0% to partial compliance. Portfolio story works with or without a big schema_valid number.
