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

## 2. v4: focused iteration (optional)

Your debug showed: **most failures = "got list"** (model outputs array instead of object); **2 had full envelope** but failed on payload (`time_block`, `time_buffer`). So the direction is right; a bit more training can help.

**v4 scope (minimal):**

- **Same data** as v3 (v2 splits + SFT). No new data pipeline.
- **7 epochs** (v3 had 5) → ~140 optimizer steps.
- **Same** max_seq_len 2048, LoRA r=32, warmup, etc.

**Config:** `configs/train_lora_v4.yaml` (already added). Run ID will be `<timestamp>-v4`.

**Commands (Lightning):**

```bash
# 1. Train v4 (same SFT as v3; no extra data step)
python3 -m pocketguide.train.train --config configs/train_lora_v4.yaml

# 2. Eval v4 (with batching; replace RUN_ID with your v4 run folder)
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
