# V1 → V2 → MORE: What’s done, what to run, and v3 plan

## 1. Do you need to re-run v1 comparison?

**No.** v1 comparison was **already run**. You have:

- **v1 run:** `runs/train/20260131_224510`
- **v1 samples:** `runs/train/20260131_224510/samples/` with `base_outputs.jsonl`, `finetuned_outputs.jsonl`, `comparison_metrics.json`

So the **v1 baseline is done**. You can use it as-is for the story.

**Optional (apples-to-apples):** If you want v1 and v2 evaluated with **exactly** the same code (e.g. current `max_new_tokens` ≥ 2048, lenient parse preferring object), re-run v1 **once** with the current code. Same command, v1 run dir:

```bash
python -m pocketguide.train.run_samples \
  --run_dir runs/train/20260131_224510 \
  --prompts eval/suites/fixed20_v1.jsonl
```

If you don’t re-run, your story is still valid: “v1 finetuned improved parse vs base but 0% schema_valid; we then improved validation and generation settings and ran v2.”

**Summary:** v1 comparison is done. Re-run only if you want v1 metrics under the same evaluation code as v2.

---

## 2. Is v2 validation correct? Should you run it completely or iterate on validation?

**Yes. Validation is correct.** The envelope schema and checks match the contract (all 7 root fields; object, not array). Rejections are due to **model output** (missing fields or wrong root type), not validator bugs.

**Run v2 evaluation completely.** Use the current code and let it finish even if `schema_valid_rate` is 0%. You will get:

- Correct, comparable metrics (base vs finetuned).
- `samples/schema_failures_debug.jsonl` and `[schema fail]` logs to inspect **why** samples failed (missing fields, array vs object, etc.).

Do **not** change validation to “make” v2 pass. Fix the **model** (data + training) in v3 instead.

**Summary:** Validation is correct. Run v2 evaluation to completion; use the results and debug file to drive v3 improvements.

---

## 3. v3: same data, better training so the model is clearly useful

v2 had **only 10 optimizer steps** (1 epoch, ~160 samples with grad_accum 16). That’s too little for the model to reliably learn the full envelope. v3 should keep **the same dataset (v2)** and change **training** so the model actually learns the format.

### v3 changes (all training/config, no new data)

1. **More training**
   - **num_epochs: 3** (or at least 2) so the model sees each example multiple times.
   - Optionally **grad_accum_steps: 8** so you get more optimizer steps per epoch (e.g. ~20 steps/epoch instead of ~10), for finer updates.

2. **Longer sequences so the full envelope is in the loss**
   - **max_seq_len: 2048** (if GPU memory allows; e.g. 24GB L4 with gradient_checkpointing can often do 2048).  
   - If you keep 1024, many responses will be truncated during training, so the model may never see `verification_steps` / `payload_type` / end of `payload` in the loss. That directly hurts schema compliance.

3. **LR warmup**
   - **warmup_steps: 10** (or 5% of total steps) so the adapter isn’t hit with full LR on step 1.

4. **Same data, no schema changes**
   - Keep using `dataset_v2` / `data/processed/sft/v2/`. Ensure every SFT example has all 7 envelope fields (your pipeline already targets this). No validation changes.

### What “MORE” means in the story

- **V1:** Baseline finetune; 0% schema_valid; parse improved vs base.
- **V2:** Same data, one LR tweak; still 0% schema_valid (and only 10 steps).
- **MORE (v3):** Same data, **more epochs + longer seq + warmup** so the model actually learns the envelope; goal is **non-zero schema_valid** and clearly useful behavior.

### v3 config and commands

Config: `configs/train_lora_v3.yaml` (already added). Same data as v2; changes:

- `data.max_seq_len: 2048` (if OOM on 24GB, set to 1536 in the config).
- `training.num_epochs: 3`, `training.warmup_steps: 10`, `training.grad_accum_steps: 8`.

**Commands:** Same splits as v2; **re-run prepare SFT v2** once so SFT data gets envelope normalization (all 7 keys in every example). Then train and eval.

**On Lightning.ai (no make)** — use Python only. See [docs/lightning_python_commands.md](lightning_python_commands.md) for the full list.

1. Prepare SFT v2 (envelope normalization, required before v3):
   ```bash
   python3 -m pocketguide.train.prepare_sft_data \
     --splits_dir data/processed/splits/v2 \
     --out_dir data/processed/sft/v2 \
     --seed 42 \
     --fixed_prompts_out eval/suites/fixed20_v1.jsonl
   ```
2. Train v3:
   ```bash
   python3 -m pocketguide.train.train --config configs/train_lora_v3.yaml
   ```
3. After training, run evaluation (replace `<RUN_ID>` with your run folder, e.g. `20260204_123456-v3`):
   ```bash
   python3 -m pocketguide.train.run_samples \
     --run_dir runs/train/<RUN_ID> \
     --prompts eval/suites/fixed20_v1.jsonl
   ```
   Outputs: `runs/train/<RUN_ID>/samples/comparison_metrics.json`, `schema_failures_debug.jsonl`, and per-sample logs.

Compare v3 finetuned vs v2 finetuned (and vs v1 finetuned) using `comparison_metrics.json` and `schema_failures_debug.jsonl`.

---

## Quick reference

| Question | Answer |
|----------|--------|
| Re-run v1? | No. Already done. Optional: re-run once with current code for same eval settings as v2. |
| v1 re-run command (optional) | `python -m pocketguide.train.run_samples --run_dir runs/train/20260131_224510 --prompts eval/suites/fixed20_v1.jsonl` |
| v2 validation correct? | Yes. Run v2 evaluation completely even if 0% schema_valid. |
| v2 eval command | `python -m pocketguide.train.run_samples --run_dir runs/train/20260203_013228-v2 --prompts eval/suites/fixed20_v1.jsonl` (or `make run-samples RUN_DIR=runs/train/20260203_013228-v2`) |
| v3 focus | Same data (v2). More epochs, max_seq_len 2048, warmup. No validation changes. |
| v3 train + eval | `make train-v3` (or `make split-v2 && make prepare-sft-v2` first if needed); then `make run-samples RUN_DIR=runs/train/<run_id>-v3` |
