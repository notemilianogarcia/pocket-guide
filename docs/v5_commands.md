# v5 Training and Evaluation Commands

Complete command list for v5 iteration (inference fixes: stop tokens, higher max_new_tokens, JSON extraction).

## Prerequisites

- **Fresh Lightning.ai environment?** See [lightning_setup_v5.md](lightning_setup_v5.md) for complete setup instructions.
- v2 dataset and splits already exist (`data/processed/dataset_v2.jsonl`, `data/processed/splits/v2/`)
- Python environment set up with `pip install -e .`
- **v5 requires regenerating SFT data** (see Step 1 below)

## Step 1: Regenerate SFT Data with Validation (Required for v5)

**v5 requires regenerating SFT data** to drop schema-invalid examples:

```bash
# Regenerate SFT with validation (drops invalid examples)
python3 -m pocketguide.train.prepare_sft_data \
  --splits_dir data/processed/splits/v2 \
  --out_dir data/processed/sft/v2 \
  --seed 42 \
  --fixed_prompts_out eval/suites/fixed20_v1.jsonl \
  --validate
```

**What this does:**
- Validates every example with `parse_and_validate()` (envelope + payload schemas)
- Drops examples that fail validation (prints which ones were dropped)
- Normalizes payloads (itinerary: time_block; decision_tree/checklist: ensures title + nodes/groups)

**Expected output:**
- `data/processed/sft/v2/train_sft.jsonl` - validated train examples (may be slightly fewer than before)
- `data/processed/sft/v2/val_sft.jsonl` - validated val examples
- Console output showing how many examples were dropped (if any)

**Note:** If you see "Dropped 0 train, 0 val examples", all examples were already valid (good!).

## Step 2: Train v5

**On Lightning.ai** (recommended):

```bash
# From project root on Lightning
python3 -m pocketguide.train.train --config configs/train_lora_v5.yaml
```

**Local** (if you have GPU):

```bash
# From project root
python3 -m pocketguide.train.train --config configs/train_lora_v5.yaml
```

**Important:** Make sure Step 1 (SFT regeneration) completed successfully before training!

**Training config:** v5 improvements:
- **10 epochs** (v4 had 8) - more training
- **grad_accum_steps: 2** (v4 had 4) - more optimizer steps (~640 total vs v4's ~320)
- Same LoRA (r=32, alpha=64), max_seq_len 2048
- Training takes ~1.5-2.5 hours on A100/L40S (longer due to more epochs).

**Output:** `runs/train/<timestamp>-v5/` with adapter weights.

## Step 3: Evaluate v5

**On Lightning.ai** (after training completes):

```bash
# Replace <RUN_ID> with your actual v5 run folder (e.g., 20260204_123456-v5)
python3 -m pocketguide.train.run_samples \
  --run_dir runs/train/<RUN_ID> \
  --prompts eval/suites/fixed20_v1.jsonl \
  --batch_size 8
```

**What v5 fixes do:**
- **max_new_tokens: 4096** (up from 2048) - handles longer responses
- **repetition_penalty: 1.2** (up from 1.1) - reduces duplicate JSON generation
- **Stop sequences:** `["\n}\n", "\n}\n\n"]` - stops after complete JSON object
- **JSON extraction:** Extracts first complete `{...}` object from output (handles prose prefix, multiple JSONs)

**Output:** `runs/train/<RUN_ID>/samples/` with:
- `base_outputs.jsonl` - base model outputs
- `finetuned_outputs.jsonl` - v5 adapter outputs
- `comparison_metrics.json` - aggregate metrics
- `schema_failures_debug.jsonl` - per-sample failure details

## Step 4: Compare Results

Check `comparison_metrics.json`:

```bash
cat runs/train/<RUN_ID>/samples/comparison_metrics.json | python3 -m json.tool
```

**Expected improvements (v5 vs v4):**
- **"got list" failures:** Should drop from 15/20 to ~5-10/20 (truncation fixed)
- **schema_valid_rate:** Should improve from 0% to 10-30% (if truncation was main issue)
- **parse_success_rate:** Should stay at 100% (v4 already achieved this)

## Step 5: Re-evaluate Previous Versions (Optional)

To compare v5 against v2/v3/v4 with the **same** inference fixes:

```bash
# v2
python3 -m pocketguide.train.run_samples \
  --run_dir runs/train/20260203_013228-v2 \
  --prompts eval/suites/fixed20_v1.jsonl \
  --batch_size 8

# v3
python3 -m pocketguide.train.run_samples \
  --run_dir runs/train/20260203_041745-v3 \
  --prompts eval/suites/fixed20_v1.jsonl \
  --batch_size 8

# v4
python3 -m pocketguide.train.run_samples \
  --run_dir runs/train/20260203_053120-v4 \
  --prompts eval/suites/fixed20_v1.jsonl \
  --batch_size 8
```

**Note:** These will use v5 inference fixes (4096 tokens, stop sequences, JSON extraction), so results may differ from original v2/v3/v4 runs.

## Troubleshooting

**OOM (Out of Memory):**
- Reduce `--batch_size` from 8 to 4 or 2
- Or use smaller GPU (L4 instead of L40S)

**Still seeing "got list" failures:**
- Check `schema_failures_debug.jsonl` - if `raw_output_len` is very high (>4000), model may be generating extremely long responses
- Consider increasing `max_new_tokens` further (5120) if needed

**Stop sequences not working:**
- JSON extraction (`_extract_first_complete_json`) is the primary fix - stop sequences are supplementary
- Check `extracted_json` field in outputs to see if extraction happened

## Next Steps After v5

If v5 shows improvement but still not perfect:
1. **Fix payload schema in training data** (Section 9.2 in `schema_validity_analysis.md`)
2. **Validate all SFT examples** before training (drop schema-invalid ones)
3. **More training steps** (10-12 epochs instead of 8)

See `docs/schema_validity_analysis.md` Section 9 for detailed v5+ improvements.
