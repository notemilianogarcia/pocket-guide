# v5 Implementation Summary

Complete implementation of all three additional recommendations for v5.

## What Was Implemented

### 1. SFT Data Validation ✅

**Location:** `src/pocketguide/train/prepare_sft_data.py`

**Changes:**
- Added `--validate` flag (default: True) to `prepare_sft_data`
- `record_to_sft()` now validates each example with `parse_and_validate()` (envelope + payload schemas)
- Drops examples that fail validation (returns `None` with error message)
- Prints summary of dropped examples during SFT generation

**Impact:**
- Model only sees schema-compliant examples during training
- Catches issues like missing `title` in decision_tree/checklist payloads
- Prevents training on malformed data

**Usage:**
```bash
python3 -m pocketguide.train.prepare_sft_data \
  --splits_dir data/processed/splits/v2 \
  --out_dir data/processed/sft/v2 \
  --validate  # Default, can omit
```

### 2. Decision Tree & Checklist Payload Normalization ✅

**Location:** `src/pocketguide/train/prepare_sft_data.py` → `_normalize_payload_for_sft()`

**Changes:**
- Added normalization for `decision_tree`: ensures `title` and `nodes` are present
- Added normalization for `checklist`: ensures `title` and `groups` are present
- Provides sensible defaults if missing (e.g., `title: "Decision Tree"`)

**Impact:**
- Fixes schema-invalid payloads before validation
- Ensures all training examples have required payload fields
- Works alongside validation (normalization fixes, validation catches other issues)

**Note:** Found 1 invalid decision_tree example in dataset_v2 (`v2_hard_missing_verification_steps_0040` missing `title`). Normalization fixes this automatically.

### 3. More Training (10 epochs, grad_accum 2) ✅

**Location:** `configs/train_lora_v5.yaml`

**Changes:**
- `num_epochs`: 8 → **10** (25% more epochs)
- `grad_accum_steps`: 4 → **2** (doubles optimizer steps per epoch)
- `warmup_steps`: 30 → **40** (~6% of total steps)

**Impact:**
- **Total optimizer steps:** ~640 (v4 had ~320)
  - v4: 8 epochs × ~40 steps/epoch = ~320 steps
  - v5: 10 epochs × ~64 steps/epoch = ~640 steps
- Model sees each example **more times** (10 epochs vs 8)
- **More optimizer updates** per epoch (grad_accum 2 vs 4)
- Better learning signal for schema compliance

**Training time:** ~1.5-2.5 hours on A100/L40S (longer due to more epochs)

## Complete v5 Pipeline

### Step 1: Regenerate SFT with Validation
```bash
python3 -m pocketguide.train.prepare_sft_data \
  --splits_dir data/processed/splits/v2 \
  --out_dir data/processed/sft/v2 \
  --seed 42 \
  --fixed_prompts_out eval/suites/fixed20_v1.jsonl \
  --validate
```

**Expected:** May drop 0-5 examples (prints which ones). All remaining examples are schema-valid.

### Step 2: Train v5
```bash
python3 -m pocketguide.train.train --config configs/train_lora_v5.yaml
```

**Output:** `runs/train/<timestamp>-v5/` with adapter weights.

### Step 3: Evaluate v5
```bash
python3 -m pocketguide.train.run_samples \
  --run_dir runs/train/<RUN_ID> \
  --prompts eval/suites/fixed20_v1.jsonl \
  --batch_size 8
```

## Expected Improvements

Combining all v5 changes:

1. **Inference fixes** (from previous implementation):
   - `max_new_tokens: 4096` → handles longer responses
   - Stop sequences → prevents multiple JSON generation
   - JSON extraction → handles prose prefix, multiple JSONs
   - `repetition_penalty: 1.2` → reduces repetition

2. **SFT validation**:
   - Model only sees schema-compliant examples
   - Catches payload schema issues before training

3. **More training**:
   - 2× optimizer steps (640 vs 320)
   - 25% more epochs (10 vs 8)
   - Better learning signal

**Expected schema_valid_rate:** 0% → **20-40%** (depending on how many examples were schema-invalid in training data)

## Files Changed

1. `src/pocketguide/train/prepare_sft_data.py` - Added validation, payload normalization
2. `src/pocketguide/inference/base_model.py` - Stop sequences (already done)
3. `src/pocketguide/train/run_samples.py` - JSON extraction, higher max_new_tokens (already done)
4. `configs/train_lora_v5.yaml` - Updated training params
5. `docs/v5_commands.md` - Updated command list
6. `docs/v5_summary.md` - This file

## Testing

All code changes compile and pass linting. Validation logic tested:
- ✅ Valid examples are accepted
- ✅ Invalid examples are caught (normalization fixes some, validation catches others)

Ready for training!
