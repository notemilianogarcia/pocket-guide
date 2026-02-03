# Lightning.ai Setup Guide for v5

Complete setup instructions for a fresh Lightning.ai environment.

## Prerequisites

- Lightning.ai account (new account, fresh environment)
- GitHub repo with your code pushed (or access to clone it)
- Hugging Face account (for Llama-2-7b-hf access, if needed)

---

## Step 1: Clone Repository

**On Lightning.ai** (in a Run or Studio):

```bash
# Clone your repo (replace with your actual repo URL)
git clone https://github.com/YOUR_USERNAME/pocket-guide.git
cd pocket-guide

# Verify you're on the right branch (usually 'main')
git branch
git status
```

**If repo is private:** You may need to set up SSH keys or use a personal access token:
```bash
# Option A: SSH (if you have SSH keys set up)
git clone git@github.com:YOUR_USERNAME/pocket-guide.git

# Option B: Personal access token (replace TOKEN with your GitHub token)
git clone https://TOKEN@github.com/YOUR_USERNAME/pocket-guide.git
```

---

## Step 2: Set Up Python Environment

**On Lightning.ai:**

```bash
cd pocket-guide

# Check Python version (should be 3.11+)
python3 --version

# Create virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode (installs all dependencies)
pip install -e .

# Verify installation
python3 -c "import pocketguide; print('✓ Package installed')"
```

**Note:** Lightning.ai environments usually have Python 3.11+ pre-installed. The `pip install -e .` command reads `pyproject.toml` and installs all dependencies automatically.

---

## Step 3: Verify Data Files

**Check that training data exists in the repo** (should be tracked in git):

```bash
# Check dataset exists
ls -lh data/processed/dataset_v2.jsonl

# Check splits exist
ls -lh data/processed/splits/v2/train.jsonl
ls -lh data/processed/splits/v2/val.jsonl

# Check if SFT data exists (will be regenerated in Step 5 with validation)
ls -lh data/processed/sft/v2/train_sft.jsonl 2>/dev/null || echo "SFT data will be regenerated in Step 5"
```

**Expected:** All these files should exist after cloning (they're tracked in git).

**If data files are missing:**
- Check `git status` - they should be tracked
- If missing, you may need to pull from a different branch, or
- Upload them manually to Lightning (see "Alternative: Upload Data Files" below)

---

## Step 4: Set Up Hugging Face (Required for Llama-2)

**Required for training** (Llama-2-7b-hf requires Hugging Face access):

```bash
# Login to Hugging Face
huggingface-cli login

# Enter your Hugging Face token when prompted
# Get token from: https://huggingface.co/settings/tokens
```

**If you don't have access to Llama-2:**
1. Request access at: https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Wait for approval (usually quick)
3. Then run `huggingface-cli login` again

**Note:** You don't need OpenRouter API keys for v5 training/eval - we're using existing data.

---

## Step 5: Regenerate SFT Data with Validation (v5 Requirement - Important!)

**This is required for v5** - regenerates SFT data with schema validation:

```bash
# Make sure you're in the repo root
cd /path/to/pocket-guide  # or wherever you cloned it

# Regenerate SFT with validation
python3 -m pocketguide.train.prepare_sft_data \
  --splits_dir data/processed/splits/v2 \
  --out_dir data/processed/sft/v2 \
  --seed 42 \
  --fixed_prompts_out eval/suites/fixed20_v1.jsonl \
  --validate

# Expected output:
# v5 validation: Dropped X train, Y val examples (or "Dropped 0" if all valid)
# Wrote data/processed/sft/v2/train_sft.jsonl (N records)
# Wrote data/processed/sft/v2/val_sft.jsonl (M records)
# Wrote eval/suites/fixed20_v1.jsonl (20 prompts)
```

**Verify SFT files were created:**
```bash
wc -l data/processed/sft/v2/train_sft.jsonl data/processed/sft/v2/val_sft.jsonl
```

---

## Step 6: Verify Everything is Ready

**Quick sanity checks:**

```bash
# 1. Package installed
python3 -c "import pocketguide; print('✓ Package OK')"

# 2. Config exists
ls -lh configs/train_lora_v5.yaml

# 3. SFT data exists
ls -lh data/processed/sft/v2/train_sft.jsonl data/processed/sft/v2/val_sft.jsonl

# 4. Splits exist
ls -lh data/processed/splits/v2/train.jsonl data/processed/splits/v2/val.jsonl

# 5. Fixed prompts exist
ls -lh eval/suites/fixed20_v1.jsonl

# 6. Test config loading
python3 -c "
import yaml
with open('configs/train_lora_v5.yaml') as f:
    cfg = yaml.safe_load(f)
    print(f'✓ Config OK: {cfg[\"experiment\"][\"name\"]}')
"
```

---

## Step 7: Train v5

**Now you're ready to train:**

```bash
# Train v5 (takes ~1.5-2.5 hours on A100/L40S)
python3 -m pocketguide.train.train --config configs/train_lora_v5.yaml
```

**Monitor training:**
- Training logs print every 10 steps
- Checkpoints saved every 200 steps
- Evaluation runs every 100 steps (if val set is small enough)

**Output:** `runs/train/<timestamp>-v5/` with:
- `adapter/` - LoRA weights
- `config.yaml` - Training config snapshot
- `meta.json` - Run metadata
- `train_metrics.json` - Loss curves

---

## Step 8: Evaluate v5

**After training completes:**

```bash
# Replace <RUN_ID> with your actual v5 run folder (e.g., 20260204_123456-v5)
# Find it with: ls -lt runs/train/ | head -5

python3 -m pocketguide.train.run_samples \
  --run_dir runs/train/<RUN_ID> \
  --prompts eval/suites/fixed20_v1.jsonl \
  --batch_size 8
```

**If you get OOM (Out of Memory):**
```bash
# Reduce batch size
python3 -m pocketguide.train.run_samples \
  --run_dir runs/train/<RUN_ID> \
  --prompts eval/suites/fixed20_v1.jsonl \
  --batch_size 4  # or 2
```

**Check results:**
```bash
cat runs/train/<RUN_ID>/samples/comparison_metrics.json | python3 -m json.tool
```

---

## Alternative: Upload Data Files Manually

**If data files aren't in the repo** (gitignored or too large):

### Option A: Upload via Lightning UI
1. In Lightning file browser, navigate to `pocket-guide/data/processed/`
2. Upload `dataset_v2.jsonl` and `splits/v2/` folder
3. Or upload entire `data/processed/` directory

### Option B: Use `scp` or `rsync` from your local machine
```bash
# From your local machine (replace with your Lightning SSH details)
scp -r data/processed/ user@lightning.ai:/path/to/pocket-guide/data/
```

### Option C: Download from cloud storage
If you have data in S3/GCS/etc:
```bash
# Example for S3
aws s3 cp s3://your-bucket/pocket-guide/data/processed/ ./data/processed/ --recursive
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'pocketguide'"
```bash
# Make sure you installed the package
pip install -e .

# Verify you're in the repo root
pwd  # Should show .../pocket-guide
```

### "FileNotFoundError: data/processed/splits/v2/train.jsonl"
```bash
# Check if splits exist
ls -la data/processed/splits/v2/

# If missing, you may need to create splits first (see data pipeline docs)
# Or upload them manually (see "Alternative: Upload Data Files" above)
```

### "HuggingFace model access denied"
```bash
# Make sure you're logged in
huggingface-cli login

# Request access to meta-llama/Llama-2-7b-hf if needed
# https://huggingface.co/meta-llama/Llama-2-7b-hf
```

### "CUDA out of memory" during training
- Use a larger GPU (L40S 48GB instead of L4 24GB)
- Or reduce `max_seq_len` in config (2048 → 1536)
- Or enable gradient checkpointing (already enabled in v5 config)

### "CUDA out of memory" during evaluation
- Reduce `--batch_size` from 8 to 4 or 2
- Or use a larger GPU

---

## Quick Reference: Complete Setup Commands

**Copy-paste this entire block for a fresh setup:**

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/pocket-guide.git
cd pocket-guide

# 2. Install package
pip install -e .

# 3. Login to Hugging Face (required for Llama-2)
huggingface-cli login
# Enter token from https://huggingface.co/settings/tokens

# 4. Verify data exists (should be tracked in git)
ls -lh data/processed/splits/v2/train.jsonl

# 5. Regenerate SFT with validation (v5 requirement!)
python3 -m pocketguide.train.prepare_sft_data \
  --splits_dir data/processed/splits/v2 \
  --out_dir data/processed/sft/v2 \
  --seed 42 \
  --fixed_prompts_out eval/suites/fixed20_v1.jsonl \
  --validate

# 6. Train v5
python3 -m pocketguide.train.train --config configs/train_lora_v5.yaml

# 7. Evaluate (after training completes - replace <RUN_ID> with actual folder)
python3 -m pocketguide.train.run_samples \
  --run_dir runs/train/<RUN_ID> \
  --prompts eval/suites/fixed20_v1.jsonl \
  --batch_size 8
```

---

## Next Steps After Setup

Once v5 training completes:
1. Check `comparison_metrics.json` for schema_valid_rate
2. Compare against v2/v3/v4 results
3. If still not perfect, see `docs/schema_validity_analysis.md` Section 9 for further improvements

Good luck! 🚀
