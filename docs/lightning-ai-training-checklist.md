# Lightning.ai training checklist

Use this after the data pipeline is complete (all batches run, `data/processed/dataset_v1.jsonl` is final). It replicates the environment on Lightning.ai and runs LoRA training.

---

## 1. Local: prepare splits and SFT data

From the **repo root** on your machine:

```bash
make split
make prepare-sft
```

- **Input:** `data/processed/dataset_v1.jsonl`
- **Output:**
  - `data/processed/splits/v1/train.jsonl`, `val.jsonl`, `test.jsonl`
  - `data/processed/sft/v1/train_sft.jsonl`, `val_sft.jsonl`
  - `eval/suites/fixed20_v1.jsonl` (fixed eval prompts)

Training uses **`train_sft.jsonl`** and **`val_sft.jsonl`**. Ensure these files exist and have the expected number of lines (e.g. ~90 train, ~11 val for 112 accepted samples with 80/10/10 split).

---

## 2. Decide how data will be available on Lightning

Training needs **both** split JSONL and SFT JSONL:

- **Splits:** `data/processed/splits/v1/train.jsonl`, `val.jsonl` (used for config validation and run metadata).
- **SFT:** `data/processed/sft/v1/train_sft.jsonl`, `val_sft.jsonl` (used for actual training).

Choose one:

- **Option A – Data in repo:** Commit and push `data/processed/splits/v1/` and `data/processed/sft/v1/` (or the whole `data/processed/`) so they are in the cloned repo on Lightning. Easiest if the repo is private and the files are not huge.
- **Option B – Data elsewhere:** Push only code. Upload both `data/processed/splits/v1/` and `data/processed/sft/v1/` to cloud storage or a Lightning Data artifact. On Lightning, download or mount them and set the config paths in step 9.

---

## 3. Push code (and optionally data)

```bash
git add .
git commit -m "Add SFT data and config for Lightning training"
git push origin main
```

If using Option B, upload `data/processed/sft/v1/` to your chosen storage and note the path or URL you will use on Lightning.

---

## 4. On Lightning: create a Run or Studio

- Go to [lightning.ai](https://lightning.ai) and create a **Run** (or **Studio**).
- Choose a machine with a GPU (e.g. A100 or similar) and enough disk for the repo + data + model checkpoints.

---

## 5. On Lightning: clone the repo

```bash
git clone https://github.com/YOUR_ORG/pocket-guide.git
cd pocket-guide
```

Replace `YOUR_ORG` with your GitHub org or username.

---

## 6. On Lightning: get data (if not in repo)

- **If Option A:** Splits and SFT data are already under `data/processed/`. Skip to step 7.
- **If Option B:** Download or mount so that both exist:
  - Splits: `train.jsonl`, `val.jsonl` (e.g. under `/data/splits_v1/`).
  - SFT: `train_sft.jsonl`, `val_sft.jsonl` (e.g. under `/data/sft_v1/`).
  You will set the training config paths in step 9.

---

## 7. On Lightning: install (no venv)

**Lightning Studio allows only one environment (the default conda env).** Do **not** run `make env`; use the system/conda Python and install into it:

```bash
cd pocket-guide
pip install --upgrade pip
pip install -e .
```

For dev tools (pytest, ruff) as well:

```bash
pip install -e ".[dev]"
```

This installs the project and dependencies (PyTorch, transformers, peft, accelerate, etc.) from `pyproject.toml` into the current environment.

---

## 8. On Lightning: Hugging Face token (for Llama-2)

The default base model in `configs/train_lora.yaml` is **`meta-llama/Llama-2-7b-hf`**, which is gated. You must be logged in to Hugging Face:

```bash
pip install -U huggingface_hub
huggingface-cli login
```

Paste your HF token when prompted. Alternatively, set the environment variable:

```bash
export HF_TOKEN=your_token_here
```

If you prefer to avoid gated models, edit `configs/train_lora.yaml` and set e.g. `base_model_id: "gpt2"` (or another open model) for a quick test; use Llama-2 for real training.

---

## 9. On Lightning: point config at data (if not in repo)

- **If data is in the repo** at `data/processed/splits/v1/` and `data/processed/sft/v1/`, the default config is already correct; no change needed.
- **If data is elsewhere**, edit `configs/train_lora.yaml` and set all four paths:

  ```yaml
  data:
    train_path:     /path/on/lightning/train.jsonl
    val_path:       /path/on/lightning/val.jsonl
    train_sft_path: /path/on/lightning/train_sft.jsonl
    val_sft_path:   /path/on/lightning/val_sft.jsonl
  ```

  The script validates that `train_path` and `val_path` exist and have the required fields; it uses `train_sft_path` and `val_sft_path` for training.

---

## 10. On Lightning: dry run (optional)

Validate config and data without loading the full model:

```bash
python -m pocketguide.train.train --config configs/train_lora.yaml --dry_run
```

(On Lightning, skip `make train-dry-run`—it uses the venv. Use the `python -m` command above.)

---

## 11. On Lightning: run training

From the repo root:

```bash
python -m pocketguide.train.train --config configs/train_lora.yaml
```

(On Lightning Studio, do **not** use `make train-run`—it expects a venv. Use the command above.)

Checkpoints and logs go to **`runs/train/`** (or the `output.runs_dir` from the config). Training uses the SFT dataset built from `train_sft.jsonl` and `val_sft.jsonl`.

---

## 12. After training: save or download outputs

- **Checkpoints** are under `runs/train/<run_id>/` (e.g. `checkpoint-200`).
- Download the run directory or sync it to cloud storage so you can use the fine-tuned model for inference or evaluation.

---

## Getting updates after you push (on Lightning)

If you already cloned the repo on Lightning and then push changes from your machine (e.g. new config, new base model), pull on Lightning:

```bash
cd ~/pocket-guide   # or wherever you cloned
git pull origin main
```

Then re-run training (or dry run) with the updated config. No need to re-run `pip install -e .` unless you changed `pyproject.toml` or dependencies.

---

## Quick reference

| Step | Action |
|------|--------|
| 1 | Local: `make split && make prepare-sft` |
| 2 | Choose: data in repo (A) or data in cloud/mount (B) |
| 3 | Push code (and data if A) |
| 4 | Lightning: create Run/Studio with GPU |
| 5 | Clone repo on Lightning |
| 6 | Get SFT data on Lightning (if B) |
| 7 | `pip install -e .` (no venv on Lightning Studio) |
| 8 | `huggingface-cli login` (for Llama-2) |
| 9 | Set `train_sft_path` / `val_sft_path` in config if data not in repo |
| 10 | Optional: `make train-dry-run` |
| 11 | `make train-run` or `python -m pocketguide.train.train --config configs/train_lora.yaml` |
| 12 | Download/sync `runs/train/` when done |
