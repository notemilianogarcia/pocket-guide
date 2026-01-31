# Full data pipeline → SFT training data

The **2+1** records you see come from minimal `train.jsonl` / `val.jsonl` files that were created by hand for smoke tests. They are **not** the output of the full pipeline.

To get **full** SFT data, you need to run the pipeline below. Order matters.

---

## Sample run (few prompts, real API)

To verify the full pipeline with real API calls without processing the whole plan:

1. **Set real-run mode:** In `configs/teacher.yaml` set **`runtime.dry_run: false`** and ensure **`OPENROUTER_API_KEY`** is in `.env`. If `dry_run` stays `true`, drafts and critiques return placeholder text and you will see 0% parse/schema pass rates.
2. **Run the sample pipeline** (default: 3 prompts; uses **dedicated dirs** so it does not mix with full-run data):

   ```bash
   make pipeline-sample
   ```

   Outputs go to **`data/interim/sample/`** (drafts, critiques) and **`data/processed/sample/`** (dataset, splits, SFT). So your main `data/interim/` and `data/processed/` files are untouched.

   Use a different number of samples:

   ```bash
   make pipeline-sample SAMPLE_N=5
   ```

   To re-run the sample pipeline from scratch, remove the sample dirs first:

   ```bash
   rm -rf data/interim/sample data/processed/sample
   make pipeline-sample
   ```

   After a successful run, check `data/processed/sample/dataset_v1.jsonl` and `data/processed/sample/sft/v1/`.

---

## If you already have plan + drafts

You have `data/interim/prompt_plan_v1.jsonl` and `data/interim/drafts_v1.jsonl`. Run only the rest:

```bash
make critiques && make dataset && make split && make prepare-sft
```

Then train with `make train-run` (or upload the repo and use `data/processed/sft/v1/` on Lightning).

---

## Pipeline overview

```
prompt_plan_v1.jsonl  →  drafts_v1.jsonl  →  critiques_v1.jsonl  →  dataset_v1.jsonl
       (you have)           (teacher API)        (teacher API)         (teacher API)
                                                                              ↓
                                                        dataset_v1_clean.jsonl (optional QA)
                                                                              ↓
                                                    split_dataset_v1  →  train/val/test.jsonl
                                                                              ↓
                                                    prepare_sft_data  →  train_sft.jsonl, val_sft.jsonl
```

---

## Step-by-step

### 1. Prompt plan (you already have this)

- **Output:** `data/interim/prompt_plan_v1.jsonl` (e.g. 120 records)
- If you need to regenerate:
  ```bash
  python -m pocketguide.data_generation.plan_prompts \
    --spec data/specs/dataset_v1_spec.yaml \
    --out_dir data/interim
  ```

### 2. Drafts (teacher / API)

- **Input:** `data/interim/prompt_plan_v1.jsonl`
- **Output:** `data/interim/drafts_v1.jsonl`
- **Requires:** Teacher config (OpenRouter or similar) and API key (e.g. `OPENROUTER_API_KEY` in `.env`).
- **Command:**
  ```bash
  python -m pocketguide.data_generation.generate_drafts \
    --plan data/interim/prompt_plan_v1.jsonl \
    --out_dir data/interim
  ```

### 3. Critiques (teacher / API)

- **Input:** `data/interim/drafts_v1.jsonl`
- **Output:** `data/interim/critiques_v1.jsonl`
- **Command:**
  ```bash
  python -m pocketguide.data_generation.generate_critiques \
    --drafts data/interim/drafts_v1.jsonl \
    --out_dir data/interim
  ```

### 4. Dataset (join + refine with teacher)

- **Input:** plan + drafts + critiques
- **Output:** `data/processed/dataset_v1.jsonl` (and rejected JSONL)
- **Command:**
  ```bash
  python -m pocketguide.data_generation.generate_dataset_v1 \
    --plan data/interim/prompt_plan_v1.jsonl \
    --drafts data/interim/drafts_v1.jsonl \
    --critiques data/interim/critiques_v1.jsonl \
    --out_dir data/processed \
    --gating_mode lenient \
    --resume
  ```

### 5. (Optional) QA / clean dataset

- **Input:** `data/processed/dataset_v1.jsonl`
- **Output:** `data/processed/dataset_v1_clean.jsonl`
- **Command:**
  ```bash
  python -m pocketguide.data_generation.qa_pipeline_v1 \
    --in_path data/processed/dataset_v1.jsonl \
    --out_clean data/processed/dataset_v1_clean.jsonl
  ```
- Use the clean file as input to the next step if you run this.

### 6. Split into train / val / test

- **Input:** `data/processed/dataset_v1.jsonl` **or** `data/processed/dataset_v1_clean.jsonl`
- **Output:** `data/processed/splits/v1/train.jsonl`, `val.jsonl`, `test.jsonl`
- **Command:**
  ```bash
  python -m pocketguide.data_generation.split_dataset_v1 \
    --in_path data/processed/dataset_v1.jsonl \
    --out_dir data/processed/splits/v1 \
    --seed 42 \
    --train_frac 0.8 --val_frac 0.1 --test_frac 0.1
  ```
  (Use `dataset_v1_clean.jsonl` if you ran step 5.)

### 7. Prepare SFT format

- **Input:** `data/processed/splits/v1/train.jsonl`, `val.jsonl`
- **Output:** `data/processed/sft/v1/train_sft.jsonl`, `val_sft.jsonl`, and `eval/suites/fixed20_v1.jsonl`
- **Command:**
  ```bash
  python -m pocketguide.train.prepare_sft_data \
    --splits_dir data/processed/splits/v1 \
    --out_dir data/processed/sft/v1 \
    --seed 42 \
    --fixed_prompts_out eval/suites/fixed20_v1.jsonl
  ```

After step 7, use `train_sft.jsonl` and `val_sft.jsonl` for full training (and for Lightning).

---

## Why you only had 2+1

- Steps 1–6 were never run with the **real** dataset (or only minimal files were created for tests).
- So `data/processed/splits/v1/` only had 2 train + 1 val row.
- `prepare_sft_data` only converts what’s in those files → hence 2+1 SFT records.

Once you run steps 2 → 3 → 4 → 6 → 7 (with step 5 optional), you get full splits and full SFT data.

---

## Quick reference: from dataset JSONL to SFT

If you **already have** a full `dataset_v1.jsonl` (or `dataset_v1_clean.jsonl`):

```bash
# Split
python -m pocketguide.data_generation.split_dataset_v1 \
  --in_path data/processed/dataset_v1.jsonl \
  --out_dir data/processed/splits/v1 \
  --seed 42

# SFT format
python -m pocketguide.train.prepare_sft_data \
  --splits_dir data/processed/splits/v1 \
  --out_dir data/processed/sft/v1 \
  --seed 42
```

Then point training config at `data/processed/sft/v1/train_sft.jsonl` and `val_sft.jsonl`.

---

## Full data pipeline (all prompts)

To generate data for **all** prompts in the plan (e.g. 120 from the spec), run the full pipeline. This uses `data/interim/` and `data/processed/` (not the sample dirs).

**Prerequisites:**

- `configs/teacher.yaml`: `runtime.dry_run: false`
- `.env`: `OPENROUTER_API_KEY` set (for drafts, critiques, and refinement)
- Expect **many API calls** and runtime (e.g. 120 drafts + 120 critiques + up to 120 refinements)

**One command (plan → drafts → critiques → dataset → split → SFT):**

```bash
make data-full
```

This runs: `make data` → `make drafts` → `make critiques` → `make dataset` → `make split` → `make prepare-sft`. Outputs:

- `data/interim/drafts_v1.jsonl`, `critiques_v1.jsonl`
- `data/processed/dataset_v1.jsonl`, `dataset_v1_rejected.jsonl`
- `data/processed/splits/v1/train.jsonl`, `val.jsonl`, `test.jsonl`
- `data/processed/sft/v1/train_sft.jsonl`, `val_sft.jsonl`
- `eval/suites/fixed20_v1.jsonl`

**Resume:** `make dataset` uses `--resume`, so you can re-run `make data-full` and it will skip already-processed IDs. For a clean full run, remove outputs first (e.g. `make clean-data` or remove the files above) and then run from the step you need.

---

## Batch pipeline (run in chunks, then resume)

To process data in batches (e.g. 20 at a time), check for problems, then continue:

**1. First batch (e.g. 20 items):**

```bash
make data
make pipeline-batch
```

This runs: `data` → `drafts-batch` (next 20 drafts, resume) → `critiques-batch` (next 20 critiques, resume) → `dataset-batch` (next 20 samples, resume). Default batch size is 20; override with `BATCH_SIZE=30` if you like.

**2. Check outputs:**

- `data/interim/drafts_v1.jsonl`, `critiques_v1.jsonl` (e.g. 20 lines each)
- `data/processed/dataset_v1.jsonl` (e.g. up to 20 accepted)
- `data/processed/dataset_v1_stats.json` (accepted/rejected counts)

**3. Next batch (next 20):**

```bash
make pipeline-batch
```

Each run adds the **next** `BATCH_SIZE` items (drafts and critiques limits are computed as “current count + BATCH_SIZE” with `--resume`; dataset uses `--limit BATCH_SIZE --resume`).

**4. When all batches are done:**

```bash
make split
make prepare-sft
```

Then train with `make train-run` or Lightning.

**Per-step batches (optional):**

- `make drafts-batch` — next BATCH_SIZE drafts only
- `make critiques-batch` — next BATCH_SIZE critiques only  
- `make dataset-batch` — next BATCH_SIZE joined samples only  

Use these if you want to run or debug one stage at a time.

---

## After data: Lightning.ai (clone + train)

Once you have SFT data locally (`data/processed/sft/v1/train_sft.jsonl`, `val_sft.jsonl`), you can train on Lightning.ai as follows.

### 1. Push code and data

- **Option A – Git only:** Commit and push the repo. On Lightning, clone the repo. You still need to get the SFT data onto the Lightning run (e.g. upload `data/processed/sft/v1/` as a dataset or mount it).
- **Option B – Git + data elsewhere:** Push the repo to GitHub. Upload `data/processed/sft/v1/` to cloud storage (e.g. S3, GCS) or use Lightning Data from a URL, then point the training config at that path.

### 2. Clone on Lightning

- Create a Lightning Studio or a Run.
- Clone your GitHub repo, e.g.:
  ```bash
  git clone https://github.com/YOUR_ORG/pocket-guide.git
  cd pocket-guide
  ```

### 3. Environment

- Create a venv and install the project (as in the repo README), or use the project’s `Makefile`:
  ```bash
  make env
  source .venv/bin/activate
  ```
- Ensure `configs/train_lora.yaml` (or your training config) points to the correct data paths. For runs on Lightning, that is usually the path where you placed the SFT data (e.g. `data/processed/sft/v1/` if you cloned a repo that includes it, or a mounted/dataset path).

### 4. Run training

- From the repo root:
  ```bash
  python -m pocketguide.train.train --config configs/train_lora.yaml
  ```
  Or, if using the Makefile: `make train-run`.

### 5. Config and data paths

- Training config is in `configs/train_lora.yaml`. Set `data.train_path` and `data.val_path` (or the equivalent in your config) to your `train_sft.jsonl` and `val_sft.jsonl` paths on the Lightning machine.
- If SFT data is not in the repo, configure the app to read from the mounted or downloaded path (e.g. a Lightning Data mount or a path you populated from cloud storage).
