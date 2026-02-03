# Training v2 on Lightning.ai (Lesson 7.3)

Everything syncs via **GitHub**. Lightning runs your repo; no Makefile, no local venv. You run the **same Python entrypoint** as locally, with flags.

---

## 1. Before: On your machine (push what Lightning needs)

### 1.1 Check .gitignore

The repo is set up so these are **not** ignored (they will be pushed):

- **Code:** `src/`, `configs/`, `pyproject.toml`, etc. (already tracked)
- **V2 data (now tracked):**
  - `data/processed/dataset_v2.jsonl`
  - `data/processed/splits/v2/` (train.jsonl, val.jsonl, test.jsonl)
  - `data/processed/sft/v2/` (train_sft.jsonl, val_sft.jsonl)

**Ignored (correct):** `runs/*/`, `.env`, `data/interim/*`, raw drafts/critiques, so they do **not** get pushed.

### 1.2 Create v2 splits and SFT (if not done yet)

On your machine (with venv / `make`):

```bash
make split-v2
make prepare-sft-v2
```

This creates:

- `data/processed/splits/v2/train.jsonl`, `val.jsonl`, `test.jsonl`
- `data/processed/sft/v2/train_sft.jsonl`, `val_sft.jsonl`

### 1.3 Commit and push to GitHub

```bash
git add data/processed/dataset_v2.jsonl
git add data/processed/splits/v2/
git add data/processed/sft/v2/
git add configs/train_lora_v2.yaml
git add .gitignore
# add any other changed files
git status   # double-check
git commit -m "Add v2 data and config for Lightning training"
git push origin main
```

Lightning will use whatever is on `main` (or the branch you tell it to use).

---

## 2. On Lightning.ai: Run training

No Makefile, no venv. Use `pip` and `python` with the **same module and flags** you use locally.

### 2.1 Open your Lightning project

- Use the repo that’s connected to GitHub (so it has the v2 data and configs).

### 2.2 Install the package

In the Lightning shell or “Run” command:

```bash
cd /path/to/pocket-guide   # or wherever the repo is mounted
pip install -e .
```

(If you use optional deps: `pip install -e ".[dev]"`.)

### 2.3 Run training (single A100)

You do **not** need `accelerate launch` for a single GPU. The script already uses `Accelerator()`; it will use the one available GPU.

Run:

```bash
python -m pocketguide.train.train --config configs/train_lora_v2.yaml
```

Optional:

- Limit steps (e.g. smoke run):  
  `python -m pocketguide.train.train --config configs/train_lora_v2.yaml --max_steps 100`
- Pin run ID:  
  `python -m pocketguide.train.train --config configs/train_lora_v2.yaml --run_id my-v2-run`

So: **same entrypoint as local** (`python -m pocketguide.train.train`), **same flags**; only the environment (Lightning A100) is different.

### 2.4 Where outputs go

Training writes under the project directory, e.g.:

- `runs/train/<run_id>-v2/config.yaml`
- `runs/train/<run_id>-v2/meta.json`
- `runs/train/<run_id>-v2/adapter/`   (saved LoRA weights)
- `runs/train/<run_id>-v2/train_metrics.json`

Because `runs/*/` is in `.gitignore`, these are **not** pushed back to GitHub; they stay on Lightning (and in any Lightning-provided storage/artifacts you use).

---

## 3. After training: Get the adapter and metrics

- **From Lightning UI:** Use “Artifacts” or “Files” (or whatever your project uses) to download the run directory, e.g. `runs/train/<run_id>-v2/`, so you get `adapter/` and `train_metrics.json`.
- **Or** configure your Lightning app to copy `runs/train/<run_id>-v2/` to a cloud bucket or to your machine.

You do **not** need to change .gitignore for run outputs; they are meant to stay off GitHub and be retrieved from Lightning.

---

## Quick reference

| Step        | Where      | Command |
|------------|------------|--------|
| Split v2   | Local      | `make split-v2` |
| SFT v2     | Local      | `make prepare-sft-v2` |
| Push       | Local      | `git add ... && git commit && git push` |
| Install    | Lightning  | `pip install -e .` |
| Train v2   | Lightning  | `python -m pocketguide.train.train --config configs/train_lora_v2.yaml` |
| Get output | Lightning  | Download `runs/train/<run_id>-v2/` (adapter + train_metrics.json) |

**Why no `accelerate launch`?**  
For a single A100, `python -m pocketguide.train.train ...` is enough. The script uses `Accelerator()` internally; `accelerate launch` is only needed if you use a custom Accelerate config or multi-GPU. Same script, same flags, different environment.
