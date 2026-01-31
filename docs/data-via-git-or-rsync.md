# Getting training data onto Lightning: Git vs rsync

**Recommendation: use Git.** Your training data is small (~2 MB). Tracking it in the repo means one clone on Lightning and you’re done. Rsync is better if you keep data out of Git and sync it manually.

---

## Option A: Git (recommended)

Only the paths needed for training are tracked; the rest of `data/` stays ignored.

### 1. Update .gitignore (already done)

`.gitignore` is set up so that:

- `data/` is ignored by default.
- These are **not** ignored (and will be committed):
  - `data/processed/dataset_v1.jsonl`
  - `data/processed/splits/v1/*` (train/val/test JSONL + manifest/stats)
  - `data/processed/sft/v1/*` (train_sft.jsonl, val_sft.jsonl)
  - `data/benchmarks/v1/*` (test prompts from split)

Everything else under `data/` (e.g. `data/interim/`, raw specs, etc.) stays ignored.

### 2. Commit and push

From the repo root:

```bash
git add .gitignore
git add data/processed/dataset_v1.jsonl
git add data/processed/splits/v1/
git add data/processed/sft/v1/
git add data/benchmarks/v1/
git status   # confirm only intended files are staged
git commit -m "Track training data (splits + SFT) for Lightning"
git push origin main
```

### 3. On Lightning

Clone and run:

```bash
git clone https://github.com/YOUR_ORG/pocket-guide.git
cd pocket-guide
# data/processed/splits/v1 and data/processed/sft/v1 are already there
make env && source .venv/bin/activate
huggingface-cli login   # if using Llama-2
make train-run
```

No extra data copy step.

### 4. Updating data later

After re-running `make split` and `make prepare-sft` locally:

```bash
git add data/processed/dataset_v1.jsonl data/processed/splits/v1/ data/processed/sft/v1/ data/benchmarks/v1/
git commit -m "Update training data (splits + SFT)"
git push origin main
```

On Lightning, `git pull` and re-run training.

---

## Option B: Rsync (data stays ignored)

Keep `data/` fully ignored. On Lightning, clone the repo then sync only the data folder from your machine.

### 1. Leave .gitignore as “data/” only

Revert to ignoring all of `data/`:

```gitignore
data/
!data/.gitkeep
```

(Do not add the `!data/processed/...` exceptions.)

### 2. On Lightning: clone and get SSH/upload info

- Create a Run or Studio and clone the repo.
- Note how to send files **to** that machine (e.g. SSH host and user, or Lightning’s “Upload” / Data UI if available).

### 3. From your laptop: rsync data into the run

If Lightning gives you SSH access (e.g. `ssh user@<run-ip>`):

```bash
# From your laptop, repo root
rsync -avz --progress \
  data/processed/dataset_v1.jsonl \
  data/processed/splits/v1/ \
  data/processed/sft/v1/ \
  data/benchmarks/v1/ \
  user@<LIGHTNING_SSH_HOST>:~/pocket-guide/data/
```

You may need to create the target dirs first on Lightning:

```bash
ssh user@<LIGHTNING_SSH_HOST> "mkdir -p ~/pocket-guide/data/processed/splits/v1 ~/pocket-guide/data/processed/sft/v1 ~/pocket-guide/data/benchmarks/v1"
```

Then run the `rsync` above. Paths on the right should match where you cloned the repo (e.g. `~/pocket-guide`).

### 4. If Lightning has no SSH: zip + upload

From your laptop:

```bash
mkdir -p /tmp/pocket-guide-data
cp data/processed/dataset_v1.jsonl /tmp/pocket-guide-data/
cp -r data/processed/splits/v1 /tmp/pocket-guide-data/splits_v1
cp -r data/processed/sft/v1 /tmp/pocket-guide-data/sft_v1
cp -r data/benchmarks/v1 /tmp/pocket-guide-data/benchmarks_v1
cd /tmp && zip -r pocket-guide-data.zip pocket-guide-data
```

Upload `pocket-guide-data.zip` via Lightning’s UI (e.g. Data or file upload). On the Lightning machine, after clone:

```bash
cd pocket-guide
unzip /path/to/pocket-guide-data.zip
mv pocket-guide-data/splits_v1 data/processed/splits/v1
mv pocket-guide-data/sft_v1 data/processed/sft/v1
mv pocket-guide-data/benchmarks_v1 data/benchmarks/v1
cp pocket-guide-data/dataset_v1.jsonl data/processed/
```

Then run training as usual.

---

## Summary

| | Git | Rsync |
|---|-----|--------|
| **Setup** | Commit + push once | Keep data ignored; sync or zip each time |
| **On Lightning** | `git clone` and train | Clone, then rsync/upload + unpack |
| **Updates** | `git add` + push, then `git pull` on Lightning | Re-rsync or re-upload zip |
| **Repo size** | +~2 MB | Unchanged |

Use **Git** unless you have a reason to keep data out of the repo (e.g. very large or private elsewhere); then use **rsync** (or zip + upload) as in Option B.
