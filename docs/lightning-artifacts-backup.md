# What to copy from Lightning before shutting down

Before you turn off the Lightning machine, copy the **training run** so you can continue locally.

---

## What you need

**One folder:** the full run directory for your training run.

| On Lightning | Copy to local |
|--------------|----------------|
| `~/pocket-guide/runs/train/<run_id>/` | `pocket-guide/runs/train/<run_id>/` |

Example: if your run ID was `20260131_224510`, copy the whole folder:

- **From:** `~/pocket-guide/runs/train/20260131_224510/`
- **To (on your Mac):** `pocket-guide/runs/train/20260131_224510/`

That folder contains:

| Item | Purpose |
|------|--------|
| `adapter/` | LoRA weights (adapter_config.json, adapter_model.safetensors). **Required** for inference or further training. |
| `config.yaml` | Snapshot of the config used for the run. |
| `meta.json` | Run metadata (device, git commit, dataset hashes). |
| `train_metrics.json` | Loss curves and metrics (if written). |

You need **the whole folder** (including `adapter/`). Everything else in the repo (code, data) you already have locally or in GitHub.

---

## Option A: Download via Lightning UI (simplest)

1. On Lightning, in the file browser, go to `runs/train/`.
2. Right‑click the run folder (e.g. `20260131_224510`) → **Download** / **Zip and download** (if available).
3. On your Mac, unzip (if needed) and move the folder to your local repo:
   ```bash
   # e.g. after downloading 20260131_224510.zip to ~/Downloads
   unzip ~/Downloads/20260131_224510.zip -d /path/to/pocket-guide/runs/train/
   ```

---

## Option B: Zip on Lightning, then download

**On Lightning (terminal):**

```bash
cd ~/pocket-guide
zip -r run_20260131_224510.zip runs/train/20260131_224510
```

Then in Lightning’s file browser, download `run_20260131_224510.zip` to your Mac.

**On your Mac:**

```bash
cd /path/to/pocket-guide
unzip ~/Downloads/run_20260131_224510.zip
# This recreates runs/train/20260131_224510/
```

---

## Option C: GitHub (code only; not for adapter)

- **Code:** If you changed any code on Lightning, commit and push from Lightning so your local repo can `git pull`. The run folder is **not** in git (runs are in `.gitignore`), so the adapter is **not** on GitHub.
- **Adapter:** Do **not** commit the run folder to GitHub unless you want to use Git LFS for the `.safetensors` file. Prefer Option A or B for the run.

---

## After copying

On your Mac you should have:

```
pocket-guide/
  runs/
    train/
      20260131_224510/   # or whatever your run_id
        adapter/
          adapter_config.json
          adapter_model.safetensors
        config.yaml
        meta.json
        train_metrics.json
```

You can then run evaluation, inference, or report generation (5.4 / 5.5) locally pointing at this run dir.

---

## Checklist

- [ ] Copy `runs/train/<run_id>/` from Lightning to local `pocket-guide/runs/train/<run_id>/`.
- [ ] If you edited code on Lightning: push those changes to GitHub, then `git pull` on your Mac.
- [ ] Optionally download any other files you created on Lightning (e.g. extra scripts or notes).
