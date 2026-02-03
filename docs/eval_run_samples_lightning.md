# Run evaluation (run_samples) on Lightning.ai

Evaluation uses the **training run folder** (adapter), not GGUF. Same flow for v1 and v2.

- **v1 run folder:** `runs/train/20260131_224510`
- **v2 run folder:** `runs/train/20260203_013228-v2`

---

## On Lightning.ai (recommended: avoid OOM)

1. **Open your Lightning project** (repo synced from GitHub; v2 run folder must be present — e.g. you trained v2 there, or you uploaded `runs/train/20260203_013228-v2`).

2. **Install the package:**
   ```bash
   cd /path/to/pocket-guide
   pip install -e .
   ```

3. **Run evaluation for v2** (base + finetuned inference on fixed20, then comparison metrics):
   ```bash
   python -m pocketguide.train.run_samples \
     --run_dir runs/train/20260203_013228-v2 \
     --prompts eval/suites/fixed20_v1.jsonl
   ```

   For v1 (same command, different run dir):
   ```bash
   python -m pocketguide.train.run_samples \
     --run_dir runs/train/20260131_224510 \
     --prompts eval/suites/fixed20_v1.jsonl
   ```

4. **Outputs** (under the run folder):
   - `runs/train/20260203_013228-v2/samples/base_outputs.jsonl`
   - `runs/train/20260203_013228-v2/samples/finetuned_outputs.jsonl`
   - `runs/train/20260203_013228-v2/samples/comparison_metrics.json`

5. **Download** the `samples/` folder (or at least `comparison_metrics.json`) from Lightning to your Mac if you want it locally.

---

## Optional: add local_regression suite

If you want the same suites as local eval (fixed20 + local_regression), run **twice** (run_samples takes a single prompts file), or concatenate the two JSONL files into one and pass that path. Default Make target uses only fixed20:

```bash
python -m pocketguide.train.run_samples \
  --run_dir runs/train/20260203_013228-v2 \
  --prompts eval/suites/fixed20_v1.jsonl
```

---

## Summary

| Step        | Command |
|------------|--------|
| Install    | `pip install -e .` |
| Eval v2    | `python -m pocketguide.train.run_samples --run_dir runs/train/20260203_013228-v2 --prompts eval/suites/fixed20_v1.jsonl` |
| Outputs    | `runs/train/20260203_013228-v2/samples/` |

No GGUF, no llama.cpp — only the adapter in the training run folder.
