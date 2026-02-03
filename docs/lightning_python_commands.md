# Lightning.ai: Python commands only (no make)

All training and evaluation run on Lightning. Use these commands from the **project root** (where `configs/` and `data/` live). Install once: `pip install -e .`

---

## 1. Prepare data for v3 (run once before v3 training)

If you don’t have v2 splits yet:

```bash
python3 -m pocketguide.data_generation.split_dataset_v1 \
  --in_path data/processed/dataset_v2.jsonl \
  --out_dir data/processed/splits/v2 \
  --seed 42 \
  --train_frac 0.8 --val_frac 0.1 --test_frac 0.1
```

Then prepare SFT with envelope normalization (required for v3):

```bash
python3 -m pocketguide.train.prepare_sft_data \
  --splits_dir data/processed/splits/v2 \
  --out_dir data/processed/sft/v2 \
  --seed 42 \
  --fixed_prompts_out eval/suites/fixed20_v1.jsonl
```

This writes `data/processed/sft/v2/train_sft.jsonl`, `val_sft.jsonl`, and (if you want) updates the fixed suite. Use `eval/suites/fixed20_v1.jsonl` for evaluation so v1/v2/v3 are comparable.

---

## 2. Train v3

From project root:

```bash
python3 -m pocketguide.train.train --config configs/train_lora_v3.yaml
```

Optional: validate config and data without loading the model:

```bash
python3 -m pocketguide.train.train --config configs/train_lora_v3.yaml --dry_run
```

When training finishes, the run dir will be something like `runs/train/20260204_123456-v3`. Use that path in the next step.

---

## 3. Run evaluation (with debugging)

This runs **base** and **finetuned** on the fixed prompt suite, writes comparison metrics and **schema failure debug** (per-sample failures). Replace `<RUN_ID>` with your actual run folder name (e.g. `20260204_123456-v3`).

```bash
python3 -m pocketguide.train.run_samples \
  --run_dir runs/train/<RUN_ID> \
  --prompts eval/suites/fixed20_v1.jsonl
```

Example for a v3 run:

```bash
python3 -m pocketguide.train.run_samples \
  --run_dir runs/train/20260204_123456-v3 \
  --prompts eval/suites/fixed20_v1.jsonl
```

**Outputs (under `runs/train/<RUN_ID>/samples/`):**

- `base_outputs.jsonl` — base model outputs
- `finetuned_outputs.jsonl` — finetuned model outputs
- `comparison_metrics.json` — parse_success_rate, schema_valid_rate, etc.
- `schema_failures_debug.jsonl` — one line per schema failure (error_code, missing_fields, parsed_top_level_keys, schema_error_message) for debugging

Verbose logging is on by default; you’ll see per-sample `[schema fail]` lines when schema validation fails.

---

## Quick reference

| Step | Command |
|------|--------|
| Split v2 (if needed) | `python3 -m pocketguide.data_generation.split_dataset_v1 --in_path data/processed/dataset_v2.jsonl --out_dir data/processed/splits/v2 --seed 42 --train_frac 0.8 --val_frac 0.1 --test_frac 0.1` |
| Prepare SFT v2 (before v3) | `python3 -m pocketguide.train.prepare_sft_data --splits_dir data/processed/splits/v2 --out_dir data/processed/sft/v2 --seed 42 --fixed_prompts_out eval/suites/fixed20_v1.jsonl` |
| Train v3 | `python3 -m pocketguide.train.train --config configs/train_lora_v3.yaml` |
| Eval (after training) | `python3 -m pocketguide.train.run_samples --run_dir runs/train/<RUN_ID> --prompts eval/suites/fixed20_v1.jsonl` |

Run from the repo root. If you get “file not found”, check that `data/processed/dataset_v2.jsonl` and (after split) `data/processed/splits/v2/train.jsonl` exist.
