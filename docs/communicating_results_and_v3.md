# Is significant improvement attainable? How to communicate results

## Is a significant (not incremental) improvement attainable?

**Yes, with the right fixes.** v1 and v2 got 0% schema_valid for concrete, fixable reasons:

1. **Too little training** — v2 had only **10 optimizer steps** (1 epoch). The model barely saw the format.
2. **Truncation during training** — v2 used **max_seq_len: 1024**. Long responses were cut; the **end** of the JSON (verification_steps, payload_type, payload) was often dropped. Loss was never computed on those tokens, so the model never learned to output them.
3. **Incomplete SFT examples** — If any training record was missing verification_steps or payload_type, the model was trained on incomplete structure.

**v3 addresses all three:**

- **More training:** 5 epochs, grad_accum 8 → ~100 optimizer steps (10× v2). The model sees each example multiple times.
- **Full envelope in loss:** max_seq_len 2048 so the full response (including verification_steps, payload_type, payload) is in the loss. If 2048 OOMs on 24GB, use 1536.
- **Consistent SFT data:** prepare_sft_data now ensures every response has all 7 envelope keys (fills missing with defaults). System instruction explicitly reminds the model to include verification_steps and payload_type.
- **Larger LoRA:** r=32, alpha=64 (v2 used r=16) so the adapter has more capacity to learn the envelope structure.

So **yes, a meaningful jump in schema_valid_rate is attainable** — not guaranteed, but the main failure modes are addressed. If v3 still shows low schema_valid, the next levers are: even more epochs, loss weighting on the last N tokens of the response, or slightly more/better data.

---

## How to communicate results to recruiters and on your portfolio/README

**Regardless of whether v3 hits high schema_valid or not**, you can present this project clearly and professionally.

### 1. Frame the project as evaluation- and iteration-driven

- **What it is:** A travel-assistant LLM with a strict output contract (envelope + payload schemas), offline-capable, with reproducible evaluation.
- **What you did:** Data pipeline (teacher → QA → splits → SFT), LoRA fine-tuning, and a fixed benchmark (run_samples on fixed20) to compare base vs finetuned and v1 vs v2 vs v3.
- **What you learned:** v1/v2 had 0% schema compliance; you identified causes (too few steps, truncation, data consistency) and iterated (v3: more epochs, longer context, envelope normalization, larger LoRA).

### 2. Emphasize process and metrics, not just “did it work?”

- **Metrics to report:** parse_success_rate, schema_valid_rate, required_field_presence_rate, uncertainty_marker_presence_rate (and optionally latency). These are in `comparison_metrics.json` for each run.
- **Storyline:** “V1 finetuned improved parse vs base but 0% schema_valid. We diagnosed (debug logs, schema_failures_debug.jsonl) and found truncation + too few steps. V3 increased training and context length and normalized SFT data; we report before/after metrics.”
- **If v3 is still low:** “V3 improved X% over v2; remaining gaps are [e.g. long-tail payload types]. Next steps: [more data, loss masking, or curriculum].” That still shows debugging, iteration, and clear metrics.

### 3. What to put in README / portfolio

- **One-line:** “Travel-assistant LLM with structured JSON output, LoRA fine-tuning, and evaluation-driven iteration (parse/schema metrics on a fixed benchmark).”
- **Results table (fill in after runs):**

  | Run   | parse_success_rate | schema_valid_rate | Notes                          |
  |-------|---------------------|-------------------|--------------------------------|
  | Base  | —                   | —                 | No adapter                     |
  | v1 FT | 0.80                | 0.00              | 1 epoch, max_seq_len 1024      |
  | v2 FT | 1.00                | 0.00              | LR tweak, 1 epoch, 1024        |
  | v3 FT | _TBD_               | _TBD_             | 5 epochs, 2048, envelope norm |

- **“Key learnings” or “Iteration”:** Short bullets: e.g. “Identified training length and sequence truncation as main causes of 0% schema compliance; v3 adds more epochs, longer context, and SFT envelope normalization.”

### 4. For recruiters / interviews

- **Technical:** “I built a pipeline from synthetic data generation to LoRA fine-tuning and evaluation. We measure parse and schema compliance on a fixed prompt set. When schema_valid was 0%, I added debug output and traced it to truncation and too few optimizer steps; v3 fixes those and we compare metrics.”
- **Outcome:** Either “V3 achieved X% schema compliance, showing the model can learn the contract” or “V3 improved over v2; remaining work is Y.” Both are valid engineering outcomes.
- **Repo:** Point them to README (one-liner + results table), `docs/v1_v2_more_story.md`, and `docs/communicating_results_and_v3.md` for the story and how you communicate results.

### 5. Bottom line

- **Success case:** v3 shows a clear gain (e.g. 20–80%+ schema_valid). Lead with “Evaluation-driven fine-tuning; we iterated on training length and context and achieved X% schema compliance.”
- **Partial success:** v3 is better than v2 but still low. Lead with “Structured-output LLM with rigorous evaluation; we identified and fixed truncation and training length; current metrics and next steps are documented.”
- **No change:** If v3 is still 0%, you still have a clear story: “We have a fixed benchmark and debug artifacts; we’ve ruled out X and Y; next steps are Z.” That’s still valuable for portfolios and interviews.

Use the same metrics and the same benchmark (run_samples on fixed20) for v1, v2, and v3 so the comparison is apples-to-apples.
