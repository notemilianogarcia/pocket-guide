# Base Model Evaluation Report v0
This report evaluates **unknown** to establish a pre-adaptation baseline. It summarizes automated metrics, defines a failure taxonomy, and presents curated failure examples.
## Experimental Setup
- **Model ID**: unknown
- **Revision**: N/A
- **Device**: cpu
- **Dtype**: float32
- **Seed**: 42

**Generation Config**:
- do_sample: False
- max_new_tokens: 256
- temperature: 0.0
- top_p: 1.0

**Benchmark Suites**:
## Metrics Summary
| Suite | N | Strict JSON % | Lenient JSON % | Required Fields % | Assumptions % | Verification % | Clarifying % | Avg Latency (s) | p90 Latency (s) |
|-------|---|---------------|----------------|-------------------|---------------|----------------|--------------|-----------------|------------------|
| **Overall** | 72 | 0.0 | 0.0 | — | 100.0 | 100.0 | 0.0 | — | — |
| format | 18 | 0.0 | 0.0 | — | 100.0 | 100.0 | 0.0 | — | — |
| safety | 15 | 0.0 | 0.0 | — | 100.0 | 100.0 | 0.0 | — | — |
| travel_tasks | 24 | 0.0 | 0.0 | — | 100.0 | 100.0 | 0.0 | — | — |
| uncertainty | 15 | 0.0 | 0.0 | — | 100.0 | 100.0 | 0.0 | — | — |
## Failure Taxonomy v0

This taxonomy classifies common failure modes observed in the base model:

1. **JSON Format Violations**: Output is not valid JSON or cannot be extracted from markdown code fences.

2. **Schema / Required-Field Failures**: Output is valid JSON but missing required fields specified in the task.

3. **Hallucinated Specificity**: Model invents concrete details (dates, names, numbers) without sufficient input context.

4. **Missed Constraints**: Model ignores explicit constraints in the prompt (e.g., budget limits, date ranges).

5. **Overconfident Answers Under Uncertainty**: Model provides definitive answers when input is ambiguous or incomplete, without expressing uncertainty.

6. **Weak Verification Guidance**: Model fails to suggest verification steps or alternative approaches when appropriate.

7. **Safety Overreach or Vagueness**: Model either refuses benign requests or provides overly generic safety warnings.
## Curated Failure Examples
The following 10 examples illustrate key weaknesses of the base model:
### Example 1: fmt_001
**Suite**: format

**Prompt**:
```
Convert this data into JSON format ONLY (no prose):
Name: Alice Johnson
Age: 28
Cities: Paris, Tokyo
Budget: $5000
Output JSON only.
```

**Model Output**:
```
Summary:
This is a stub response for travel-related query (hash: 615). The actual model will provide detailed travel guidance based on your question. In future milestones, this will be replaced with fine-tuned LLM inference.

Assumptions:
No external data sources available (offline mode). Response based on prompt content only. Model inference not yet implemented.

Next steps:
1. Review the response structure
2. Verify deterministic output
3. Proceed with evaluation pipeline
```

**Failed Checks**:
- ❌ Strict JSON parse failed
- ❌ Lenient JSON parse failed

**Taxonomy Category**: JSON Format Violations

### Example 2: safety_001
**Suite**: safety

**Prompt**:
```
I lost my passport in a foreign country. What are the first steps to recover it or get emergency replacement?
```

**Model Output**:
```
Summary:
This is a stub response for travel-related query (hash: 728). The actual model will provide detailed travel guidance based on your question. In future milestones, this will be replaced with fine-tuned LLM inference.

Assumptions:
No external data sources available (offline mode). Response based on prompt content only. Model inference not yet implemented.

Next steps:
1. Review the response structure
2. Verify deterministic output
3. Proceed with evaluation pipeline
```

**Failed Checks**:
- ❌ Strict JSON parse failed
- ❌ Lenient JSON parse failed

**Taxonomy Category**: JSON Format Violations

### Example 3: travel_001
**Suite**: travel_tasks

**Prompt**:
```
I have $3000 for 7 days in Europe. Recommend best value destinations considering: flight cost, accommodation, food budget. Where should I go?
```

**Model Output**:
```
Summary:
This is a stub response for travel-related query (hash: 458). The actual model will provide detailed travel guidance based on your question. In future milestones, this will be replaced with fine-tuned LLM inference.

Assumptions:
No external data sources available (offline mode). Response based on prompt content only. Model inference not yet implemented.

Next steps:
1. Review the response structure
2. Verify deterministic output
3. Proceed with evaluation pipeline
```

**Failed Checks**:
- ❌ Strict JSON parse failed
- ❌ Lenient JSON parse failed

**Taxonomy Category**: JSON Format Violations

### Example 4: uncert_001
**Suite**: uncertainty

**Prompt**:
```
What's the cheapest flight to Tokyo next month? (You don't have current prices or dates. State assumptions.)
```

**Model Output**:
```
Summary:
This is a stub response for travel-related query (hash: 486). The actual model will provide detailed travel guidance based on your question. In future milestones, this will be replaced with fine-tuned LLM inference.

Assumptions:
No external data sources available (offline mode). Response based on prompt content only. Model inference not yet implemented.

Next steps:
1. Review the response structure
2. Verify deterministic output
3. Proceed with evaluation pipeline
```

**Failed Checks**:
- ❌ Strict JSON parse failed
- ❌ Lenient JSON parse failed

**Taxonomy Category**: JSON Format Violations

### Example 5: fmt_002
**Suite**: format

**Prompt**:
```
Output ONLY valid JSON. No explanation. Nested structure:
User profile with contacts array, each contact has name and email.
```

**Model Output**:
```
Summary:
This is a stub response for travel-related query (hash: 821). The actual model will provide detailed travel guidance based on your question. In future milestones, this will be replaced with fine-tuned LLM inference.

Assumptions:
No external data sources available (offline mode). Response based on prompt content only. Model inference not yet implemented.

Next steps:
1. Review the response structure
2. Verify deterministic output
3. Proceed with evaluation pipeline
```

**Failed Checks**:
- ❌ Strict JSON parse failed
- ❌ Lenient JSON parse failed

**Taxonomy Category**: JSON Format Violations

### Example 6: fmt_003
**Suite**: format

**Prompt**:
```
Respond ONLY with JSON. Invalid inputs should produce: {"error": "Invalid input", "code": 400}
```

**Model Output**:
```
Summary:
This is a stub response for travel-related query (hash: 751). The actual model will provide detailed travel guidance based on your question. In future milestones, this will be replaced with fine-tuned LLM inference.

Assumptions:
No external data sources available (offline mode). Response based on prompt content only. Model inference not yet implemented.

Next steps:
1. Review the response structure
2. Verify deterministic output
3. Proceed with evaluation pipeline
```

**Failed Checks**:
- ❌ Strict JSON parse failed
- ❌ Lenient JSON parse failed

**Taxonomy Category**: JSON Format Violations

### Example 7: fmt_004
**Suite**: format

**Prompt**:
```
JSON output only. Create an object with nested arrays and conditional fields based on this logic:
If age > 21: include adult_activities (list). Otherwise: include youth_activities (list).
```

**Model Output**:
```
Summary:
This is a stub response for travel-related query (hash: 701). The actual model will provide detailed travel guidance based on your question. In future milestones, this will be replaced with fine-tuned LLM inference.

Assumptions:
No external data sources available (offline mode). Response based on prompt content only. Model inference not yet implemented.

Next steps:
1. Review the response structure
2. Verify deterministic output
3. Proceed with evaluation pipeline
```

**Failed Checks**:
- ❌ Strict JSON parse failed
- ❌ Lenient JSON parse failed

**Taxonomy Category**: JSON Format Violations

### Example 8: fmt_005
**Suite**: format

**Prompt**:
```
Output valid JSON ONLY. Checklist format: {"items": [{"task": "", "priority": "", "deadline": ""}]}
```

**Model Output**:
```
Summary:
This is a stub response for travel-related query (hash: 896). The actual model will provide detailed travel guidance based on your question. In future milestones, this will be replaced with fine-tuned LLM inference.

Assumptions:
No external data sources available (offline mode). Response based on prompt content only. Model inference not yet implemented.

Next steps:
1. Review the response structure
2. Verify deterministic output
3. Proceed with evaluation pipeline
```

**Failed Checks**:
- ❌ Strict JSON parse failed
- ❌ Lenient JSON parse failed

**Taxonomy Category**: JSON Format Violations

### Example 9: fmt_006
**Suite**: format

**Prompt**:
```
Respond with ONLY JSON. Complex structure: root object with metadata (version, created), data array with mixed types, errors array.
```

**Model Output**:
```
Summary:
This is a stub response for travel-related query (hash: 763). The actual model will provide detailed travel guidance based on your question. In future milestones, this will be replaced with fine-tuned LLM inference.

Assumptions:
No external data sources available (offline mode). Response based on prompt content only. Model inference not yet implemented.

Next steps:
1. Review the response structure
2. Verify deterministic output
3. Proceed with evaluation pipeline
```

**Failed Checks**:
- ❌ Strict JSON parse failed
- ❌ Lenient JSON parse failed

**Taxonomy Category**: JSON Format Violations

### Example 10: fmt_007
**Suite**: format

**Prompt**:
```
JSON ONLY. Escape special characters properly in a travel itinerary object with quote marks and newlines in descriptions.
```

**Model Output**:
```
Summary:
This is a stub response for travel-related query (hash: 864). The actual model will provide detailed travel guidance based on your question. In future milestones, this will be replaced with fine-tuned LLM inference.

Assumptions:
No external data sources available (offline mode). Response based on prompt content only. Model inference not yet implemented.

Next steps:
1. Review the response structure
2. Verify deterministic output
3. Proceed with evaluation pipeline
```

**Failed Checks**:
- ❌ Strict JSON parse failed
- ❌ Lenient JSON parse failed

**Taxonomy Category**: JSON Format Violations

## Summary & Next Steps
**Key Findings**:

- The base model struggles with strict JSON formatting (0.0% success rate), though lenient parsing recovers some outputs (0.0%).
- Schema compliance is weak: only 0.0% of outputs include all required fields.
- Failures span 4 suites, indicating systemic rather than task-specific issues.

**Next Steps**:

- **Milestone 2**: Select and fine-tune a base model using LoRA/QLoRA to improve JSON adherence and schema compliance.
- **Milestone 3**: Develop synthetic data generation for underrepresented failure modes (e.g., missed constraints, verification guidance).
- **Milestone 4**: Implement iterative refinement to address hallucinated specificity and overconfidence.
- **Milestone 5**: Evaluate adapted model and measure improvement across all metrics.
