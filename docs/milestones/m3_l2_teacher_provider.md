# Teacher Provider Interface (Milestone 3, Lesson 3.2)

**Status**: ✅ COMPLETED

## Overview

This lesson implements a teacher provider interface with OpenRouter backend, enabling synthetic data generation using large language models with cost-controlled fallback chains.

## Implementation

### Core Components

#### 1. Base Interface ([base.py](src/pocketguide/teachers/base.py))
- `TeacherRequest`: Input dataclass with OpenAI-format messages, generation params
- `TeacherResponse`: Output dataclass with text, model, usage, timing, metadata
- `TeacherClient`: Abstract base interface for all teacher providers

#### 2. Error Types ([errors.py](src/pocketguide/teachers/errors.py))
- `TeacherAuthError`: 401/403 authentication failures (fail fast)
- `TeacherRateLimitError`: 429 rate limits (retry with backoff)
- `TeacherTransientError`: 5xx/timeout errors (retry with backoff)
- `TeacherBadRequestError`: 400 client errors (fail fast)

#### 3. OpenRouter Client ([openrouter.py](src/pocketguide/teachers/openrouter.py))
- OpenAI-compatible chat completions API
- Retry with exponential backoff + jitter
- Rate limiting (default: 15 RPM)
- Dry-run mode for testing without API calls
- Environment variable configuration:
  - `OPENROUTER_API_KEY`: Required for real API calls
  - `OPENROUTER_APP_NAME`: Optional app name for headers

#### 4. Router with Fallback ([router.py](src/pocketguide/teachers/router.py))
- Tries models in order from config
- Falls back on transient errors (429, 5xx, timeouts)
- Fails fast on auth/bad request errors
- `fallback_to_paid` flag to prevent accidental spending
- Tracks attempted models and fallback metadata

#### 5. Rate Limiter ([../utils/rate_limiter.py](src/pocketguide/utils/rate_limiter.py))
- Token bucket implementation
- Enforces requests-per-minute (RPM) limits
- Prevents hitting API rate limits

#### 6. Configuration ([../../configs/teacher.yaml](configs/teacher.yaml))
```yaml
models:
  - meta-llama/llama-3.3-70b-instruct:free  # Primary free model
  - deepseek/deepseek-chat-v3.1:free        # Secondary free model
  - openai/gpt-4o                           # Paid fallback

generation:
  temperature: 0.2
  top_p: 0.9
  max_tokens: 900
  seed: 42

runtime:
  dry_run: true           # Safe default: no API calls
  timeout_s: 60
  fallback_to_paid: true  # Set to false to prevent paid API usage

rate_limit:
  rpm: 15
  max_retries_per_model: 6
  backoff_base_s: 1.0
  backoff_max_s: 30.0

safety:
  max_requests: 500
  max_total_tokens: 2000000
```

## Features

### Environment Setup

See [docs/environment-setup.md](../environment-setup.md) for detailed instructions.

**Quick start:**
```bash
# Copy example and add your API key
cp .env.example .env
# Edit .env and add: OPENROUTER_API_KEY=sk-or-v1-...

# Test it works
python -m pocketguide.teachers.smoke --dry-run  # No key needed
python -m pocketguide.teachers.smoke --real     # Uses .env key
```

The system automatically loads API keys from `.env` (via `python-dotenv`). No need to manually export environment variables!

### Cost Control
1. **Dry-run mode**: Test without API calls (returns `"DRY_RUN"`)
2. **Free models first**: Try free models before paid ones
3. **Paid gating**: `fallback_to_paid=false` stops before paid models
4. **Model detection**: Identifies paid models by prefix (`openai/`, `anthropic/`, etc.) or lack of `:free` suffix

### Reliability
1. **Exponential backoff**: `base_s * 2^attempt` with jitter (±25%)
2. **Max retries**: Configurable per model (default: 6)
3. **Fail fast**: Auth and bad request errors skip retries
4. **Retry transient**: Rate limits and server errors retry with backoff

### Observability
- Request ID tracking
- Token usage stats
- Latency measurements
- Attempted models list
- Fallback occurrence flag

## Usage

### Dry-Run (No API Key Required)
```python
from pocketguide.teachers.base import TeacherRequest
from pocketguide.teachers.openrouter import OpenRouterTeacherClient
from pocketguide.teachers.router import TeacherRouterClient

# Create backend
backend = OpenRouterTeacherClient(
    model="meta-llama/llama-3.3-70b-instruct:free",
    dry_run=True  # No API calls
)

# Create router
router = TeacherRouterClient(
    backend=backend,
    models=[
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-chat-v3.1:free",
        "openai/gpt-4o"
    ],
    fallback_to_paid=False  # Stop before paid models
)

# Generate
request = TeacherRequest(
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.2,
    max_tokens=100
)

response = router.generate(request)
# response.text == "DRY_RUN"
```

### Real API Calls
```bash
export OPENROUTER_API_KEY=your-key-here
export OPENROUTER_APP_NAME=pocket-guide  # Optional
```

```python
backend = OpenRouterTeacherClient(
    model="meta-llama/llama-3.3-70b-instruct:free",
    dry_run=False,  # Enable real API calls
    rpm=15,
    max_retries=6
)

router = TeacherRouterClient(
    backend=backend,
    models=[
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-chat-v3.1:free",
        "openai/gpt-4o"
    ],
    fallback_to_paid=True  # Allow fallback to paid models
)

response = router.generate(request)
print(response.text)
print(response.raw["selected_model"])
```

### CLI Smoke Test
```bash
# Dry-run mode (default)
python -m pocketguide.teachers.smoke --dry-run

# Real API calls (requires OPENROUTER_API_KEY)
python -m pocketguide.teachers.smoke --real --fallback-paid

# Custom prompt
python -m pocketguide.teachers.smoke --dry-run --prompt "What is Python?"
```

## Testing

### Test Coverage
- ✅ Dry-run returns `"DRY_RUN"` without API call
- ✅ Successful generation with mocked response
- ✅ Auth error (401) fails fast without retries
- ✅ Bad request (400) fails fast without retries
- ✅ Rate limit (429) retries with exponential backoff
- ✅ Server error (5xx) retries with backoff
- ✅ Router tries first model first
- ✅ Router falls back on transient errors
- ✅ Router fails fast on auth/bad request
- ✅ Router stops before paid models when `fallback_to_paid=False`
- ✅ Router raises error when all models exhausted
- ✅ Paid model detection logic

### Run Tests
```bash
make test  # All tests (156 tests)
python -m pytest tests/test_teacher_router_openrouter.py -v  # Teacher tests only (14 tests)
```

## Dependencies

Added `httpx>=0.27.0` to [pyproject.toml](../../pyproject.toml) for HTTP client with:
- Timeout support
- Retry-friendly error handling
- Async/sync API (using sync for now)

## Files Created

```
src/pocketguide/
  teachers/
    __init__.py           # Package init
    base.py               # Base interface + dataclasses
    errors.py             # Typed exceptions
    openrouter.py         # OpenRouter client
    router.py             # Router with fallback chain
    smoke.py              # CLI smoke test
  utils/
    rate_limiter.py       # RPM rate limiter

configs/
  teacher.yaml            # Teacher configuration

tests/
  test_teacher_router_openrouter.py  # 14 comprehensive tests
```

## Next Steps (Milestone 3, Lesson 3.3)

1. **Batch generation CLI**: Generate all 120 prompts from `prompt_plan_v1.jsonl`
2. **Load config**: Parse `teacher.yaml` for model chain + generation params
3. **Progress tracking**: Show progress bar, retry counts, fallback occurrences
4. **Safety limits**: Enforce `max_requests` and `max_total_tokens`
5. **Output format**: Save to `generated_examples_v1.jsonl` with metadata
6. **Resume capability**: Skip already-generated examples on restart

## Design Decisions

### Why OpenRouter?
- Single API for multiple providers (Meta, DeepSeek, OpenAI)
- Free tier models available
- OpenAI-compatible API (easy integration)
- Cost-effective for experimentation

### Why Fallback Chain?
- **Reliability**: Free models may be rate-limited or unavailable
- **Cost control**: Try free models first, paid only as last resort
- **Flexibility**: Easy to reorder models or add new ones

### Why Dry-Run Default?
- **Safety**: Prevents accidental API spending during testing
- **Testing**: No API key needed for CI/CD
- **Development**: Fast iteration without network calls

### Why Exponential Backoff?
- **API-friendly**: Respects rate limits, reduces server load
- **Jitter**: Prevents thundering herd from parallel requests
- **Configurable**: Can adjust base/max backoff times

## Performance Characteristics

- **Rate limit**: 15 RPM (4s between requests)
- **Timeout**: 60s per request
- **Max retries**: 6 per model (with backoff: ~63s max per model)
- **Backoff range**: 1s base → 30s max (with jitter)
- **Total time (worst case)**: ~3 minutes for 3-model chain with full retries

## Error Scenarios

| Error Type | HTTP Status | Action |
|------------|-------------|--------|
| Auth failure | 401, 403 | Fail fast (no retry) |
| Bad request | 400 | Fail fast (no retry) |
| Rate limit | 429 | Retry with backoff |
| Server error | 5xx | Retry with backoff |
| Timeout | - | Retry with backoff |
| Network error | - | Retry with backoff |

## Validation

- ✅ All 156 tests pass
- ✅ `make lint` passes
- ✅ Dry-run smoke test works
- ✅ No runtime dependencies on API keys in dry-run mode
- ✅ Comprehensive test coverage for all error paths
