# Environment Setup

## Quick Start

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Add your OpenRouter API key:**
   - Get your key from: https://openrouter.ai/keys
   - Open `.env` and replace `your-api-key-here` with your actual key:
     ```
     OPENROUTER_API_KEY=sk-or-v1-abc123...
     ```

3. **Test it works:**
   ```bash
   # Dry-run mode (no API key needed)
   python -m pocketguide.teachers.smoke --dry-run

   # Real API calls (uses your key from .env)
   python -m pocketguide.teachers.smoke --real
   ```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes (for real API calls) | - | Your OpenRouter API key |
| `OPENROUTER_APP_NAME` | No | `pocket-guide` | App name shown in OpenRouter dashboard |

## Security Notes

- ✅ `.env` is in `.gitignore` - your secrets won't be committed
- ✅ Use `.env.example` to share configuration structure without secrets
- ✅ Dry-run mode is the default - no accidental API spending

## Troubleshooting

**"OPENROUTER_API_KEY environment variable is required"**
- Make sure you've created `.env` from `.env.example`
- Check that your API key is set in `.env`
- Verify you're using `--real` flag (or `dry_run=False` in code)

**API key not being loaded**
- The `.env` file must be in the project root directory
- Check that `python-dotenv` is installed: `pip list | grep python-dotenv`
- Verify the file is named exactly `.env` (not `.env.txt`)

## Alternative: Export Directly

Instead of using `.env`, you can export variables in your shell:

```bash
export OPENROUTER_API_KEY=sk-or-v1-abc123...
export OPENROUTER_APP_NAME=pocket-guide
```

Both methods work - `.env` is more convenient for development.
