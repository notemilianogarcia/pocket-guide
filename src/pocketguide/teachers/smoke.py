#!/usr/bin/env python3
"""Smoke test CLI for teacher router + OpenRouter backend.

Usage:
    python -m pocketguide.teachers.smoke --dry-run
    python -m pocketguide.teachers.smoke --real --fallback-paid

Environment variables:
    OPENROUTER_API_KEY: OpenRouter API key (required for --real mode)
    OPENROUTER_APP_NAME: Optional app name for headers
"""

import argparse
import json

from pocketguide.teachers.base import TeacherRequest
from pocketguide.teachers.openrouter import OpenRouterTeacherClient
from pocketguide.teachers.router import TeacherRouterClient


def main():
    parser = argparse.ArgumentParser(description="Smoke test for teacher router")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (no API calls)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Run with real API calls (requires OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--fallback-paid",
        action="store_true",
        default=False,
        help="Allow fallback to paid models",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What are the top 3 tourist attractions in Paris?",
        help="User prompt to send",
    )

    args = parser.parse_args()

    # Determine dry_run mode
    dry_run = not args.real

    print("🔧 Teacher Router Smoke Test")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if dry_run else 'REAL API CALLS'}")
    print(f"Fallback to paid: {args.fallback_paid}")
    print(f"Prompt: {args.prompt[:50]}...")
    print("=" * 60)

    # Create OpenRouter backend
    backend = OpenRouterTeacherClient(
        model="meta-llama/llama-3.3-70b-instruct:free",  # Will be overridden by router
        dry_run=dry_run,
        rpm=15,
        max_retries=2,
    )

    # Create router with fallback chain
    router = TeacherRouterClient(
        backend=backend,
        models=[
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-chat-v3.1:free",
            "openai/gpt-4o",
        ],
        fallback_to_paid=args.fallback_paid,
    )

    # Create request
    request = TeacherRequest(
        messages=[{"role": "user", "content": args.prompt}],
        temperature=0.2,
        max_tokens=300,
        seed=42,
    )

    # Generate
    print("\n📤 Sending request...")
    try:
        response = router.generate(request)

        print("\n✅ Response received:")
        print(f"  Text: {response.text[:100]}{'...' if len(response.text) > 100 else ''}")
        print(f"  Model: {response.model}")
        print(f"  Provider: {response.provider}")
        print(f"  Request ID: {response.request_id}")
        print(f"  Latency: {response.timing.get('latency_s', 0):.2f}s")

        if response.usage:
            print(f"  Tokens: {response.usage}")

        if response.raw:
            print("\n📋 Metadata:")
            print(f"  Selected model: {response.raw.get('selected_model')}")
            print(f"  Attempted models: {response.raw.get('attempted_models')}")
            print(f"  Fallback occurred: {response.raw.get('fallback_occurred')}")

        if dry_run:
            print("\n🔍 Request preview:")
            print(json.dumps(response.raw.get("request_preview"), indent=2))

    except Exception as e:
        print(f"\n❌ Error: {e.__class__.__name__}: {e}")
        return 1

    print("\n✅ Smoke test completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
