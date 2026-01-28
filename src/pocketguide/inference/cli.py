"""CLI for PocketGuide inference.

This module provides a command-line interface for running inference with PocketGuide.
Currently implements a deterministic stub that returns structured responses.
"""

import argparse


def generate_stub_response(prompt: str) -> dict[str, str]:
    """Generate a deterministic stub response for a given prompt.

    Args:
        prompt: The input prompt/question

    Returns:
        Dictionary with keys: summary, assumptions, next_steps
    """
    # Simple deterministic hash-based response to ensure consistency
    prompt_hash = hash(prompt) % 1000

    return {
        "summary": (
            f"This is a stub response for travel-related query (hash: {prompt_hash}). "
            "The actual model will provide detailed travel guidance based on your question. "
            "In future milestones, this will be replaced with fine-tuned LLM inference."
        ),
        "assumptions": (
            "No external data sources available (offline mode). "
            "Response based on prompt content only. "
            "Model inference not yet implemented."
        ),
        "next_steps": (
            "1. Review the response structure\n"
            "2. Verify deterministic output\n"
            "3. Proceed with evaluation pipeline"
        ),
    }


def format_response(response: dict[str, str]) -> str:
    """Format response dictionary into required structured text format.

    Args:
        response: Dictionary with summary, assumptions, next_steps

    Returns:
        Formatted string with required sections
    """
    return (
        f"Summary:\n{response['summary']}\n\n"
        f"Assumptions:\n{response['assumptions']}\n\n"
        f"Next steps:\n{response['next_steps']}"
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PocketGuide CLI - Offline travel assistant inference"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The travel-related prompt/question to process",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    # Generate response
    response = generate_stub_response(args.prompt)

    # Output in requested format
    if args.format == "json":
        import json

        print(json.dumps(response, indent=2))
    else:
        print(format_response(response))


if __name__ == "__main__":
    main()
