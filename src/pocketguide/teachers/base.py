"""Base interface for teacher model clients."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TeacherRequest:
    """Request to a teacher model.

    Attributes:
        messages: Chat messages in OpenAI format
        temperature: Sampling temperature (0.0 - 2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter (optional)
        seed: Random seed for determinism (optional, may be ignored by provider)
        metadata: Additional metadata (prompt_id, template_version, etc.)
    """

    messages: list[dict[str, str]]
    temperature: float = 0.2
    max_tokens: int = 900
    top_p: float | None = 0.9
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TeacherResponse:
    """Response from a teacher model.

    Attributes:
        text: Generated text content
        model: Model identifier used for generation
        provider: Provider name (e.g., "openrouter")
        request_id: Provider's request ID (if available)
        usage: Token usage statistics (prompt_tokens, completion_tokens, total_tokens)
        timing: Timing information (latency_s)
        raw: Raw response data (small subset, no huge payloads)
    """

    text: str
    model: str
    provider: str
    request_id: str | None = None
    usage: dict[str, int] | None = None
    timing: dict[str, float] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class TeacherClient:
    """Base interface for teacher model clients."""

    def generate(self, request: TeacherRequest) -> TeacherResponse:
        """Generate a response from the teacher model.

        Args:
            request: Teacher request

        Returns:
            Teacher response

        Raises:
            TeacherAuthError: Authentication failed
            TeacherRateLimitError: Rate limit exceeded
            TeacherTransientError: Transient error (can retry)
            TeacherBadRequestError: Bad request (client error)
        """
        raise NotImplementedError
