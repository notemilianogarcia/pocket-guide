"""OpenRouter client for teacher model generation."""

import os
import random
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from pocketguide.teachers.base import TeacherClient, TeacherRequest, TeacherResponse
from pocketguide.teachers.errors import (
    TeacherAuthError,
    TeacherBadRequestError,
    TeacherRateLimitError,
    TeacherTransientError,
    TeacherUnsupportedParameterError,
)
from pocketguide.utils.rate_limiter import RateLimiter

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(project_root / ".env")


class OpenRouterTeacherClient(TeacherClient):
    """OpenRouter client for teacher model generation.

    Supports OpenAI-compatible chat completions API with:
    - Rate limiting
    - Retry with exponential backoff + jitter
    - Dry-run mode
    - Typed error handling
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        app_name: str | None = None,
        dry_run: bool = True,
        timeout_s: float = 60.0,
        rpm: int = 15,
        max_retries: int = 6,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
    ):
        """Initialize OpenRouter client.

        Args:
            model: Model identifier
            base_url: Base URL for API (default: OpenRouter)
            api_key: API key (uses OPENROUTER_API_KEY env var if None)
            app_name: App name for optional headers (uses OPENROUTER_APP_NAME env var if None)
            dry_run: If True, skip actual API calls
            timeout_s: Request timeout in seconds
            rpm: Requests per minute limit
            max_retries: Maximum retry attempts for transient errors
            backoff_base_s: Base backoff time in seconds
            backoff_max_s: Maximum backoff time in seconds
        """
        self.model = model
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.dry_run = dry_run
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s

        # Get API key
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")

        if not dry_run and not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required when dry_run=False. "
                "Set it with: export OPENROUTER_API_KEY=your-key-here"
            )

        self.api_key = api_key

        # Optional app name for headers
        self.app_name = app_name or os.getenv("OPENROUTER_APP_NAME")

        # Rate limiter
        self.rate_limiter = RateLimiter(rpm=rpm)

        # HTTP client
        self.client = httpx.Client(timeout=timeout_s)

    def generate(self, request: TeacherRequest) -> TeacherResponse:
        """Generate a response from the teacher model.

        Args:
            request: Teacher request

        Returns:
            Teacher response

        Raises:
            TeacherAuthError: Authentication failed
            TeacherRateLimitError: Rate limit exceeded after retries
            TeacherTransientError: Transient error after retries
            TeacherBadRequestError: Bad request (client error)
        """
        # Dry-run mode
        if self.dry_run:
            return self._dry_run_response(request)

        # Rate limiting
        self.rate_limiter.wait_if_needed()

        # Build request
        payload = {
            "model": self.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        if request.top_p is not None:
            payload["top_p"] = request.top_p

        if request.seed is not None:
            payload["seed"] = request.seed

        schema_name: str | None = None
        if request.response_format is not None:
            payload["response_format"] = request.response_format
            schema_name = request.response_format.get("json_schema", {}).get("name")
        if request.require_parameters is not None:
            payload["require_parameters"] = request.require_parameters

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        if self.app_name:
            headers["X-Title"] = self.app_name

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()

                response = self.client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )

                latency_s = time.time() - start_time

                # Handle errors
                if response.status_code == 401 or response.status_code == 403:
                    raise TeacherAuthError(
                        f"Authentication failed (status {response.status_code}): {response.text[:200]}"
                    )

                if response.status_code == 400:
                    text_lower = response.text.lower()
                    if any(
                        k in text_lower
                        for k in (
                            "unsupported",
                            "response_format",
                            "require_parameters",
                            "structured",
                            "json_schema",
                            "parameter",
                        )
                    ):
                        raise TeacherUnsupportedParameterError(
                            f"Structured outputs or parameter unsupported (400): {response.text[:300]}"
                        )
                    raise TeacherBadRequestError(
                        f"Bad request (status {response.status_code}): {response.text[:200]}"
                    )

                if response.status_code == 429:
                    # Rate limit - retry with backoff
                    if attempt < self.max_retries:
                        backoff = self._calculate_backoff(attempt)
                        time.sleep(backoff)
                        continue
                    else:
                        raise TeacherRateLimitError(
                            f"Rate limit exceeded after {self.max_retries} retries"
                        )

                if response.status_code == 404:
                    # Model not found – don't retry; router will try next model
                    raise TeacherTransientError(
                        f"Model not found (404): {self.model}. "
                        "Check https://openrouter.ai/api/v1/models for current IDs."
                    )

                if response.status_code == 402:
                    # Payment required – don't retry; router will try next model
                    raise TeacherTransientError(
                        f"Payment required (402) for model {self.model}. "
                        "Add credits or use a different model."
                    )

                if response.status_code >= 500:
                    # Server error - retry with backoff
                    if attempt < self.max_retries:
                        backoff = self._calculate_backoff(attempt)
                        time.sleep(backoff)
                        continue
                    else:
                        raise TeacherTransientError(
                            f"Server error (status {response.status_code}) after {self.max_retries} retries: {response.text[:200]}"
                        )

                if response.status_code != 200:
                    raise TeacherTransientError(
                        f"Unexpected status {response.status_code}: {response.text[:200]}"
                    )

                # Parse response
                data = response.json()

                # Extract text
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Extract usage
                usage = None
                if "usage" in data:
                    usage = {
                        "prompt_tokens": data["usage"].get("prompt_tokens"),
                        "completion_tokens": data["usage"].get("completion_tokens"),
                        "total_tokens": data["usage"].get("total_tokens"),
                    }

                # Build response with structured-outputs metadata
                raw_meta = {
                    "status_code": response.status_code,
                    "model": data.get("model"),
                    "used_structured_outputs": request.response_format is not None,
                    "structured_outputs_schema_name": schema_name,
                }
                return TeacherResponse(
                    text=text,
                    model=data.get("model", self.model),
                    provider="openrouter",
                    request_id=data.get("id"),
                    usage=usage,
                    timing={"latency_s": latency_s},
                    raw=raw_meta,
                )

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries:
                    backoff = self._calculate_backoff(attempt)
                    time.sleep(backoff)
                    continue
                else:
                    raise TeacherTransientError(
                        f"Timeout after {self.max_retries} retries: {e}"
                    ) from e

            except httpx.RequestError as e:
                last_error = e
                if attempt < self.max_retries:
                    backoff = self._calculate_backoff(attempt)
                    time.sleep(backoff)
                    continue
                else:
                    raise TeacherTransientError(
                        f"Request error after {self.max_retries} retries: {e}"
                    ) from e

        # Should not reach here, but just in case
        raise TeacherTransientError(f"Failed after {self.max_retries} retries: {last_error}")

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Backoff time in seconds
        """
        # Exponential backoff: base * 2^attempt
        backoff = self.backoff_base_s * (2**attempt)

        # Cap at max
        backoff = min(backoff, self.backoff_max_s)

        # Add jitter (±25%)
        jitter = backoff * 0.25 * (random.random() * 2 - 1)
        backoff += jitter

        return max(0.1, backoff)  # Minimum 0.1s

    def _dry_run_response(self, request: TeacherRequest) -> TeacherResponse:
        """Return a dry-run response without making API call.

        Args:
            request: Teacher request

        Returns:
            Dry-run teacher response
        """
        # Redact API key in preview
        schema_name = None
        if request.response_format is not None:
            schema_name = request.response_format.get("json_schema", {}).get("name")
        request_preview = {
            "model": self.model,
            "messages": [
                {"role": m.get("role"), "content": m.get("content", "")[:100] + "..."}
                for m in request.messages[:2]
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        return TeacherResponse(
            text="DRY_RUN",
            model=self.model,
            provider="openrouter",
            request_id="dry_run",
            usage=None,
            timing={"latency_s": 0.0},
            raw={
                "dry_run": True,
                "request_preview": request_preview,
                "used_structured_outputs": request.response_format is not None,
                "structured_outputs_schema_name": schema_name,
            },
        )

    def __del__(self):
        """Close HTTP client on cleanup."""
        if hasattr(self, "client"):
            self.client.close()
