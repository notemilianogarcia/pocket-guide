"""Tests for teacher router and OpenRouter client."""

from unittest.mock import Mock, patch

import pytest
from pocketguide.teachers.base import TeacherRequest, TeacherResponse
from pocketguide.teachers.errors import (
    TeacherAuthError,
    TeacherBadRequestError,
    TeacherRateLimitError,
    TeacherTransientError,
)
from pocketguide.teachers.openrouter import OpenRouterTeacherClient
from pocketguide.teachers.router import TeacherRouterClient


class TestOpenRouterClient:
    """Test OpenRouter client."""

    def test_dry_run_returns_dry_run(self):
        """Test dry-run mode returns DRY_RUN without API call."""
        client = OpenRouterTeacherClient(
            model="meta-llama/llama-3.3-70b-instruct:free", dry_run=True
        )

        request = TeacherRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=100,
        )

        response = client.generate(request)

        assert response.text == "DRY_RUN"
        assert response.model == "meta-llama/llama-3.3-70b-instruct:free"
        assert response.provider == "openrouter"
        assert response.raw["dry_run"] is True

    @patch("httpx.Client.post")
    def test_successful_generation(self, mock_post):
        """Test successful generation with mocked API call."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_post.return_value = mock_response

        client = OpenRouterTeacherClient(
            model="meta-llama/llama-3.3-70b-instruct:free",
            dry_run=False,
            api_key="test-key",
        )

        request = TeacherRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=100,
        )

        response = client.generate(request)

        assert response.text == "Test response"
        assert response.model == "meta-llama/llama-3.3-70b-instruct:free"
        assert response.provider == "openrouter"
        assert response.usage["total_tokens"] == 30

    @patch("httpx.Client.post")
    def test_auth_error_fails_fast(self, mock_post):
        """Test 401 raises TeacherAuthError immediately."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        client = OpenRouterTeacherClient(
            model="meta-llama/llama-3.3-70b-instruct:free",
            dry_run=False,
            api_key="bad-key",
        )

        request = TeacherRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=100,
        )

        with pytest.raises(TeacherAuthError):
            client.generate(request)

        # Should only call once (no retries)
        assert mock_post.call_count == 1

    @patch("httpx.Client.post")
    def test_bad_request_fails_fast(self, mock_post):
        """Test 400 raises TeacherBadRequestError immediately."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response

        client = OpenRouterTeacherClient(
            model="meta-llama/llama-3.3-70b-instruct:free",
            dry_run=False,
            api_key="test-key",
        )

        request = TeacherRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=100,
        )

        with pytest.raises(TeacherBadRequestError):
            client.generate(request)

        # Should only call once (no retries)
        assert mock_post.call_count == 1

    @patch("httpx.Client.post")
    @patch("time.sleep")
    def test_retry_with_backoff(self, mock_sleep, mock_post):
        """Test retry with exponential backoff on 429."""
        # First two calls: 429, third call: success
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.text = "Rate limit"

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "id": "test-id",
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "choices": [{"message": {"content": "Success after retry"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        mock_post.side_effect = [mock_response_429, mock_response_429, mock_response_success]

        client = OpenRouterTeacherClient(
            model="meta-llama/llama-3.3-70b-instruct:free",
            dry_run=False,
            api_key="test-key",
            max_retries=6,
        )

        request = TeacherRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=100,
        )

        response = client.generate(request)

        assert response.text == "Success after retry"
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2  # Two sleeps between three attempts

    @patch("httpx.Client.post")
    @patch("time.sleep")
    def test_rate_limit_exhausts_retries(self, mock_sleep, mock_post):
        """Test rate limit error after exhausting retries."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit"
        mock_post.return_value = mock_response

        client = OpenRouterTeacherClient(
            model="meta-llama/llama-3.3-70b-instruct:free",
            dry_run=False,
            api_key="test-key",
            max_retries=2,
        )

        request = TeacherRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=100,
        )

        with pytest.raises(TeacherRateLimitError):
            client.generate(request)

        # Should call max_retries + 1 times
        assert mock_post.call_count == 3

    @patch("httpx.Client.post")
    @patch("time.sleep")
    def test_server_error_retries(self, mock_sleep, mock_post):
        """Test retry on 500 server error."""
        # First call: 500, second call: success
        mock_response_500 = Mock()
        mock_response_500.status_code = 500
        mock_response_500.text = "Internal server error"

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "id": "test-id",
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "choices": [{"message": {"content": "Success after retry"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        mock_post.side_effect = [mock_response_500, mock_response_success]

        client = OpenRouterTeacherClient(
            model="meta-llama/llama-3.3-70b-instruct:free",
            dry_run=False,
            api_key="test-key",
        )

        request = TeacherRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=100,
        )

        response = client.generate(request)

        assert response.text == "Success after retry"
        assert mock_post.call_count == 2


class TestTeacherRouter:
    """Test teacher router with fallback chain."""

    def test_first_model_succeeds(self):
        """Test router returns immediately if first model succeeds."""
        backend = Mock()
        backend.generate.return_value = TeacherResponse(
            text="Success from first model",
            model="meta-llama/llama-3.3-70b-instruct:free",
            provider="openrouter",
            request_id="test-id",
            usage=None,
            timing={"latency_s": 0.5},
            raw={},
        )

        router = TeacherRouterClient(
            backend=backend,
            models=[
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-chat-v3.1:free",
                "openai/gpt-4o",
            ],
            fallback_to_paid=True,
        )

        request = TeacherRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=100,
        )

        response = router.generate(request)

        assert response.text == "Success from first model"
        assert response.raw["selected_model"] == "meta-llama/llama-3.3-70b-instruct:free"
        assert response.raw["fallback_occurred"] is False
        assert backend.generate.call_count == 1

    def test_fallback_chain(self):
        """Test router falls back to second model on transient error."""

        def side_effect(request):
            # First call: transient error, second call: success
            if backend.model == "meta-llama/llama-3.3-70b-instruct:free":
                raise TeacherTransientError("First model failed")
            else:
                return TeacherResponse(
                    text="Success from second model",
                    model=backend.model,
                    provider="openrouter",
                    request_id="test-id",
                    usage=None,
                    timing={"latency_s": 0.5},
                    raw={},
                )

        backend = Mock()
        backend.generate.side_effect = side_effect

        router = TeacherRouterClient(
            backend=backend,
            models=[
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-chat-v3.1:free",
            ],
            fallback_to_paid=True,
        )

        request = TeacherRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=100,
        )

        response = router.generate(request)

        assert response.text == "Success from second model"
        assert response.raw["selected_model"] == "deepseek/deepseek-chat-v3.1:free"
        assert response.raw["fallback_occurred"] is True
        assert backend.generate.call_count == 2

    def test_auth_error_fails_fast_no_fallback(self):
        """Test router fails fast on auth error without trying other models."""
        backend = Mock()
        backend.generate.side_effect = TeacherAuthError("Auth failed")

        router = TeacherRouterClient(
            backend=backend,
            models=[
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-chat-v3.1:free",
            ],
            fallback_to_paid=True,
        )

        request = TeacherRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=100,
        )

        with pytest.raises(TeacherAuthError):
            router.generate(request)

        # Should only try once (no fallback)
        assert backend.generate.call_count == 1

    def test_bad_request_fails_fast_no_fallback(self):
        """Test router fails fast on bad request without trying other models."""
        backend = Mock()
        backend.generate.side_effect = TeacherBadRequestError("Bad request")

        router = TeacherRouterClient(
            backend=backend,
            models=[
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-chat-v3.1:free",
            ],
            fallback_to_paid=True,
        )

        request = TeacherRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=100,
        )

        with pytest.raises(TeacherBadRequestError):
            router.generate(request)

        # Should only try once (no fallback)
        assert backend.generate.call_count == 1

    def test_fallback_to_paid_false_stops_at_free(self):
        """Test router stops before paid models when fallback_to_paid=False."""
        backend = Mock()
        backend.generate.side_effect = TeacherTransientError("Failed")

        router = TeacherRouterClient(
            backend=backend,
            models=[
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-chat-v3.1:free",
                "openai/gpt-4o",  # Paid model
            ],
            fallback_to_paid=False,  # Don't use paid models
        )

        request = TeacherRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=100,
        )

        with pytest.raises(TeacherTransientError) as exc_info:
            router.generate(request)

        # Should only try free models (2 calls)
        assert backend.generate.call_count == 2

        # Error message should mention fallback_to_paid
        assert "fallback_to_paid=True" in str(exc_info.value)

    def test_all_models_exhausted(self):
        """Test router raises error when all models fail."""
        backend = Mock()
        backend.generate.side_effect = TeacherTransientError("Failed")

        router = TeacherRouterClient(
            backend=backend,
            models=[
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-chat-v3.1:free",
            ],
            fallback_to_paid=True,
        )

        request = TeacherRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            max_tokens=100,
        )

        with pytest.raises(TeacherTransientError) as exc_info:
            router.generate(request)

        # Should try all models
        assert backend.generate.call_count == 2

        # Error message should list attempted models
        assert "meta-llama/llama-3.3-70b-instruct:free" in str(exc_info.value)
        assert "deepseek/deepseek-chat-v3.1:free" in str(exc_info.value)

    def test_is_paid_model_detection(self):
        """Test paid model detection logic."""
        router = TeacherRouterClient(
            backend=Mock(),
            models=[],
            fallback_to_paid=True,
        )

        # Free models
        assert router._is_paid_model("meta-llama/llama-3.3-70b-instruct:free") is False
        assert router._is_paid_model("deepseek/deepseek-chat-v3.1:free") is False

        # Paid models
        assert router._is_paid_model("openai/gpt-4o") is True
        assert router._is_paid_model("anthropic/claude-3-opus") is True
        assert router._is_paid_model("google/gemini-pro") is True

        # Unknown models default to paid (safe)
        assert router._is_paid_model("unknown/model") is True
