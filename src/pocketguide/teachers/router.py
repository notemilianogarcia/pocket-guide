"""Router for teacher model generation with fallback chain."""


from pocketguide.teachers.base import TeacherClient, TeacherRequest, TeacherResponse
from pocketguide.teachers.errors import (
    TeacherAuthError,
    TeacherBadRequestError,
    TeacherRateLimitError,
    TeacherTransientError,
    TeacherUnsupportedParameterError,
)


class TeacherRouterClient(TeacherClient):
    """Router client with model fallback chain.

    Tries models in order, falls back on transient errors,
    fails fast on auth/bad request errors.
    """

    def __init__(
        self,
        backend: TeacherClient,
        models: list[str],
        fallback_to_paid: bool = False,
        max_retries_per_model: int = 6,
    ):
        """Initialize router client.

        Args:
            backend: Backend client (e.g., OpenRouterTeacherClient)
            models: List of model IDs to try in order
            fallback_to_paid: If True, allow fallback to paid models; if False, stop at last free model
            max_retries_per_model: Maximum retries per model
        """
        self.backend = backend
        self.models = models
        self.fallback_to_paid = fallback_to_paid
        self.max_retries_per_model = max_retries_per_model

    def generate(self, request: TeacherRequest) -> TeacherResponse:
        """Generate a response from the teacher model with fallback chain.

        Args:
            request: Teacher request

        Returns:
            Teacher response

        Raises:
            TeacherAuthError: Authentication failed
            TeacherBadRequestError: Bad request (client error)
            TeacherTransientError: All models exhausted
        """
        attempted_models = []
        last_error = None

        for model in self.models:
            # Check if model is paid and fallback_to_paid is False
            is_paid = self._is_paid_model(model)
            if is_paid and not self.fallback_to_paid:
                # Stop before paid models
                break

            attempted_models.append(model)

            # Update backend model
            self.backend.model = model

            try:
                # Try to generate with this model
                response = self.backend.generate(request)

                # Success! Add metadata
                if response.raw is None:
                    response.raw = {}

                response.raw["attempted_models"] = attempted_models
                response.raw["selected_model"] = model
                response.raw["chosen_model"] = model
                response.raw["fallback_occurred"] = len(attempted_models) > 1

                return response

            except TeacherUnsupportedParameterError as e:
                # Retry same model once without structured outputs if fallback_request set
                if request.fallback_request is not None:
                    try:
                        response = self.backend.generate(request.fallback_request)
                        if response.raw is None:
                            response.raw = {}
                        response.raw["attempted_models"] = attempted_models
                        response.raw["selected_model"] = model
                        response.raw["chosen_model"] = model
                        response.raw["fallback_occurred"] = len(attempted_models) > 1
                        response.raw["used_structured_outputs"] = False
                        response.raw["structured_outputs_schema_name"] = None
                        return response
                    except Exception:
                        last_error = e
                        if model == self.models[-1]:
                            raise TeacherTransientError(
                                f"All models exhausted. Attempted: {attempted_models}. Last error: {e}"
                            ) from e
                        continue
                last_error = e
                if model == self.models[-1]:
                    raise TeacherTransientError(
                        f"All models exhausted. Attempted: {attempted_models}. Last error: {e}"
                    ) from e
                continue

            except (TeacherAuthError, TeacherBadRequestError):
                # Fail fast - no retries or fallback
                raise

            except (TeacherRateLimitError, TeacherTransientError) as e:
                # Save error and try next model
                last_error = e

                # If this is the last model, re-raise
                if model == self.models[-1]:
                    raise TeacherTransientError(
                        f"All models exhausted. Attempted: {attempted_models}. Last error: {e}"
                    ) from e

                # Otherwise continue to next model
                continue

        # If we get here, we stopped before paid models
        raise TeacherTransientError(
            f"All free models exhausted. Attempted: {attempted_models}. "
            f"Set fallback_to_paid=True to try paid models. Last error: {last_error}"
        )

    def _is_paid_model(self, model: str) -> bool:
        """Check if model is paid.

        Args:
            model: Model ID

        Returns:
            True if model is paid, False if free
        """
        # Free models have :free suffix
        if model.endswith(":free"):
            return False

        # Paid models typically don't have :free suffix
        # Common paid models
        paid_prefixes = ["openai/", "anthropic/", "google/"]
        for prefix in paid_prefixes:
            if model.startswith(prefix):
                return True

        # Default to paid to be safe
        return True
