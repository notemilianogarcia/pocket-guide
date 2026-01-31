"""Exception types for teacher model clients."""


class TeacherError(Exception):
    """Base exception for teacher model errors."""

    pass


class TeacherAuthError(TeacherError):
    """Authentication error (401, 403)."""

    pass


class TeacherRateLimitError(TeacherError):
    """Rate limit exceeded (429)."""

    pass


class TeacherTransientError(TeacherError):
    """Transient error that can be retried (5xx, timeouts)."""

    pass


class TeacherBadRequestError(TeacherError):
    """Bad request error (400, client-side issue)."""

    pass


class TeacherUnsupportedParameterError(TeacherTransientError):
    """Provider/model does not support a parameter (e.g. response_format / structured outputs)."""

    pass
