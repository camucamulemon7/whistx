"""Public error envelopes never include provider exception text."""
from __future__ import annotations

from contextvars import ContextVar
import logging
from uuid import uuid4

correlation_id: ContextVar[str | None] = ContextVar('correlation_id', default=None)


def public_error(code: str, exc: BaseException, logger: logging.Logger) -> dict[str, str]:
    identifier = correlation_id.get() or uuid4().hex
    logger.error('%s correlation_id=%s exception_type=%s', code, identifier, type(exc).__name__,
                 exc_info=(type(exc), exc, exc.__traceback__))
    return {'error': code, 'correlationId': identifier}
