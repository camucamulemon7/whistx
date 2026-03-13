from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator


logger = logging.getLogger(__name__)


class _NoopObservation:
    def update(self, **_: Any) -> None:
        return None

    def end(self) -> None:
        return None


class _NoopObserver:
    enabled = False

    @contextmanager
    def span(self, **_: Any) -> Iterator[_NoopObservation]:
        yield _NoopObservation()

    @contextmanager
    def generation(self, **_: Any) -> Iterator[_NoopObservation]:
        yield _NoopObservation()

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class LangfuseObserver:
    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str | None = None,
        environment: str | None = None,
        release: str | None = None,
        enabled: bool = True,
    ) -> None:
        self._client: Any = None
        self.enabled = False

        if not enabled or not public_key or not secret_key:
            return

        try:
            from langfuse import Langfuse
        except Exception as exc:  # noqa: BLE001
            logger.warning("langfuse disabled: sdk import failed: %s", exc)
            return

        try:
            kwargs: dict[str, Any] = {
                "public_key": public_key,
                "secret_key": secret_key,
            }
            if host:
                kwargs["host"] = host
            if environment:
                kwargs["environment"] = environment
            if release:
                kwargs["release"] = release
            self._client = Langfuse(**kwargs)
            self.enabled = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("langfuse disabled: init failed: %s", exc)

    @contextmanager
    def span(
        self,
        *,
        name: str,
        input: Any = None,
        output: Any = None,
        metadata: Any = None,
    ) -> Iterator[Any]:
        if not self.enabled or self._client is None:
            yield _NoopObservation()
            return

        try:
            with self._client.start_as_current_span(
                name=name,
                input=_safe_serialize(input),
                output=_safe_serialize(output),
                metadata=_safe_serialize(metadata),
            ) as span:
                yield span
        except Exception as exc:  # noqa: BLE001
            logger.debug("langfuse span skipped: %s", exc, exc_info=True)
            yield _NoopObservation()

    @contextmanager
    def generation(
        self,
        *,
        name: str,
        model: str | None = None,
        input: Any = None,
        output: Any = None,
        metadata: Any = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        if not self.enabled or self._client is None:
            yield _NoopObservation()
            return

        try:
            with self._client.start_as_current_observation(
                name=name,
                as_type="generation",
                model=model,
                input=_safe_serialize(input),
                output=_safe_serialize(output),
                metadata=_safe_serialize(metadata),
                model_parameters=_safe_serialize(model_parameters),
            ) as generation:
                yield generation
        except Exception as exc:  # noqa: BLE001
            logger.debug("langfuse generation skipped: %s", exc, exc_info=True)
            yield _NoopObservation()

    def flush(self) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            self._client.flush()
        except Exception:  # noqa: BLE001
            logger.debug("langfuse flush failed", exc_info=True)

    def shutdown(self) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            self._client.shutdown()
        except Exception:  # noqa: BLE001
            logger.debug("langfuse shutdown failed", exc_info=True)


def make_langfuse_observer(
    *,
    public_key: str,
    secret_key: str,
    host: str | None,
    environment: str | None,
    release: str | None,
    enabled: bool,
) -> LangfuseObserver | _NoopObserver:
    observer = LangfuseObserver(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        environment=environment,
        release=release,
        enabled=enabled,
    )
    if observer.enabled:
        return observer
    return _NoopObserver()


def _safe_serialize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str):
            return _clip_text(value)
        return value
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    if isinstance(value, dict):
        return {str(key): _safe_serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_serialize(item) for item in list(value)[:64]]
    return _clip_text(str(value))


def _clip_text(text: str, limit: int = 8000) -> str:
    clean = str(text or "")
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "...(truncated)"
