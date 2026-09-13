from __future__ import annotations

import logging
import re
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

    def current_trace_context(self) -> dict[str, str] | None:
        return None

    def create_trace_context(self, **_: Any) -> dict[str, str] | None:
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
        enabled: bool = False,
        capture_content: bool = False,
    ) -> None:
        self._client: Any = None
        self.enabled = False
        self.capture_content = capture_content

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
    def _observe(self, method: str, **values: Any) -> Iterator[Any]:
        if not self.enabled or self._client is None:
            yield _NoopObservation()
            return
        arguments = {key: _safe_serialize(value, capture_content=self.capture_content)
                     for key, value in values.items()}
        arguments['name'] = values['name'] if re.fullmatch(r'[a-zA-Z0-9_.:-]{1,100}', values['name']) else 'observation'
        if 'trace_context' in values:
            arguments['trace_context'] = values['trace_context']
        if 'as_type' in values:
            arguments['as_type'] = values['as_type']
        try:
            context = getattr(self._client, method)(**arguments)
            observation = context.__enter__()
        except Exception:
            logger.debug('langfuse observation could not start', exc_info=True)
            yield _NoopObservation()
            return
        try:
            yield _PrivateObservation(observation, self.capture_content)
        except BaseException:
            try:
                # Provider exceptions may contain URLs or request text. Do not
                # pass them to an SDK context manager that exports exceptions.
                context.__exit__(None, None, None)
            except Exception:
                logger.debug('langfuse observation cleanup failed', exc_info=True)
            raise
        else:
            try:
                context.__exit__(None, None, None)
            except Exception:
                logger.debug('langfuse observation cleanup failed', exc_info=True)

    def span(self, *, name: str, input: Any = None, output: Any = None, metadata: Any = None):
        return self._observe('start_as_current_span', name=name, input=input, output=output, metadata=metadata)

    def generation(self, *, name: str, model: str | None = None, input: Any = None,
                   output: Any = None, metadata: Any = None, model_parameters: Any = None,
                   trace_context: dict[str, str] | None = None):
        return self._observe('start_as_current_observation', name=name, as_type='generation',
                             model=model, input=input, output=output, metadata=metadata,
                             model_parameters=model_parameters, trace_context=trace_context)

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

    def current_trace_context(self) -> dict[str, str] | None:
        if not self.enabled or self._client is None:
            return None
        try:
            trace_id = self._client.get_current_trace_id()
            observation_id = self._client.get_current_observation_id()
            if not trace_id:
                return None
            payload = {"trace_id": trace_id}
            if observation_id:
                payload["parent_span_id"] = observation_id
            return payload
        except Exception:  # noqa: BLE001
            logger.debug("langfuse current trace lookup failed", exc_info=True)
            return None

    def create_trace_context(
        self,
        *,
        name: str,
        input: Any = None,
        metadata: Any = None,
    ) -> dict[str, str] | None:
        if not self.enabled or self._client is None:
            return None

        trace_id: str | None = None
        observation_id: str | None = None
        try:
            with self._client.start_as_current_span(
                name=name,
                input=_safe_serialize(input, capture_content=self.capture_content),
                metadata=_safe_serialize(metadata, capture_content=self.capture_content),
            ):
                trace_id = self._client.get_current_trace_id()
                observation_id = self._client.get_current_observation_id()
        except Exception:  # noqa: BLE001
            logger.debug("langfuse trace root skipped: %s", name, exc_info=True)
            return None

        if not trace_id:
            return None
        payload = {"trace_id": trace_id}
        if observation_id:
            payload["parent_span_id"] = observation_id
        return payload


def make_langfuse_observer(
    *,
    public_key: str,
    secret_key: str,
    host: str | None,
    environment: str | None,
    release: str | None,
    enabled: bool,
    capture_content: bool = False,
) -> LangfuseObserver | _NoopObserver:
    observer = LangfuseObserver(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        environment=environment,
        release=release,
        enabled=enabled,
        capture_content=capture_content,
    )
    if observer.enabled:
        return observer
    return _NoopObserver()


class _PrivateObservation:
    def __init__(self, observation: Any, capture_content: bool):
        self._observation = observation
        self._capture_content = capture_content

    def update(self, **values: Any) -> None:
        self._call('update', values)

    def end(self, **values: Any) -> None:
        self._call('end', values)

    def _call(self, method: str, values: dict[str, Any]) -> None:
        try:
            getattr(self._observation, method)(**{
                key: _safe_serialize(value, capture_content=self._capture_content)
                for key, value in values.items()
            })
        except Exception:
            logger.debug('langfuse observation update failed', exc_info=True)


_SENSITIVE_KEY = re.compile(r'(password|secret|token|authorization|cookie|email|endpoint|base.?url|api.?key|host)', re.I)
_SAFE_ENUMS = {'ja', 'en', 'auto', 'mic', 'display', 'both', 'strict', 'normal', 'api', 'estimated', 'generation'}


def _safe_serialize(value: Any, *, capture_content: bool = False, _depth: int = 0) -> Any:
    if _depth > 8:
        return '[truncated]'
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        if not capture_content:
            return value if value in _SAFE_ENUMS else '[redacted]'
        text = re.sub(r'https?://[^\s<>]+', '[url]', value)
        text = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[email]', text)
        text = re.sub(r'(?i)\b(?:sk-|bearer\s+)[A-Za-z0-9_-]{8,}', '[credential]', text)
        return text[:8000] + ('...(truncated)' if len(text) > 8000 else '')
    if isinstance(value, bytes):
        return {'bytes': len(value)}
    if isinstance(value, dict):
        return {str(key)[:64]: '[redacted]' if _SENSITIVE_KEY.search(str(key)) else
                _safe_serialize(item, capture_content=capture_content, _depth=_depth + 1)
                for key, item in list(value.items())[:64]}
    if isinstance(value, (list, tuple, set)):
        return [_safe_serialize(item, capture_content=capture_content, _depth=_depth + 1) for item in list(value)[:64]]
    return '[redacted]'
