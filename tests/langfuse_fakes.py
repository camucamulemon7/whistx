"""Synthetic SDK boundary, exercising the real observer's privacy filter."""
from types import SimpleNamespace

from server.langfuse_observer import LangfuseObserver


def recording_observer(*, capture_content=False, fail_at=None, enabled=True):
    calls = []

    def start(**values):
        if fail_at == 'start':
            raise RuntimeError('synthetic telemetry unavailable')
        call = dict(start=values, updates=[], exits=[])
        calls.append(call)

        class Context:
            def __enter__(self):
                if fail_at == 'enter':
                    raise RuntimeError('synthetic telemetry unavailable')
                return SimpleNamespace(update=update)

            def __exit__(self, *args):
                call['exits'].append(args)
                if fail_at == 'exit':
                    raise RuntimeError('synthetic telemetry unavailable')

        def update(**values):
            if fail_at == 'update':
                raise RuntimeError('synthetic telemetry unavailable')
            call['updates'].append(values)

        return Context()

    observer = LangfuseObserver(public_key='', secret_key='', capture_content=capture_content)
    observer.enabled = enabled
    observer._client = SimpleNamespace(start_as_current_observation=start)
    return observer, calls
