"""Administrator overrides, persisted in the mounted data directory."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

FIELDS = {
    'HISTORY_RETENTION_DAYS', 'ENABLE_SELF_SIGNUP', 'ALLOW_GUEST_TRANSCRIPTION',
    'ASR_BACKEND', 'ASR_BASE_URL', 'ASR_MODEL', 'ASR_API_KEY',
    'SUMMARY_BASE_URL', 'SUMMARY_MODEL', 'SUMMARY_API_KEY',
}
SECRETS = {'ASR_API_KEY', 'SUMMARY_API_KEY'}


def overrides_path() -> Path:
    return Path(os.getenv('APP_DATA_DIR') or os.getenv('DATA_DIR') or 'data') / 'admin-settings.json'


def read_overrides() -> dict:
    path = overrides_path()
    return json.loads(path.read_text()) if path.exists() else {}


def validate(values: dict) -> dict:
    if not isinstance(values, dict) or set(values) - FIELDS:
        raise ValueError('invalid_settings')
    result = {}
    for key, value in values.items():
        if not isinstance(value, str) or len(value) > 4096 or '\n' in value or '\r' in value:
            raise ValueError('invalid_settings')
        value = value.strip()
        if key == 'HISTORY_RETENTION_DAYS' and (not value.isdigit() or not 0 <= int(value) <= 36500):
            raise ValueError('invalid_retention')
        if key in {'ENABLE_SELF_SIGNUP', 'ALLOW_GUEST_TRANSCRIPTION'} and value not in {'0', '1'}:
            raise ValueError('invalid_boolean')
        if key == 'ASR_BACKEND' and value not in {'whisper', 'qwen3_vllm'}:
            raise ValueError('invalid_backend')
        if key.endswith('_BASE_URL'):
            url = urlsplit(value)
            if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError('invalid_url')
        if key.endswith('_MODEL') and not value:
            raise ValueError('invalid_model')
        if key in SECRETS and not value:
            continue  # Blank password inputs preserve the existing credential.
        result[key] = value
    return result


def load_overrides() -> None:
    os.environ.update(validate(read_overrides()))


def save_overrides(values: dict) -> None:
    values = validate(values)
    path = overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {**read_overrides(), **values}
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.admin-settings-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(current, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
