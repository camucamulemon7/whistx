"""A deployment-controlled OpenWebUI origin, separate from the ASR endpoint."""
from urllib.parse import urlsplit

DEFAULT_GENERATION_MODEL = 'qwopus3.8-27b-flash-v2'


def normalize_openwebui_url(value: str) -> str:
    value = value.strip().rstrip('/')
    if not value:
        return ''
    url = urlsplit(value)
    if (url.scheme not in {'http', 'https'} or not url.hostname or url.username
            or url.password or url.query or url.fragment):
        raise ValueError('invalid_openwebui_url')
    if value.endswith('/api'):
        value = value[:-4]
    if value.endswith('/api/v1'):
        value = value[:-7]
    return value
