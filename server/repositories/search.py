"""Bounded literal substring searches shared by history and user lists."""
MAX_SEARCH_CHARS = 200


def normalize_query(query: str | None) -> str:
    value = (query or '').strip()
    if len(value) > MAX_SEARCH_CHARS:
        raise ValueError('search_query_too_long')
    return value


def substring_pattern(query: str) -> str:
    return '%' + query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%'
