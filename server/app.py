"""ASGI entrypoint.

Application construction, route registration, and lifespan wiring live in one
place: :mod:`server.core.application`.
"""

from .core.application import create_app

app = create_app()
