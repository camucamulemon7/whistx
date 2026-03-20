FROM python:3.12-slim

ARG INSTALL_DIARIZATION=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

COPY pyproject.toml /app/pyproject.toml
COPY requirements.txt /app/requirements.txt
COPY requirements-diarization.txt /app/requirements-diarization.txt
RUN uv pip install --system --no-cache -r /app/requirements.txt \
    && if [ "${INSTALL_DIARIZATION}" = "1" ]; then uv pip install --system --no-cache -r /app/requirements-diarization.txt; fi

COPY alembic.ini /app/alembic.ini
COPY alembic /app/alembic
COPY server /app/server
COPY web /app/web
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
RUN mkdir -p /app/data/transcripts

VOLUME ["/app/data"]
EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8005/api/health', timeout=3).read()" || exit 1

CMD ["/app/entrypoint.sh"]
