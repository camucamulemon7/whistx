PYTHON ?= python3.12
VENV ?= .venv
PY = $(VENV)/bin/python
UV_IMAGE = ghcr.io/astral-sh/uv:0.8.17-python3.12-bookworm-slim

.PHONY: help setup lint test test-web test-browser check migrate migration-status locks hygiene
help:
	@echo 'setup lint test test-web test-browser check migrate migration-status locks hygiene'
setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --require-hashes -r requirements-dev.lock
lint:
	$(VENV)/bin/ruff check .
	$(PY) -m compileall -q server tests
	git ls-files -z '*.sh' | xargs -0 -n1 bash -n
	git ls-files -z 'web/*.js' 'web/**/*.js' | xargs -0 -n1 node --check
hygiene:
	$(PYTHON) scripts/check_repository.py
test:
	$(PY) -m unittest discover -s tests
	$(PY) scripts/eval_asr.py --dataset tests/fixtures/asr_eval --baseline tests/fixtures/asr_eval/baselines/whisper.json --max-regression 0.01 --strict
test-web:
	npm run test:web
test-browser:
	npm run test:browser
check: lint hygiene test test-web test-browser
migrate:
	$(VENV)/bin/alembic upgrade head
migration-status:
	$(VENV)/bin/alembic current
locks:
	docker run --rm --user "$$(id -u):$$(id -g)" -e UV_CACHE_DIR=/tmp/uv-cache -v "$(CURDIR):/workspace" -w /workspace $(UV_IMAGE) uv pip compile requirements.txt --python-version 3.12 --generate-hashes -o requirements.lock
	docker run --rm --user "$$(id -u):$$(id -g)" -e UV_CACHE_DIR=/tmp/uv-cache -v "$(CURDIR):/workspace" -w /workspace $(UV_IMAGE) uv pip compile requirements-diarization.txt --python-version 3.12 --generate-hashes -o requirements-diarization.lock
	docker run --rm --user "$$(id -u):$$(id -g)" -e UV_CACHE_DIR=/tmp/uv-cache -v "$(CURDIR):/workspace" -w /workspace $(UV_IMAGE) uv pip compile requirements-dev.txt --python-version 3.12 --generate-hashes -o requirements-dev.lock
