# Contributing

Install Python 3.12 with venv/pip support, Node.js 22+, GNU Make, FFmpeg, Git, and Chrome or Chromium. Set `CHROME_BIN` if the browser is not installed at a standard path. Docker is needed for reproducible dependency-lock regeneration and container checks.

Run `make setup`, copy `.env.example` to `.env`, and set your own session secret and provider settings. Never commit credentials or recordings. The application launch scripts load `.env`; Make commands use the calling shell's environment. Export `APP_SESSION_SECRET` and `APP_DB_URL` before migration or application commands. Use an isolated test database, never production.

Run `make migrate` before starting the application with `./run.sh`. `make migration-status` shows the current revision. Use `make check` for Python lint/compilation, repository hygiene, Python tests, synthetic ASR regression, JavaScript unit tests, and browser recording/UI tests. Individual targets are listed by `make help`. Override an existing environment with `make VENV=.venv-meeting test`. Tests requiring live ASR providers are separate from synthetic regression; see README's ASR evaluation instructions.

Change dependency declarations in `requirements*.txt`, then run `make locks` and `make setup`. Commit matching declarations and generated hash locks together. The lock target pins the same uv version as the container.

Keep behavior changes separate from broad refactors. Describe the trigger, resulting behavior, and validation in each PR. Add behavior tests for bugs and preserve the architecture boundaries in `docs/architecture.md`. New documents belong in `docs/`; generated test output belongs in `artifacts/`. Only shell scripts/entrypoints carry executable bits; Python utilities run through Python.

Use [SECURITY.md](SECURITY.md) for vulnerability reporting and [operations](docs/operations.md) for release and restore procedures. The repository maintainer reviews releases and security reports; deployment operators own production credentials, backups, and incident response.
