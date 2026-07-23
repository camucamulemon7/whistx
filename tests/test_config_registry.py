from __future__ import annotations

import ast
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("APP_SESSION_SECRET", "test-session-secret-abcdefghijklmnopqrstuvwxyz12")

from server.core.config.registry import ENV_BY_NAME, ENV_REGISTRY
from server.core.config.app import load_app_config
from server.core.config.asr import load_asr_config
from server.core.config.observability import load_observability_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_CALLS = {
    "getenv",
    "env_first_non_empty",
    "to_int",
    "to_float",
    "to_bool",
    "to_int_alias",
    "to_float_alias",
    "to_bool_alias",
}


def config_environment_names() -> set[str]:
    names: set[str] = set()
    for path in (ROOT / "server" / "core" / "config").glob("*.py"):
        if path.name == "registry.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if function not in CONFIG_CALLS:
                continue
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if re.fullmatch(r"[A-Z][A-Z0-9_]+", argument.value):
                        names.add(argument.value)
    return names


def example_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name] = value
    return values


class ConfigRegistryTests(unittest.TestCase):
    def test_registry_covers_every_config_loader_variable_and_alias(self) -> None:
        registered = set(ENV_BY_NAME)
        for item in ENV_REGISTRY:
            registered.update(item.aliases)
        self.assertEqual(config_environment_names(), registered)

    def test_env_example_contains_every_canonical_variable(self) -> None:
        values = example_values()
        self.assertEqual(set(ENV_BY_NAME) - set(values), set())

    def test_env_example_defaults_match_registry(self) -> None:
        values = example_values()
        mismatches = {
            item.name: (item.default, values[item.name])
            for item in ENV_REGISTRY
            if not item.secret and values[item.name] != item.default
        }
        self.assertEqual(mismatches, {})

    def test_container_forwards_every_canonical_variable(self) -> None:
        source = (ROOT / "scripts" / "container_common.sh").read_text(encoding="utf-8")
        forwarded = set(re.findall(r'-e "([A-Z][A-Z0-9_]+)=', source))
        self.assertEqual(set(ENV_BY_NAME) - forwarded, set())

    def test_container_preserves_host_runtime_values(self) -> None:
        environment = {
            **os.environ,
            "APP_ENV": "production",
            "APP_HOST": "127.0.0.1",
            "APP_PORT": "9001",
            "APP_ENTRYPOINT": "server.app:app",
            "APP_WS_PATH": "/custom/ws",
            "ASR_API_TIMEOUT_SECONDS": "17.5",
            "ASR_RETRY_MAX_ATTEMPTS": "9",
            "ASR_VAD_SPEECH_RATIO_MIN": "0.15",
            "ASR_CONTEXT_RECENT_LINES": "12",
            "HISTORY_RETENTION_DAYS": "31",
        }
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source scripts/container_common.sh; build_common_container_env; printf "%s\\n" "${COMMON_CONTAINER_ENV[@]}"',
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        arguments = result.stdout.splitlines()
        self.assertIn("APP_ENV=production", arguments)
        self.assertIn("ASR_API_TIMEOUT_SECONDS=17.5", arguments)
        self.assertIn("ASR_RETRY_MAX_ATTEMPTS=9", arguments)
        self.assertIn("ASR_VAD_SPEECH_RATIO_MIN=0.15", arguments)
        self.assertIn("ASR_CONTEXT_RECENT_LINES=12", arguments)
        self.assertIn("HISTORY_RETENTION_DAYS=31", arguments)

    def test_docker_and_podman_use_the_shared_environment_builder(self) -> None:
        for script_name in ("start.sh", "podman-run.sh"):
            source = (ROOT / script_name).read_text(encoding="utf-8")
            self.assertIn('source "${SCRIPT_DIR}/scripts/container_common.sh"', source)
            self.assertIn("build_common_container_env", source)
            self.assertIn('"${COMMON_CONTAINER_ENV[@]}"', source)

    def test_aliases_have_an_explicit_removal_policy(self) -> None:
        for item in ENV_REGISTRY:
            self.assertEqual(item.deprecated_aliases, item.aliases)

    def test_high_risk_defaults_match_runtime_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            with patch.dict(
                os.environ,
                {
                    "APP_DATA_DIR": data_dir,
                    "APP_SESSION_SECRET": "test-session-secret-abcdefghijklmnopqrstuvwxyz12",
                },
                clear=True,
            ):
                app = load_app_config()
                asr = load_asr_config()
                observability = load_observability_config()

        self.assertEqual(app.enable_self_signup, ENV_BY_NAME["ENABLE_SELF_SIGNUP"].default == "1")
        self.assertEqual(asr.context_max_chars, int(ENV_BY_NAME["ASR_CONTEXT_MAX_CHARS"].default))
        self.assertEqual(asr.context_recent_lines, int(ENV_BY_NAME["ASR_CONTEXT_RECENT_LINES"].default))
        self.assertEqual(asr.context_term_limit, int(ENV_BY_NAME["ASR_CONTEXT_TERM_LIMIT"].default))
        self.assertEqual(observability.langfuse_enabled, ENV_BY_NAME["LANGFUSE_ENABLED"].default == "1")


if __name__ == "__main__":
    unittest.main()
