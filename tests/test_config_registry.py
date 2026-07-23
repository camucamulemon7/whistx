from __future__ import annotations

import ast
import os
import re
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
