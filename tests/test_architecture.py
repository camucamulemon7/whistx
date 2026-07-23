from __future__ import annotations

import ast
import os
import unittest
from collections import Counter
from pathlib import Path

os.environ.setdefault("APP_SESSION_SECRET", "architecture-test-session-secret-123456789")
os.environ.setdefault("ASR_API_KEY", "test-key")

from server.app import app

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"


class ArchitectureTests(unittest.TestCase):
    def test_app_routes_are_unique(self) -> None:
        route_keys: list[tuple[str, str]] = []
        for route in app.routes:
            for method in getattr(route, "methods", None) or {"WEBSOCKET"}:
                route_keys.append((method, route.path))
        duplicates = [key for key, count in Counter(route_keys).items() if count > 1]
        self.assertEqual(duplicates, [])

    def test_only_application_module_constructs_fastapi(self) -> None:
        constructors: list[str] = []
        for path in SERVER.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "FastAPI":
                    constructors.append(str(path.relative_to(ROOT)))
        self.assertEqual(constructors, ["server/core/application.py"])

    def test_runtime_does_not_register_routes_or_static_mount(self) -> None:
        source = (SERVER / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("@app.", source)
        self.assertNotIn("FastAPI(", source)
        self.assertNotIn("StaticFiles(", source)
        self.assertNotIn("app.mount(", source)

    def test_removed_compatibility_modules_do_not_return(self) -> None:
        self.assertFalse((SERVER / "legacy_app.py").exists())
        self.assertFalse((SERVER / "config.py").exists())
        for path in SERVER.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("legacy_app", source, str(path))

    def test_asgi_entrypoint_only_publishes_app(self) -> None:
        module = ast.parse((SERVER / "app.py").read_text(encoding="utf-8"))
        functions = [node.name for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.assertEqual(functions, [])


if __name__ == "__main__":
    unittest.main()
