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

# Temporary composition boundaries. Route modules not listed here must depend on
# services rather than the legacy runtime module.
RUNTIME_IMPORT_ALLOWLIST = {
    "server/api/routes/auth.py",  # OIDC coordinator
    "server/api/routes/health.py",  # runtime resource readiness
    "server/api/routes/summary.py",  # model resources
    "server/api/ws/transcribe.py",  # live transcription coordinator
    "server/core/application.py",  # application lifecycle
}

REMOVED_RUNTIME_HANDLERS = {
    "admin_approve_pending_user",
    "admin_pending_users",
    "admin_update_user_role",
    "admin_users",
    "auth_bootstrap_admin",
    "auth_login",
    "auth_logout",
    "auth_me",
    "auth_register",
    "create_history",
    "download_history_jsonl",
    "download_history_txt",
    "download_history_zip",
    "get_history_detail",
    "get_history_list",
    "get_history_screenshot",
}


def _imports_runtime(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"server", None}:
            if any(alias.name == "runtime" for alias in node.names):
                return True
        if isinstance(node, ast.Import) and any(alias.name == "server.runtime" for alias in node.names):
            return True
    return False


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
        functions = [
            node.name
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.assertEqual(functions, [])

    def test_runtime_imports_are_limited_to_composition_boundaries(self) -> None:
        violations: list[str] = []
        for path in sorted((SERVER / "api").rglob("*.py")) + sorted((SERVER / "core").rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative in RUNTIME_IMPORT_ALLOWLIST:
                continue
            if _imports_runtime(ast.parse(path.read_text(encoding="utf-8"))):
                violations.append(relative)
        self.assertEqual(violations, [])

    def test_runtime_does_not_reintroduce_duplicate_route_handlers(self) -> None:
        runtime_path = SERVER / "runtime.py"
        tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
        definitions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertEqual(sorted(definitions & REMOVED_RUNTIME_HANDLERS), [])

    def test_runtime_remains_below_composition_boundary_budget(self) -> None:
        line_count = len((SERVER / "runtime.py").read_text(encoding="utf-8").splitlines())
        self.assertLessEqual(line_count, 1_500)

    def test_services_do_not_depend_on_api_routes(self) -> None:
        violations: list[str] = []
        for path in sorted((SERVER / "services").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and (
                    node.module.startswith("server.api") or ".api" in node.module
                ):
                    violations.append(path.relative_to(ROOT).as_posix())
                    break
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
