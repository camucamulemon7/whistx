from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

os.environ.setdefault('APP_SESSION_SECRET', 'test-session-secret-abcdefghijklmnopqrstuvwxyz12')


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.core import rate_limit








def frontend_source():
    paths = [ROOT / 'web' / 'src' / 'app.js', *sorted((ROOT / 'web' / 'src' / 'controllers').glob('*.js'))]
    return '\n'.join(path.read_text(encoding='utf-8') for path in paths)


def stylesheet_source():
    entry = (ROOT / 'web' / 'style.css').read_text(encoding='utf-8')
    imports = re.findall(r'@import url\("\./(styles/[^?]+)\?', entry)
    return '\n'.join((ROOT / 'web' / path).read_text(encoding='utf-8') for path in imports)


class RegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        rate_limit.clear()

    def test_frontend_vad_uses_soft_target_and_hard_max(self) -> None:
        source = frontend_source()
        self.assertIn('const VAD_SOFT_CUT_GRACE_MS = 6_000;', source)
        self.assertIn('function chunkHardMaxMs()', source)
        self.assertIn('if (elapsedMs >= chunkHardMaxMs()) {', source)
        self.assertIn('shouldCutChunkOnSilence({ relaxed: elapsedMs >= appDependencies.state.chunkMs })', source)


    def test_frontend_ws_url_uses_server_capability_path(self) -> None:
        app_source = frontend_source()
        protocol_source = (ROOT / 'web' / 'src' / 'transcription' / 'websocket.js').read_text(encoding='utf-8')
        self.assertIn('appDependencies.state.wsPath = normalizeWsPath(health.wsPath || appDependencies.state.wsPath);', app_source)
        self.assertIn('return buildWebSocketUrl(location, appDependencies.state.wsPath);', app_source)
        self.assertIn('normalizeWsPath(path)', protocol_source)


    def test_history_ui_shows_retention_countdown_and_no_header_toggle(self) -> None:
        app_source = frontend_source()
        index_source = (ROOT / 'web' / 'index.html').read_text(encoding='utf-8')
        help_source = (ROOT / 'web' / 'help.html').read_text(encoding='utf-8')
        self.assertIn('function formatHistoryDaysRemaining(item)', app_source)
        self.assertIn('あと${Math.ceil(remainingMs / (24 * 60 * 60 * 1000))}日', app_source)
        self.assertNotIn('id="historyToggleBtn"', index_source)
        self.assertIn('一定日数で自動削除', help_source)


    def test_frontend_workspace_guard_uses_live_auth_state(self) -> None:
        source = frontend_source()
        self.assertIn('import { canUseWorkspace as canUseWorkspaceForAuth, persistGuestMode, readGuestMode, serializeUserLabel } from "./auth/session.js";', source)
        self.assertIn('return canUseWorkspaceForAuth(appDependencies.state.auth);', source)
        self.assertIsNone(re.search(r'canUseWorkspaceForAuth\(\s*\)', source))


    def test_frontend_includes_display_name_editor(self) -> None:
        app_source = frontend_source()
        index_source = (ROOT / 'web' / 'index.html').read_text(encoding='utf-8')
        api_source = (ROOT / 'web' / 'src' / 'auth' / 'api.js').read_text(encoding='utf-8')
        self.assertIn('const authProfileEditBtn = $("#authProfileEditBtn");', app_source)
        self.assertIn('async function saveDisplayName()', app_source)
        self.assertIn('id="authProfileDisplayName"', index_source)
        self.assertIn('export function updateDisplayNameRequest(displayName)', api_source)


    def test_frontend_brand_title_wraps_instead_of_truncating(self) -> None:
        style_source = stylesheet_source()
        match = re.search(r'\.brand-text h1 \{(?P<body>.*?)\n\}', style_source, re.DOTALL)
        self.assertIsNotNone(match)
        block = match.group('body')
        self.assertIn('white-space: normal;', block)
        self.assertIn('overflow-wrap: anywhere;', block)
        self.assertNotIn('text-overflow: ellipsis;', block)
        self.assertNotIn('max-width: 200px;', block)


    def test_admin_frontend_includes_user_search(self) -> None:
        html_source = (ROOT / 'web' / 'admin.html').read_text(encoding='utf-8')
        js_source = (ROOT / 'web' / 'admin.js').read_text(encoding='utf-8')
        self.assertIn('id="userSearchInput"', html_source)
        self.assertIn('表示名またはメールアドレスで検索', html_source)
        self.assertIn('const userSearchInputEl = document.querySelector("#userSearchInput");', js_source)
        self.assertIn('q=${encodeURIComponent(userQuery)}', js_source)
        self.assertIn('offset=${usersOffset}', js_source)

