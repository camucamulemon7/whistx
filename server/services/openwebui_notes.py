"""Explicit, owner-bound Notes export. Never use the shared generation key.

OpenWebUI 0.11.4 creates server-generated note IDs without idempotency support.
A durable intent under a process-shared lock prevents repeat POSTs after an
ambiguous response. An explicit retry reconciles by our unique metadata marker;
it never overwrites an existing note or blindly repeats an uncertain POST.
"""
from __future__ import annotations

import fcntl
import hashlib
import json

import httpx

from ..core.config import settings
from .meeting_intelligence import read_insights, recap_markdown
from .meeting_source import MeetingError, MeetingSnapshot, read_json, write_json_atomic


class NotesError(MeetingError):
    pass


def _request(client: httpx.Client, method: str, path: str, **kwargs):
    try:
        response = client.request(method, path, **kwargs)
    except httpx.HTTPError:
        raise NotesError('notes_connection_failed', 502) from None
    if response.status_code in {401, 403}:
        raise NotesError('notes_auth_required', 401)
    if not response.is_success:
        raise NotesError('notes_upstream_failed', 502)
    try:
        return response.json()
    except ValueError:
        raise NotesError('notes_invalid_response', 502) from None


def _owned_note(note, owner: str, export_key: str) -> bool:
    return (isinstance(note, dict) and note.get('user_id') == owner
            and isinstance(note.get('meta'), dict)
            and note['meta'].get('whistx_export_key') == export_key
            and isinstance(note.get('id'), str) and bool(note['id']))


def save_recap(snapshot: MeetingSnapshot, *, user_id: int, email: str,
               token: str, title: str, client: httpx.Client | None = None) -> dict:
    if not settings.openwebui_base_url:
        raise NotesError('openwebui_not_configured', 503)
    if not token or len(token) > 8192 or any(char.isspace() for char in token):
        raise NotesError('notes_auth_required', 401)
    recap = read_insights(snapshot).get('recap')
    if not recap or recap.get('stale'):
        raise NotesError('notes_current_recap_required', 409)
    markdown = recap_markdown(recap)
    if not markdown.strip() or len(markdown) > 200_000:
        raise NotesError('notes_recap_size_invalid', 413)
    title = title.strip() or 'Whistx 議事録'
    canonical = json.dumps([settings.openwebui_base_url, user_id, snapshot.key, markdown], ensure_ascii=False)
    export_key = hashlib.sha256(canonical.encode()).hexdigest()
    marker = f'whistx-{export_key[:24]}'
    root = settings.app_data_dir / 'notes_exports' / str(user_id)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt_path = root / f'{export_key}.json'
    owned_client = client is None
    if client is None:
        client = httpx.Client(base_url=f'{settings.openwebui_base_url}/api/v1/',
                              headers={'Authorization': f'Bearer {token}'},
                              timeout=20, follow_redirects=False, trust_env=False)
    try:
        # A token proves access to this OpenWebUI user; email binds the destination
        # to the authenticated Whistx account rather than a shared admin account.
        identity = _request(client, 'GET', 'auths/')
        if (not isinstance(identity, dict) or not identity.get('id')
                or str(identity.get('email') or '').casefold() != email.casefold()):
            raise NotesError('notes_account_mismatch', 403)
        owner = str(identity['id'])
        with receipt_path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt = read_json(receipt_path)
            if receipt and receipt.get('owner') != owner:
                raise NotesError('notes_account_mismatch', 403)
            if receipt.get('state') == 'saved':
                return {'ok': True, 'noteId': receipt['noteId'], 'alreadySaved': True}
            if receipt:
                result = _request(client, 'GET', 'notes/search', params={'query': marker, 'page': 1})
                if not isinstance(result, dict) or not isinstance(result.get('items'), list):
                    raise NotesError('notes_invalid_response', 502)
                for note in result['items']:
                    if _owned_note(note, owner, export_key):
                        write_json_atomic(receipt_path, {'state': 'saved', 'owner': owner, 'noteId': note['id']})
                        return {'ok': True, 'noteId': note['id'], 'alreadySaved': True}
                # The request may still be executing upstream. No second POST.
                raise NotesError('notes_save_uncertain', 409)
            write_json_atomic(receipt_path, {'state': 'sending', 'owner': owner})
            body = {'title': f'{title} [{marker}]', 'data': {'content': {'md': markdown}},
                    'meta': {'whistx_export_key': export_key}, 'access_grants': []}
            try:
                response = client.post('notes/create', json=body)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                receipt_path.unlink(missing_ok=True)
                raise NotesError('notes_connection_failed', 502) from None
            except httpx.HTTPError:
                raise NotesError('notes_save_uncertain', 409) from None
            if response.status_code in {401, 403, 422}:
                # These responses reject the mutation. An explicit retry is safe.
                receipt_path.unlink(missing_ok=True)
                code = 'notes_auth_required' if response.status_code in {401, 403} else 'notes_save_rejected'
                raise NotesError(code, 401 if code == 'notes_auth_required' else 502)
            try:
                note = response.json() if response.is_success else None
            except ValueError:
                note = None
            if not _owned_note(note, owner, export_key) or note.get('access_grants') != []:
                raise NotesError('notes_save_uncertain', 409)
            write_json_atomic(receipt_path, {'state': 'saved', 'owner': owner, 'noteId': note['id']})
            return {'ok': True, 'noteId': note['id'], 'alreadySaved': False}
    finally:
        if owned_client:
            client.close()
