import { readStoredValue, writeStoredValue } from '../state/storage.js';

const GUEST_MODE_KEY = 'whistx_guest_mode';

export function serializeUserLabel(user) {
  if (!user) return 'ゲスト';
  return String(user.displayName || user.email || 'ログイン済み');
}

export function canUseWorkspace(authState) {
  return !!authState?.authenticated || !!authState?.isGuest;
}

export function readGuestMode() {
  return readStoredValue(GUEST_MODE_KEY, '0') === '1';
}

export function persistGuestMode(enabled) {
  writeStoredValue(GUEST_MODE_KEY, enabled ? '1' : null);
}
