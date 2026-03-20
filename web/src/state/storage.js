export function readStoredValue(key, fallback = null) {
  try {
    const value = localStorage.getItem(key);
    return value == null ? fallback : value;
  } catch {
    return fallback;
  }
}

export function writeStoredValue(key, value) {
  try {
    if (value == null) {
      localStorage.removeItem(key);
      return;
    }
    localStorage.setItem(key, value);
  } catch {
    // ignore
  }
}

export function readStoredJson(key, fallback = null) {
  const raw = readStoredValue(key, null);
  if (!raw) {
    return fallback;
  }
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function writeStoredJson(key, value) {
  if (value == null) {
    writeStoredValue(key, null);
    return;
  }
  writeStoredValue(key, JSON.stringify(value));
}
