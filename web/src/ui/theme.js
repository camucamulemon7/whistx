export function normalizeTheme(value) {
  return value === 'dark' ? 'dark' : 'light';
}

export function resolveInitialTheme(savedTheme, prefersDark) {
  if (savedTheme === 'light' || savedTheme === 'dark') {
    return savedTheme;
  }
  return prefersDark ? 'dark' : 'light';
}

export function themeMetaColor(theme) {
  return theme === 'dark' ? '#0a0a0a' : '#f5f5f7';
}
