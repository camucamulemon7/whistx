"""Check tracked file modes and contributor entry points without dependencies."""
from pathlib import Path
import subprocess
import sys


def main() -> int:
    errors = []
    entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z']).decode().split('\0')
    for entry in filter(None, entries):
        metadata, name = entry.split('\t', 1)
        if metadata.split()[0] == '100755' and not name.endswith('.sh'):
            errors.append(f'Unexpected executable bit: {name}')
    for name in ['CONTRIBUTING.md', 'SECURITY.md', 'docs/operations.md', 'Makefile']:
        if not Path(name).is_file():
            errors.append(f'Missing required document: {name}')
    result = subprocess.run(['git', 'check-ignore', '--no-index', 'docs/new-contributor-document.md'], capture_output=True)
    if result.returncode == 0:
        errors.append('New documentation is ignored')
    elif result.returncode != 1:
        errors.append('Could not check documentation ignore rules')
    for error in errors:
        print(error, file=sys.stderr)
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
