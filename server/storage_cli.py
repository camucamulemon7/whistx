"""Operator artifact audit. Defaults to a read-only scan."""
import argparse
import json
from .core.config import settings
from .db import db_session
from .services.artifact_deletion import process_deletions
from .services.artifact_reconciliation import quarantine, scan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='quarantine old unreferenced artifacts')
    parser.add_argument('--offline', action='store_true', help='confirm all application/cleanup writers have been stopped')
    parser.add_argument('--retry-deletions', action='store_true')
    args = parser.parse_args()
    if args.apply and not args.offline:
        parser.error('--apply requires --offline and stopped application/cleanup writers')
    with db_session() as db:
        if args.retry_deletions:
            print(json.dumps(process_deletions(db, settings.history_dir)))
        operation = quarantine if args.apply else scan
        print(json.dumps(operation(db, settings.history_dir), indent=2))


if __name__ == '__main__':
    main()
