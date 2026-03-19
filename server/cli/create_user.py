from __future__ import annotations

import argparse

from sqlalchemy import select

from ..auth import create_user
from ..db import db_session, init_db
from ..models import User


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a whistx user")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--display-name", default="")
    parser.add_argument("--admin", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(args.password) < 8:
        raise SystemExit("password must be at least 8 characters")
    init_db()
    with db_session() as db:
        existing = db.scalar(select(User).where(User.email == args.email.strip().lower()))
        if existing is not None:
            raise SystemExit(f"user already exists: {args.email}")
        create_user(
            db,
            email=args.email,
            password=args.password,
            display_name=args.display_name,
            is_admin=bool(args.admin),
        )
    print(f"created user: {args.email}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
