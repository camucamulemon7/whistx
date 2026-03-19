from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .auth import SESSION_COOKIE_NAME, get_user_by_session_id
from .db import get_db
from .models import User


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    user = get_optional_user(request, db)
    if user is None:
        raise HTTPException(status_code=401, detail="login_required")
    return user


def get_current_admin(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    user = get_current_user(request, db)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="admin_required")
    return user


def get_optional_user(
    request: Request,
    db: Session,
) -> User | None:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    return get_user_by_session_id(db, session_id)
