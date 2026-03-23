from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from ..auth import approve_user
from ..core.security import serialize_user
from ..models import User
from ..repositories import user_repository


class AdminServiceError(Exception):
    def __init__(self, code: str, status_code: int):
        super().__init__(code)
        self.code = code
        self.status_code = status_code


def list_pending_users_payload(db: Session) -> dict[str, Any]:
    items = [
        {
            'id': item.id,
            'email': item.email,
            'displayName': item.display_name,
            'createdAt': item.created_at.isoformat(),
        }
        for item in user_repository.list_pending_users(db)
    ]
    return {'items': items}


def approve_pending_user(db: Session, *, pending_user_id: int, admin: User) -> dict[str, Any]:
    pending_user = user_repository.get_user_by_id(db, pending_user_id)
    if pending_user is None or pending_user.is_active or pending_user.approved_at is not None:
        raise AdminServiceError('pending_user_not_found', 404)
    approve_user(db, user=pending_user, admin=admin)
    db.commit()
    return {'ok': True, 'user': serialize_user(pending_user)}


def list_users_payload(db: Session) -> dict[str, Any]:
    items = [
        {
            'id': item.id,
            'email': item.email,
            'displayName': item.display_name,
            'isAdmin': bool(item.is_admin),
            'isActive': bool(item.is_active),
            'createdAt': item.created_at.isoformat(),
            'lastLoginAt': item.last_login_at.isoformat() if item.last_login_at else None,
            'approvedAt': item.approved_at.isoformat() if item.approved_at else None,
        }
        for item in user_repository.list_all_users(db)
    ]
    return {'items': items}


def update_user_role(db: Session, *, user_id: int, role: str) -> dict[str, Any]:
    normalized = str(role or '').strip().lower()
    if normalized not in {'admin', 'member'}:
        raise AdminServiceError('invalid_role', 400)
    target = user_repository.get_user_by_id(db, user_id)
    if target is None:
        raise AdminServiceError('user_not_found', 404)
    make_admin = normalized == 'admin'
    if not make_admin and target.is_admin and user_repository.count_admin_users(db) <= 1:
        raise AdminServiceError('last_admin_forbidden', 409)
    target.is_admin = make_admin
    db.commit()
    return {'ok': True, 'user': serialize_user(target)}
