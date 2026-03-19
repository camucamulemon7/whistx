"""add user approval columns"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260319_0002"
down_revision = "20260319_0001"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("users")}

    if "approved_at" not in columns:
        op.add_column("users", sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True))
    if "approved_by_user_id" not in columns:
        op.add_column("users", sa.Column("approved_by_user_id", sa.Integer(), nullable=True))

    if bind.dialect.name != "sqlite":
        foreign_keys = {fk.get("name") for fk in inspector.get_foreign_keys("users")}
        if "fk_users_approved_by_user_id" not in foreign_keys:
            op.create_foreign_key(
                "fk_users_approved_by_user_id",
                "users",
                "users",
                ["approved_by_user_id"],
                ["id"],
                ondelete="SET NULL",
            )
    op.execute("UPDATE users SET approved_at = created_at WHERE is_active = 1 AND approved_at IS NULL")


def downgrade():
    op.drop_constraint("fk_users_approved_by_user_id", "users", type_="foreignkey")
    op.drop_column("users", "approved_by_user_id")
    op.drop_column("users", "approved_at")
