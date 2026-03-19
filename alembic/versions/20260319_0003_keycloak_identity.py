"""add oidc identity columns"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260319_0003"
down_revision = "20260319_0002"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("users")}

    if "auth_provider" not in columns:
        op.add_column("users", sa.Column("auth_provider", sa.String(length=32), nullable=True))
    if "auth_subject" not in columns:
        op.add_column("users", sa.Column("auth_subject", sa.String(length=255), nullable=True))

    indexes = {index["name"] for index in inspector.get_indexes("users")}
    if "uq_users_auth_identity" not in indexes:
        op.create_index("uq_users_auth_identity", "users", ["auth_provider", "auth_subject"], unique=True)


def downgrade():
    bind = op.get_bind()
    inspector = inspect(bind)
    indexes = {index["name"] for index in inspector.get_indexes("users")}
    if "uq_users_auth_identity" in indexes:
        op.drop_index("uq_users_auth_identity", table_name="users")

    columns = {column["name"] for column in inspector.get_columns("users")}
    if "auth_subject" in columns:
        op.drop_column("users", "auth_subject")
    if "auth_provider" in columns:
        op.drop_column("users", "auth_provider")
