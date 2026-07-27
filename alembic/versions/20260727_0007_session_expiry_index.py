"""index user session expiry for scheduled cleanup"""

from alembic import op


revision = "20260727_0007"
down_revision = "20260727_0006"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index(
        "ix_user_sessions_expires_at",
        "user_sessions",
        ["expires_at"],
        unique=False,
    )


def downgrade():
    op.drop_index("ix_user_sessions_expires_at", table_name="user_sessions")
