"""add shared rate limits and connection leases"""

import sqlalchemy as sa
from alembic import op


revision = "20260727_0006"
down_revision = "20260727_0005"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "rate_limit_buckets",
        sa.Column("key", sa.String(length=512), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("window_ends_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_index(
        "ix_rate_limit_buckets_window_ends_at",
        "rate_limit_buckets",
        ["window_ends_at"],
        unique=False,
    )
    op.create_table(
        "connection_quota_locks",
        sa.Column("key", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )
    op.bulk_insert(
        sa.table("connection_quota_locks", sa.column("key", sa.String(length=32))),
        [{"key": "connections"}],
    )
    op.create_table(
        "connection_leases",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("is_guest", sa.Boolean(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_connection_leases_subject", "connection_leases", ["subject"], unique=False)
    op.create_index("ix_connection_leases_is_guest", "connection_leases", ["is_guest"], unique=False)
    op.create_index("ix_connection_leases_expires_at", "connection_leases", ["expires_at"], unique=False)


def downgrade():
    op.drop_index("ix_connection_leases_expires_at", table_name="connection_leases")
    op.drop_index("ix_connection_leases_is_guest", table_name="connection_leases")
    op.drop_index("ix_connection_leases_subject", table_name="connection_leases")
    op.drop_table("connection_leases")
    op.drop_table("connection_quota_locks")
    op.drop_index("ix_rate_limit_buckets_window_ends_at", table_name="rate_limit_buckets")
    op.drop_table("rate_limit_buckets")
