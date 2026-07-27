"""serialize initial administrator bootstrap"""

import sqlalchemy as sa
from alembic import op


revision = "20260727_0005"
down_revision = "20260322_0004"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "admin_bootstrap_state",
        sa.Column("key", sa.String(length=32), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade():
    op.drop_table("admin_bootstrap_state")
