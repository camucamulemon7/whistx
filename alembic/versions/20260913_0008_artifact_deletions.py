"""durable artifact deletion outbox"""
from alembic import op
import sqlalchemy as sa

revision = '20260913_0008'
down_revision = '20260727_0007'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('artifact_deletions',
        sa.Column('history_id', sa.String(64), primary_key=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('artifact_key', sa.String(1024), nullable=True),
        sa.Column('requested_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('attempts', sa.Integer(), nullable=False),
        sa.Column('last_error', sa.String(128), nullable=True))


def downgrade():
    op.drop_table('artifact_deletions')
