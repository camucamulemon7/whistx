"""Index literal substring searches on PostgreSQL and stable admin paging."""
from alembic import op

revision = '20261004_0009'
down_revision = '20260913_0008'
branch_labels = None
depends_on = None

TRIGRAM_INDEXES = {
    'ix_history_title_trgm': ('transcript_histories', 'title'),
    'ix_history_text_trgm': ('transcript_histories', 'plain_text'),
    'ix_user_email_trgm': ('users', 'email'),
    'ix_user_display_name_trgm': ('users', 'display_name'),
}


def upgrade():
    op.create_index('ix_users_created_id', 'users', ['created_at', 'id'])
    op.create_index('ix_users_pending_created_id', 'users', ['is_active', 'approved_at', 'created_at', 'id'])
    if op.get_bind().dialect.name == 'postgresql':
        op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
        for name, (table, column) in TRIGRAM_INDEXES.items():
            op.create_index(name, table, [column], postgresql_using='gin', postgresql_ops={column: 'gin_trgm_ops'})


def downgrade():
    if op.get_bind().dialect.name == 'postgresql':
        for name, (table, _) in reversed(list(TRIGRAM_INDEXES.items())):
            op.drop_index(name, table_name=table)
    op.drop_index('ix_users_pending_created_id', table_name='users')
    op.drop_index('ix_users_created_id', table_name='users')
    # pg_trgm may be shared by other applications; leave the extension installed.
