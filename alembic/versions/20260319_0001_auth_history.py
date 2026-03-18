"""auth and history tables"""

from alembic import op
import sqlalchemy as sa


revision = "20260319_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "user_sessions",
        sa.Column("id", sa.String(length=128), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_agent", sa.String(length=512), nullable=True),
        sa.Column("ip_address", sa.String(length=128), nullable=True),
    )
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"], unique=False)

    op.create_table(
        "transcript_histories",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("runtime_session_id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=True),
        sa.Column("audio_source", sa.String(length=32), nullable=True),
        sa.Column("segment_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("plain_text", sa.Text(), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=True),
        sa.Column("proofread_text", sa.Text(), nullable=True),
        sa.Column("has_diarization", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("artifact_dir", sa.String(length=1024), nullable=True),
        sa.Column("txt_path", sa.String(length=1024), nullable=True),
        sa.Column("jsonl_path", sa.String(length=1024), nullable=True),
        sa.Column("zip_path", sa.String(length=1024), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("saved_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "runtime_session_id", name="uq_history_user_runtime_session"),
    )
    op.create_index("ix_transcript_histories_user_id", "transcript_histories", ["user_id"], unique=False)
    op.create_index(
        "ix_transcript_histories_runtime_session_id",
        "transcript_histories",
        ["runtime_session_id"],
        unique=False,
    )

    op.create_table(
        "transcript_segments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("history_id", sa.String(length=64), sa.ForeignKey("transcript_histories.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("segment_id", sa.String(length=64), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("ts_start", sa.Integer(), nullable=False),
        sa.Column("ts_end", sa.Integer(), nullable=False),
        sa.Column("chunk_offset_ms", sa.Integer(), nullable=True),
        sa.Column("chunk_duration_ms", sa.Integer(), nullable=True),
        sa.Column("language", sa.String(length=32), nullable=True),
        sa.Column("speaker", sa.String(length=64), nullable=True),
        sa.Column("screenshot_path", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_transcript_segments_history_id", "transcript_segments", ["history_id"], unique=False)


def downgrade():
    op.drop_index("ix_transcript_segments_history_id", table_name="transcript_segments")
    op.drop_table("transcript_segments")
    op.drop_index("ix_transcript_histories_runtime_session_id", table_name="transcript_histories")
    op.drop_index("ix_transcript_histories_user_id", table_name="transcript_histories")
    op.drop_table("transcript_histories")
    op.drop_index("ix_user_sessions_user_id", table_name="user_sessions")
    op.drop_table("user_sessions")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
