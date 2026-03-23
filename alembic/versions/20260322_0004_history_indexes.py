"""add history list and segment indexes"""

from alembic import op


revision = "20260322_0004"
down_revision = "20260319_0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index(
        "ix_transcript_histories_user_saved_at",
        "transcript_histories",
        ["user_id", "saved_at"],
        unique=False,
    )
    op.create_index(
        "ix_transcript_histories_user_runtime_session",
        "transcript_histories",
        ["user_id", "runtime_session_id"],
        unique=False,
    )
    op.create_index(
        "ix_transcript_segments_history_seq",
        "transcript_segments",
        ["history_id", "seq"],
        unique=False,
    )


def downgrade():
    op.drop_index("ix_transcript_segments_history_seq", table_name="transcript_segments")
    op.drop_index("ix_transcript_histories_user_runtime_session", table_name="transcript_histories")
    op.drop_index("ix_transcript_histories_user_saved_at", table_name="transcript_histories")
