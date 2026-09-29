"""Checkpoint compacted agent conversation context.

Revision ID: 098
Revises: 097
"""
from alembic import op
import sqlalchemy as sa

revision = "098"
down_revision = "097"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_conversations", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column("agent_conversations", sa.Column("summary_through_ordinal", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_conversations", "summary_through_ordinal")
    op.drop_column("agent_conversations", "summary")
