"""llm cache and calls

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_cache",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("prompt_name", sa.String(length=64), nullable=False),
        sa.Column("prompt_version", sa.Integer(), nullable=False),
        sa.Column("response", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_llm_cache")),
    )
    op.create_table(
        "llm_calls",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("prompt", sa.String(length=80), nullable=False),
        sa.Column("backend", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("attempt", sa.Integer(), server_default="1", nullable=False),
        sa.Column("n_items", sa.Integer(), server_default="1", nullable=False),
        sa.Column("input_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("output_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("cache_read_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("cache_creation_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("cost_usd_est", sa.Float(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), server_default="0", nullable=False),
        sa.Column("auth_source", sa.String(length=64), nullable=True),
        sa.Column("rate_limit_status", sa.String(length=32), nullable=True),
        sa.Column("rate_limit_utilization", sa.Float(), nullable=True),
        sa.Column("cache_key", sa.String(length=64), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], name=op.f("fk_llm_calls_run_id_runs")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_llm_calls")),
    )
    with op.batch_alter_table("llm_calls", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_llm_calls_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_llm_calls_run_id"), ["run_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("llm_calls", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_llm_calls_run_id"))
        batch_op.drop_index(batch_op.f("ix_llm_calls_created_at"))
    op.drop_table("llm_calls")
    op.drop_table("llm_cache")
