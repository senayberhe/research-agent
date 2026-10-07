"""job event metadata json

Revision ID: 9e44b0dfce26
Revises: c55e3c6c8391
Create Date: 2026-10-06 14:40:56.423717

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9e44b0dfce26'
down_revision: Union[str, Sequence[str], None] = 'c55e3c6c8391'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Moves attempt, worker_id, tool and duration_ms into metadata_json
    (a JSON object, nulls left out) and indexes event_type and created_at."""
    op.add_column('job_events', sa.Column('metadata_json', sa.Text(), nullable=True))

    # Existing events keep their details.
    op.execute(
        """
        UPDATE job_events
        SET metadata_json = NULLIF(
            jsonb_strip_nulls(
                jsonb_build_object(
                    'attempt', attempt,
                    'worker_id', worker_id,
                    'tool', tool,
                    'duration_ms', duration_ms
                )
            ),
            '{}'::jsonb
        )::text
        """
    )

    op.create_index(op.f('ix_job_events_created_at'), 'job_events', ['created_at'], unique=False)
    op.create_index(op.f('ix_job_events_event_type'), 'job_events', ['event_type'], unique=False)
    op.drop_column('job_events', 'attempt')
    op.drop_column('job_events', 'tool')
    op.drop_column('job_events', 'worker_id')
    op.drop_column('job_events', 'duration_ms')


def downgrade() -> None:
    """Moves the details back out of metadata_json into their columns
    (other metadata is lost)."""
    op.add_column('job_events', sa.Column('duration_ms', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True))
    op.add_column('job_events', sa.Column('worker_id', sa.VARCHAR(length=255), autoincrement=False, nullable=True))
    op.add_column('job_events', sa.Column('tool', sa.VARCHAR(length=50), autoincrement=False, nullable=True))
    # Nullable until it's filled in below.
    op.add_column('job_events', sa.Column('attempt', sa.INTEGER(), autoincrement=False, nullable=True))

    op.execute(
        """
        UPDATE job_events
        SET attempt = COALESCE((metadata_json::jsonb ->> 'attempt')::int, 0),
            worker_id = metadata_json::jsonb ->> 'worker_id',
            tool = metadata_json::jsonb ->> 'tool',
            duration_ms = (metadata_json::jsonb ->> 'duration_ms')::double precision
        """
    )

    op.alter_column('job_events', 'attempt', nullable=False)
    op.drop_index(op.f('ix_job_events_event_type'), table_name='job_events')
    op.drop_index(op.f('ix_job_events_created_at'), table_name='job_events')
    op.drop_column('job_events', 'metadata_json')
