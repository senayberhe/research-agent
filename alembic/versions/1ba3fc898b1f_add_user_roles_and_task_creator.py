"""add user roles and task creator

Revision ID: 1ba3fc898b1f
Revises: 15a70fa66c4f
Create Date: 2026-10-07 13:54:41.839031

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1ba3fc898b1f'
down_revision: Union[str, Sequence[str], None] = '15a70fa66c4f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


FK_NAME = "fk_research_tasks_created_by_user_id_users"


def upgrade() -> None:
    """User roles (viewer, researcher, admin) and who started each task.

    Existing users had full access before roles existed, so they become
    admins (nobody is locked out); new users default to viewer."""
    op.add_column('users', sa.Column('role', sa.String(length=20), server_default='viewer', nullable=False))
    op.execute("UPDATE users SET role = 'admin'")

    op.add_column('research_tasks', sa.Column('created_by_user_id', sa.Integer(), nullable=True))
    op.create_index(op.f('ix_research_tasks_created_by_user_id'), 'research_tasks', ['created_by_user_id'], unique=False)
    op.create_foreign_key(FK_NAME, 'research_tasks', 'users', ['created_by_user_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    op.drop_constraint(FK_NAME, 'research_tasks', type_='foreignkey')
    op.drop_index(op.f('ix_research_tasks_created_by_user_id'), table_name='research_tasks')
    op.drop_column('research_tasks', 'created_by_user_id')
    op.drop_column('users', 'role')
