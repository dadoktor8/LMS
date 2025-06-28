"""module to individual activities
Revision ID: 9961d18d89e6
Revises: 7ca061117206
Create Date: 2025-06-28 10:15:36.079308
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '9961d18d89e6'
down_revision: Union[str, None] = '7ca061117206'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Upgrade schema."""
    # Use batch mode for SQLite compatibility
    with op.batch_alter_table('student_activities', schema=None) as batch_op:
        batch_op.add_column(sa.Column('module_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            'fk_student_activities_module_id',  # Give it a name
            'course_modules', 
            ['module_id'], 
            ['id'], 
            ondelete='SET NULL'
        )

def downgrade() -> None:
    """Downgrade schema."""
    # Use batch mode for SQLite compatibility
    with op.batch_alter_table('student_activities', schema=None) as batch_op:
        batch_op.drop_constraint('fk_student_activities_module_id', type_='foreignkey')
        batch_op.drop_column('module_id')