"""assignments and modules

Revision ID: 684690eb3c7b
Revises: 85807f813a80
Create Date: 2025-06-21 13:59:03.331438

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '684690eb3c7b'
down_revision: Union[str, None] = '85807f813a80'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add the column first
    #op.add_column('assignments', sa.Column('module_id', sa.Integer(), nullable=True))

    # Use batch_alter_table to add the foreign key constraint for SQLite
    with op.batch_alter_table('assignments') as batch_op:
        batch_op.create_foreign_key(
            'fk_assignments_module_id_course_modules',  # give the constraint a name
            'course_modules',
            ['module_id'],
            ['id']
        )


def downgrade() -> None:
    with op.batch_alter_table('assignments') as batch_op:
        batch_op.drop_constraint('fk_assignments_module_id_course_modules', type_='foreignkey')

    op.drop_column('assignments', 'module_id')
