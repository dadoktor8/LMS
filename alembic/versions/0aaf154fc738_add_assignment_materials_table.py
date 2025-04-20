"""add assignment_materials table

Revision ID: 0aaf154fc738
Revises: 69991fb6c791
Create Date: 2025-04-20 21:34:21.456685

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0aaf154fc738'
down_revision = '69991fb6c791'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'assignment_materials',
        sa.Column('assignment_id', sa.Integer(), sa.ForeignKey('assignments.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('material_id', sa.Integer(), sa.ForeignKey('course_materials.id', ondelete='CASCADE'), primary_key=True),
    )

def downgrade():
    op.drop_table('assignment_materials')