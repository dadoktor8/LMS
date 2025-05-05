"""Addedflashcard quota

Revision ID: 813322c73763
Revises: adab82765aa3
Create Date: 2025-05-05 15:50:26.215409

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '813322c73763'
down_revision: Union[str, None] = 'adab82765aa3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
   ''' op.create_table(
        'flashcard_usages',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('student_id', sa.String(), nullable=False, index=True),
        sa.Column('course_id', sa.Integer(), nullable=False, index=True),
        sa.Column('usage_date', sa.Date(), nullable=False),
        sa.Column('count', sa.Integer(), nullable=False),
    )'''

def downgrade():
    op.drop_table('flashcard_usages')
