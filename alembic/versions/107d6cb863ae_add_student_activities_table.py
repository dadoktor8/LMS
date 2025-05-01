"""Add student_activities table

Revision ID: 107d6cb863ae
Revises: 93a29dd880fd
Create Date: 2025-05-01 15:05:59.766167

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '107d6cb863ae'
down_revision: Union[str, None] = '93a29dd880fd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    '''op.create_table(
        'student_activities',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('student_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('course_id', sa.Integer(), sa.ForeignKey('courses.id', ondelete='CASCADE'), nullable=False),
        sa.Column('activity_type', sa.String(length=50), nullable=False),
        sa.Column('topic', sa.String(length=255), nullable=False),
        sa.Column('user_input', sa.Text(), nullable=False),
        sa.Column('ai_response', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )'''

def downgrade() -> None:
    op.drop_table('student_activities')
