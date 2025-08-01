"""Added study-guides quota

Revision ID: caa5c5a6e75c
Revises: 813322c73763
Create Date: 2025-05-05 20:42:47.734648

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'caa5c5a6e75c'
down_revision: Union[str, None] = '813322c73763'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('studyguide_usages',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('student_id', sa.String(), nullable=True),
    sa.Column('course_id', sa.Integer(), nullable=True),
    sa.Column('usage_date', sa.Date(), nullable=True),
    sa.Column('count', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('studyguide_usages')
    # ### end Alembic commands ###
