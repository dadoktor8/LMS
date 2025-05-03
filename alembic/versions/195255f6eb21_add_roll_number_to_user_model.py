"""Add roll_number to User model
Revision ID: 195255f6eb21
Revises: 107d6cb863ae
Create Date: 2025-05-03 13:16:45.840927
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '195255f6eb21'
down_revision: Union[str, None] = '107d6cb863ae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.add_column('users', sa.Column('roll_number', sa.String(), nullable=True))

def downgrade() -> None:
    op.drop_column('users', 'roll_number')