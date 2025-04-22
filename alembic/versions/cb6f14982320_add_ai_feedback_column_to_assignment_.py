"""add ai_feedback column to assignment_submissions

Revision ID: cb6f14982320
Revises: 0aaf154fc738
Create Date: 2025-04-22

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'cb6f14982320'
down_revision = '0aaf154fc738'
branch_labels = None
depends_on = None

def upgrade():
    # Use batch operations for SQLite
    with op.batch_alter_table('assignment_submissions') as batch_op:
        batch_op.add_column(sa.Column('ai_feedback', sa.Text(), nullable=True))

def downgrade():
    with op.batch_alter_table('assignment_submissions') as batch_op:
        batch_op.drop_column('ai_feedback')