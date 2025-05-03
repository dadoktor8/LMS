"""Added embedding quota to db
Revision ID: 7bc08a254c16
Revises: 195255f6eb21
Create Date: 2025-05-03 16:06:25.951006
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '7bc08a254c16'
down_revision = '195255f6eb21'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'pdf_quota_usage',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('course_id', sa.Integer(), nullable=False),
        sa.Column('usage_date', sa.Date(), nullable=False),
        sa.Column('pages_processed', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['course_id'], ['courses.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_pdf_quota_usage_id'), 'pdf_quota_usage', ['id'], unique=False)
    # REMOVE or COMMENT OUT the problem lines below
    # op.drop_constraint(...)
    # op.drop_constraint(...)
    # op.create_foreign_key(...)
    # op.create_foreign_key(...)

def downgrade():
    # Similarly, don't touch constraints you didn't create in upgrade!
    op.drop_index(op.f('ix_pdf_quota_usage_id'), table_name='pdf_quota_usage')
    op.drop_table('pdf_quota_usage')