"""add rubric evaluation tables

Revision ID: 93a29dd880fd
Revises: cb6f14982320
Create Date: 2025-04-30 13:29:46.586143
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '93a29dd880fd'
down_revision: Union[str, None] = 'cb6f14982320'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Create rubric_evaluations table
    """
    op.create_table(
        'rubric_evaluations',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('submission_id', sa.Integer(), sa.ForeignKey('assignment_submissions.id', ondelete="CASCADE"), nullable=False),
        sa.Column('criterion_id', sa.Integer(), sa.ForeignKey('rubric_criteria.id', ondelete="CASCADE"), nullable=False),
        sa.Column('level_id', sa.Integer(), sa.ForeignKey('rubric_levels.id', ondelete="CASCADE"), nullable=True),
        sa.Column('points_awarded', sa.Float(), nullable=False),
        sa.Column('feedback', sa.Text(), nullable=True),
        sa.Column('graded_by_ai', sa.Boolean(), default=False),
        sa.Column('graded_by_user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete="SET NULL"), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)')),
    )"""

    ### -- For constraint adjustments, use batch mode for SQLite --
    # If you MUST alter 'assignment_materials' constraints, you'll need the exact constraint names.
    # Example below (EDIT constraint names as needed!):
    with op.batch_alter_table('assignment_materials') as batch_op:
        # batch_op.drop_constraint('fk_assignment_materials_material_id_course_materials', type_='foreignkey')
        # batch_op.drop_constraint('fk_assignment_materials_assignment_id_assignments', type_='foreignkey')
        # batch_op.create_foreign_key('fk_assignment_materials_material_id_course_materials', 'course_materials', ['material_id'], ['id'])
        # batch_op.create_foreign_key('fk_assignment_materials_assignment_id_assignments', 'assignments', ['assignment_id'], ['id'])
        pass  # REMOVE or EDIT this, only if you need constraint changes

def downgrade() -> None:
    # Drop the rubric_evaluations table
    op.drop_table('rubric_evaluations')

    # Undo assignment_materials constraint changes (edit constraint names as needed)
    with op.batch_alter_table('assignment_materials') as batch_op:
        # batch_op.drop_constraint('fk_assignment_materials_material_id_course_materials', type_='foreignkey')
        # batch_op.drop_constraint('fk_assignment_materials_assignment_id_assignments', type_='foreignkey')
        # batch_op.create_foreign_key('fk_assignment_materials_material_id_course_materials', 'course_materials', ['material_id'], ['id'], ondelete='CASCADE')
        # batch_op.create_foreign_key('fk_assignment_materials_assignment_id_assignments', 'assignments', ['assignment_id'], ['id'], ondelete='CASCADE')
        pass  # REMOVE or EDIT this, only if you need constraint rollbacks