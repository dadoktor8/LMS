"""added teacher id to assignments

Revision ID: 69991fb6c791
Revises: 67c60af8f722
Create Date: 2025-04-20 19:26:55.939710

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '69991fb6c791'
down_revision: Union[str, None] = '67c60af8f722'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    with op.batch_alter_table("assignments", schema=None) as batch_op:
        batch_op.add_column(sa.Column("teacher_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_teacher_id_user", "users", ["teacher_id"], ["id"]
        )


def downgrade():
    with op.batch_alter_table("assignments", schema=None) as batch_op:
        batch_op.drop_constraint("fk_teacher_id_user", type_="foreignkey")
        batch_op.drop_column("teacher_id")
