"""convert_teacher_id_to_integer

Revision ID: f0f482a4183d
Revises: 02637dc934ef
Create Date: 2025-05-15 20:11:26.902678

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f0f482a4183d'
down_revision: Union[str, None] = '02637dc934ef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Get connection and dialect info
    connection = op.get_bind()
    dialect_name = connection.dialect.name
    
    # Create a backup table (for safety)
    if dialect_name == 'postgresql':
        op.execute('CREATE TABLE IF NOT EXISTS quiz_quotas_backup AS SELECT * FROM quiz_quotas')
    
    # Different approach based on dialect
    if dialect_name == 'postgresql':
        # PostgreSQL approach - direct ALTER COLUMN TYPE
        op.execute('ALTER TABLE quiz_quotas ALTER COLUMN teacher_id TYPE INTEGER USING (teacher_id::integer)')
        print("PostgreSQL: Successfully converted teacher_id column to INTEGER type.")
    
    elif dialect_name == 'sqlite':
        # SQLite approach - create new table and copy data
        # Get table information
        table_name = 'quiz_quotas'
        
        # Step 1: Create new table with correct schema
        op.execute('''
            CREATE TABLE quiz_quotas_new (
                id INTEGER PRIMARY KEY,
                teacher_id INTEGER,
                course_id INTEGER,
                date DATE,
                count INTEGER DEFAULT 0
            )
        ''')
        
        # Step 2: Copy data with conversion
        op.execute('''
            INSERT INTO quiz_quotas_new (id, teacher_id, course_id, date, count)
            SELECT id, CAST(teacher_id AS INTEGER), course_id, date, count
            FROM quiz_quotas
        ''')
        
        # Step 3: Drop old table and rename new one
        op.execute('DROP TABLE quiz_quotas')
        op.execute('ALTER TABLE quiz_quotas_new RENAME TO quiz_quotas')
        
        # Step 4: Recreate indices
        op.execute('CREATE INDEX ix_quiz_quotas_id ON quiz_quotas (id)')
        op.execute('CREATE INDEX ix_quiz_quotas_teacher_id ON quiz_quotas (teacher_id)')
        op.execute('CREATE INDEX ix_quiz_quotas_course_id ON quiz_quotas (course_id)')
        op.execute('CREATE INDEX ix_quiz_quotas_date ON quiz_quotas (date)')
        
        print("SQLite: Successfully rebuilt quiz_quotas table with teacher_id as INTEGER.")
    
    else:
        raise Exception(f"Unsupported database dialect: {dialect_name}")

def downgrade():
    # Get connection and dialect info
    connection = op.get_bind()
    dialect_name = connection.dialect.name
    
    if dialect_name == 'postgresql':
        # Convert back to varchar for PostgreSQL
        op.execute('ALTER TABLE quiz_quotas ALTER COLUMN teacher_id TYPE VARCHAR')
    
    elif dialect_name == 'sqlite':
        # SQLite approach - recreate table
        op.execute('''
            CREATE TABLE quiz_quotas_new (
                id INTEGER PRIMARY KEY,
                teacher_id VARCHAR,
                course_id INTEGER,
                date DATE,
                count INTEGER DEFAULT 0
            )
        ''')
        
        op.execute('''
            INSERT INTO quiz_quotas_new (id, teacher_id, course_id, date, count)
            SELECT id, CAST(teacher_id AS TEXT), course_id, date, count
            FROM quiz_quotas
        ''')
        
        op.execute('DROP TABLE quiz_quotas')
        op.execute('ALTER TABLE quiz_quotas_new RENAME TO quiz_quotas')
        
        # Recreate indices
        op.execute('CREATE INDEX ix_quiz_quotas_id ON quiz_quotas (id)')
        op.execute('CREATE INDEX ix_quiz_quotas_teacher_id ON quiz_quotas (teacher_id)')
        op.execute('CREATE INDEX ix_quiz_quotas_course_id ON quiz_quotas (course_id)')
        op.execute('CREATE INDEX ix_quiz_quotas_date ON quiz_quotas (date)')
