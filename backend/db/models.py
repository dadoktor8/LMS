# backend/db/models.py
from datetime import datetime
import traceback
from sqlalchemy import Column, DateTime, Float, Integer, String, Boolean, ForeignKey, Table, Text, func
from backend.db.database import Base
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    is_verified = Column(Boolean, default=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)
    f_name = Column(String, nullable=False)
    l_name = Column(String, nullable=True)
    courses = relationship("Course",back_populates="teacher")
    attendance = relationship("AttendanceRecord", backref="student")
    assignment_submissions = relationship("AssignmentSubmission", back_populates="student", cascade="all, delete-orphan")

class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(String)
    teacher_id = Column(Integer, ForeignKey("users.id"))

    teacher = relationship("User", back_populates="courses")
    enrollments = relationship("Enrollment", back_populates="course")
    attendance_codes = relationship("AttendanceCode", backref="course", cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceRecord", backref="course", cascade="all, delete-orphan")
    tas = relationship("TeachingAssistant", back_populates="course", cascade="all, delete-orphan")
    materials = relationship("CourseMaterial", back_populates="course", cascade="all, delete-orphan")
    processed_materials = relationship("ProcessedMaterial", back_populates="course")
    text_chunks = relationship("TextChunk", back_populates="course")
    assignments = relationship("Assignment", back_populates="course", cascade="all, delete-orphan")


class Enrollment(Base):
    __tablename__ = "Enrollment"
    id = Column(Integer,primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"))
    student_id = Column(Integer, ForeignKey("users.id"))
    is_accepted = Column(Boolean, default=False)
    course = relationship("Course", back_populates="enrollments")
    student = relationship("User")

class CourseInvite(Base):
    __tablename__ = "course_invites"

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"))
    student_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="pending")  # pending | accepted

    course = relationship("Course", backref="invites")
    student = relationship("User")

class AttendanceCode(Base):
    __tablename__ = "attendance_codes"
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"))
    code = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"))
    course_id = Column(Integer, ForeignKey("courses.id"))
    attended_at = Column(DateTime, default=datetime.utcnow)
    code_used = Column(String, nullable=False)


class TeachingAssistant(Base):
    __tablename__ = "teaching_assistants"
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"))
    student_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    course = relationship("Course", back_populates="tas")
    status = Column(String, default="pending")
    student = relationship("User")

class CourseMaterial(Base):
    __tablename__ = "course_materials"

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"))
    title = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    filepath = Column(String, nullable=False)
    uploaded_by = Column(Integer, ForeignKey("users.id"))

    course = relationship("Course", back_populates="materials")
    processed_materials = relationship("ProcessedMaterial", back_populates="material")
    text_chunks = relationship("TextChunk", back_populates="material")

class ProcessedMaterial(Base):
    __tablename__ = "processed_materials"
    
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey('courses.id'), nullable=False)
    material_id = Column(Integer, ForeignKey('course_materials.id'), nullable=False)
    
    course = relationship("Course", back_populates="processed_materials")
    material = relationship("CourseMaterial", back_populates="processed_materials")


class TextChunk(Base):
    __tablename__ = 'text_chunks'

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey('courses.id'))  # Link to the course
    chunk_text = Column(String, nullable=False)
    material_id = Column(Integer, ForeignKey('course_materials.id'))  # Link to the material
    embedding = Column(String, nullable=True)  # Store the embeddings as a string or JSON
    
    # Establish relationships with Course and CourseMaterial tables
    course = relationship("Course", back_populates="text_chunks")
    material = relationship("CourseMaterial", back_populates="text_chunks")

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    course_id = Column(Integer, ForeignKey("courses.id"))
    sender = Column(Text)  # 'student' or 'ai'
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)


class StudyGuide(Base):
    __tablename__ = "study_guides"
    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"))
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    course = relationship("Course", backref="study_guides")


assignment_materials = Table(
    "assignment_materials",
    Base.metadata,
    Column("assignment_id", ForeignKey("assignments.id"), primary_key=True),
    Column("material_id", ForeignKey("course_materials.id"), primary_key=True),
)

class Assignment(Base):
    __tablename__ = "assignments"

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"))
    teacher_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, nullable=False)
    description = Column(Text)
    deadline = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    course = relationship("Course", back_populates="assignments")
    submissions = relationship("AssignmentSubmission", back_populates="assignment", cascade="all, delete-orphan")
    teacher = relationship("User")
    materials = relationship("CourseMaterial", secondary=assignment_materials, backref="assignments")
    rubric_criteria = relationship("RubricCriterion", back_populates="assignment", cascade="all, delete-orphan")


class AssignmentSubmission(Base):
    __tablename__ = "assignment_submissions"
    id = Column(Integer, primary_key=True)
    assignment_id = Column(Integer, ForeignKey("assignments.id"))
    student_id = Column(Integer, ForeignKey("users.id"))
    file_path = Column(String, nullable=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    ai_score = Column(Integer, nullable=True)
    ai_feedback = Column(Text, nullable=True)  # New column for AI feedback
    teacher_score = Column(Integer, nullable=True)
    # Relationships
    student = relationship("User")
    assignment = relationship("Assignment", back_populates="submissions")
    comments = relationship("AssignmentComment", back_populates="submission", cascade="all, delete-orphan")
    rubric_evaluations = relationship("RubricEvaluation", back_populates="submission", cascade="all, delete-orphan")


class AssignmentComment(Base):
    __tablename__ = "assignment_comments"

    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey("assignment_submissions.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User")
    submission = relationship("AssignmentSubmission", back_populates="comments")


class Quiz(Base):
    __tablename__ = "quizzes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(Integer, nullable=False, index=True)
    teacher_id = Column(String, nullable=False, index=True)
    topic = Column(String, nullable=False)
    json_data = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class RubricCriterion(Base):
    __tablename__ = "rubric_criteria"
    
    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(Integer, ForeignKey("assignments.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    weight = Column(Integer, default=10)  # Percentage weight in the overall grade
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    assignment = relationship("Assignment", back_populates="rubric_criteria")
    levels = relationship("RubricLevel", back_populates="criterion", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<RubricCriterion {self.name}>"

class RubricLevel(Base):
    __tablename__ = "rubric_levels"
    
    id = Column(Integer, primary_key=True, index=True)
    criterion_id = Column(Integer, ForeignKey("rubric_criteria.id", ondelete="CASCADE"), nullable=False)
    description = Column(String, nullable=False)
    points = Column(Float, nullable=False)  # Points awarded for this level
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    criterion = relationship("RubricCriterion", back_populates="levels")
    
    def __repr__(self):
        return f"<RubricLevel {self.description[:20]}... - {self.points} pts>"
    


class RubricEvaluation(Base):
    __tablename__ = "rubric_evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("assignment_submissions.id", ondelete="CASCADE"), nullable=False)
    criterion_id = Column(Integer, ForeignKey("rubric_criteria.id", ondelete="CASCADE"), nullable=False)
    level_id = Column(Integer, ForeignKey("rubric_levels.id", ondelete="CASCADE"), nullable=True)
    points_awarded = Column(Float, nullable=False)
    feedback = Column(Text, nullable=True)
    graded_by_ai = Column(Boolean, default=False)
    graded_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    submission = relationship("AssignmentSubmission", back_populates="rubric_evaluations")
    criterion = relationship("RubricCriterion")
    level = relationship("RubricLevel")
    graded_by_user = relationship("User")
    
    def __repr__(self):
        return f"<RubricEvaluation for criterion {self.criterion_id} - {self.points_awarded} points>"