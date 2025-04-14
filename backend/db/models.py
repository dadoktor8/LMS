# backend/db/models.py
from datetime import datetime
import traceback
from sqlalchemy import Column, DateTime, Integer, String, Boolean, ForeignKey, Text
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