# backend/db/models.py
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from db.database import Base
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


class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(String)
    teacher_id = Column(Integer, ForeignKey("users.id"))

    teacher = relationship("User", back_populates="courses")
    enrollments = relationship("Enrollment", back_populates="course")

class Enrollment(Base):
    __tablename__ = "Enrollment"
    id = Column(Integer,primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"))
    student_id = Column(Integer, ForeignKey("users.id"))

    course = relationship("Course", back_populates="enrollments")
    student = relationship("User")
