# utils/permissions.py (recommended location)
from fastapi import HTTPException, Depends, Request
from sqlalchemy.orm import Session
from backend.db.database import get_db
from backend.db.models import Course, TeachingAssistant
from backend.utils.tokens import decode_token

def require_teacher_or_ta():
    def checker(request: Request, db: Session = Depends(get_db)):
        token = request.cookies.get("access_token")
        if not token:
            raise HTTPException(status_code=401, detail="Missing token")

        payload = decode_token(token)
        user_id = payload.get("user_id")
        role = payload.get("role")
        course_id = request.path_params.get("course_id")

        if not course_id:
            raise HTTPException(status_code=400, detail="Missing course ID")

        # Check teacher
        course = db.query(Course).filter_by(id=course_id).first()
        if course and course.teacher_id == user_id:
            return payload

        # Check if TA
        ta = db.query(TeachingAssistant).filter_by(course_id=course_id, student_id=user_id).first()
        if ta:
            return payload

        raise HTTPException(status_code=403, detail="You are not allowed to access this course")
    return checker
