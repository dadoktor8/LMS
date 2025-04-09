from sqlalchemy.orm import Session
from db.database import get_db
from db.models import User, Course, Enrollment

def list_student_enrollments():

    db: Session = next(get_db())  # Get the database session

    enrollments = db.query(Enrollment).all()

    print("📋 Student Enrollments:")
    for e in enrollments:
        student = db.query(User).filter_by(id=e.student_id).first()
        course = db.query(Course).filter_by(id=e.course_id).first()
        print(f"👤 {student.f_name} ({student.email}) ➡️  📘 {course.id} | ✅ Accepted: {e.is_accepted}")

    print("📋 Student Enrollments:")
    enrollments = db.query(Enrollment).all()

    if not enrollments:
        print("No enrollments found.")
        return

    for enrollment in enrollments:
        student = db.query(User).filter(User.id == enrollment.student_id).first()
        course = db.query(Course).filter(Course.id == enrollment.course_id).first()
        if student and course:
            print(f"👤 {student.f_name} {student.l_name} ({student.email}) ➡️  📘 {course.title}")
        else:
            print("⚠️ Invalid enrollment record found.")

if __name__ == "__main__":
    list_student_enrollments()
