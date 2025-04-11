# backend/ai/ai_routes.py
from fastapi import APIRouter, UploadFile, File, Form, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import shutil
import os
from datetime import datetime
from backend.db.database import engine,get_db
from backend.db.models import Course,CourseMaterial  # Make sure this is correct
from backend.utils.permissions import require_teacher_or_ta  # Optional if you want TA access too
from fastapi.templating import Jinja2Templates

ai_router = APIRouter()

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)



templates = Jinja2Templates(directory="backend/templates")

@ai_router.post("/courses/{course_id}/upload_materials", response_class=HTMLResponse)
async def upload_course_material(
    course_id: int,
    file: UploadFile = File(...),
    title: str = Form(...),
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())  # ✅ Only teachers/TAs
):
    course = db.query(Course).filter_by(id=course_id).first()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)

    filename = f"{datetime.utcnow().timestamp()}_{file.filename}"
    save_path = f"backend/uploads/{filename}"

    with open(save_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    material = CourseMaterial(
        course_id=course_id,
        title=file.filename,
        filename=filename
    )
    db.add(material)
    db.commit()

    return HTMLResponse(
        content="<div class='toast success'>✅ File uploaded successfully!</div>",
        status_code=200
    )

@ai_router.get("/courses/{course_id}/upload_materials", response_class=HTMLResponse)
async def show_upload_form(request: Request, course_id: int):
    return templates.TemplateResponse("upload_materials.html", {
        "request": request,
        "course_id": course_id
    })
