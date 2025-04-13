# backend/ai/ai_routes.py
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import shutil
import os
from datetime import datetime
from backend.auth.routes import require_role
from backend.db.database import engine,get_db
from backend.db.models import Course,CourseMaterial, ProcessedMaterial  # Make sure this is correct
from backend.db.schemas import QueryRequest
from backend.utils.permissions import require_teacher_or_ta  # Optional if you want TA access too
from fastapi.templating import Jinja2Templates
from .text_processing import extract_text_from_pdf, chunk_text, embed_chunks, get_answer_from_rag, process_materials_in_background, save_embeddings_to_faiss,sanitize_filename

ai_router = APIRouter()

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)



templates = Jinja2Templates(directory="backend/templates")

@ai_router.post("/courses/{course_id}/upload_materials", response_class=HTMLResponse)
async def upload_course_material(
    course_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())  # ✅ Only teachers/TAs
):
    course = db.query(Course).filter_by(id=course_id).first()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)

    filename = f"{datetime.utcnow().timestamp()}_{sanitize_filename(file.filename)}"
    save_path = f"backend/uploads/{filename}"
    file_location = f"uploads/{course_id}_{file.filename}"
    os.makedirs("backend/uploads", exist_ok=True)
    with open(save_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    material = CourseMaterial(
        course_id=course_id,
        title=file.filename,
        filename=filename,
        filepath=file_location,
        uploaded_by=user["user_id"]
    )
    db.add(material)
    db.commit()

    return HTMLResponse(
        content="<div class='toast success'>✅ File uploaded successfully!</div>",
        status_code=200
    )

@ai_router.get("/courses/{course_id}/upload_materials", response_class=HTMLResponse)
async def show_upload_form(request: Request, course_id: int, user : dict = Depends(require_teacher_or_ta()), db : Session = Depends(get_db)):
    materials = db.query(CourseMaterial).filter(CourseMaterial.course_id == course_id).order_by(CourseMaterial.uploaded_at.desc()).all()
    return templates.TemplateResponse("upload_materials.html", {
        "request": request,
        "course_id": course_id,
        "role": user["role"],
        "materials":materials
    })

@ai_router.post("/courses/{course_id}/process_materials")
async def process_materials(course_id: int, background_tasks: BackgroundTasks , db: Session = Depends(get_db)):
    background_tasks.add_task(process_materials_in_background, course_id, db)
    
    return HTMLResponse(
        content="<div class='toast success'>✅ File processed successfully!</div>",
        status_code=200
    )


from fastapi import Form

@ai_router.post("/ask_tutor", response_class=HTMLResponse)
async def ask_tutor(
    query: str = Form(...),
    course_id: int = Form(...),
    db: Session = Depends(get_db)
):
    try:
        # Fetch all processed materials for the given course
        course_materials = db.query(CourseMaterial).filter_by(course_id=course_id).all()
        if not course_materials:
            raise HTTPException(status_code=404, detail="No course materials found")

        processed_materials = db.query(ProcessedMaterial).filter_by(course_id=course_id).all()
        if not processed_materials:
            raise HTTPException(status_code=404, detail="Course materials haven't been processed yet")

        # FAISS path
        faiss_index_path = f"faiss_index_{course_id}.index"
        if not os.path.exists(faiss_index_path):
            raise HTTPException(status_code=404, detail="FAISS index not found for this course")

        # Get answer
        answer = get_answer_from_rag(query, faiss_index_path=faiss_index_path, top_k=5)

        return HTMLResponse(content=f"""
            <div class="chat-bubble student">🧑‍🎓 {query}</div>
            <div class="chat-bubble ai">💡 {answer}</div>
            """)

    except Exception as e:
        logging.error(f"Error: {e}")
        return HTMLResponse(content=f"<div class='toast error'>❌ {str(e)}</div>", status_code=500)


@ai_router.get("/courses/{course_id}/tutor", response_class=HTMLResponse)
async def show_student_tutor(request: Request, course_id: int, db: Session = Depends(get_db), user=Depends(require_role("student"))):
    course = db.query(Course).filter(Course.id == course_id).first()
    materials = db.query(CourseMaterial).filter_by(course_id=course_id).order_by(CourseMaterial.uploaded_at.desc()).all()

    return templates.TemplateResponse("student_ai_tutor.html", {
        "request": request,
        "course": course,
        "materials": materials
    })