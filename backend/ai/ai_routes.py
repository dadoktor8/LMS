# backend/ai/ai_routes.py
import logging
from fastapi import APIRouter, UploadFile, File, Form, Depends, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import shutil
import os
from datetime import datetime
from backend.db.database import engine,get_db
from backend.db.models import Course,CourseMaterial, ProcessedMaterial  # Make sure this is correct
from backend.utils.permissions import require_teacher_or_ta  # Optional if you want TA access too
from fastapi.templating import Jinja2Templates
from .text_processing import extract_text_from_pdf, chunk_text, embed_chunks, save_embeddings_to_faiss,sanitize_filename

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


def process_materials_in_background(course_id: int, db: Session):
    """
    Process the materials for the given course in the background.
    Extracts text, chunks, generates embeddings, and saves them to FAISS.
    """
    # Step 1: Get all materials for the course
    materials = db.query(CourseMaterial).filter_by(course_id=course_id).all()

    for material in materials:
        # Construct the file path to the material
        logging.info(f"File path at => {material.filepath}")
        print(f"File name currently being processed is: {material.filename}")
        print(f"File path is: {material.filepath}")
        file_path = f"backend/uploads/{material.filename}"

        # Check if this material has already been processed
        existing_material = db.query(ProcessedMaterial).filter_by(course_id=course_id, material_id=material.id).first()
        if existing_material:
            logging.info(f"Skipping already processed material: {material.filename}")
            continue  # Skip processing if already done
        
        # Step 2: Extract text from the PDF file
        try:
            text = extract_text_from_pdf(file_path)
        except Exception as e:
            logging.error(f"Error extracting text from {file_path}: {e}")
            continue  # Skip this file if an error occurs
        
        # Step 3: Chunk the extracted text into manageable parts for embedding
        chunks = chunk_text(text)
        
        # Step 4: Generate embeddings for the chunks
        embeddings = embed_chunks(chunks)
        
        # Step 5: Save the embeddings and associated chunks to FAISS
        try:
            save_embeddings_to_faiss(course_id, embeddings, chunks)
        except Exception as e:
            logging.error(f"Error saving embeddings for course {course_id}: {e}")
            continue  # Skip this course if an error occurs
    
    logging.info(f"✅ Materials processed for course {course_id}")