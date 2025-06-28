# backend/ai/ai_routes.py
import asyncio
import html
import json
import logging
import tempfile
import traceback
from typing import List, Optional
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Depends, Request, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import Session
import shutil
import os
from datetime import date, datetime, time, timedelta

import urllib

from urllib3 import request
from backend.ai.ai_grader import evaluate_assignment_text, prepare_rubric_for_ai
from backend.ai.aws_ai import S3_BUCKET_NAME, delete_file_from_s3, generate_presigned_url, generate_s3_download_link, get_s3_client, upload_file_to_s3
from backend.ai.open_notes_system import check_quiz_quota, generate_quiz_export, generate_study_material, generate_study_material_quiz, increment_quiz_quota, render_flashcards_htmx, render_quiz_htmx, render_study_guide_htmx, retrieve_course_context
from backend.auth.routes import require_role
from backend.db.database import engine,get_db
from backend.db.models import ActivityGroup, ActivityParticipation, Assignment, AssignmentComment, AssignmentSubmission, ChatHistory, Course,CourseMaterial, CourseModule, CourseSubmodule, CourseUploadQuota, Enrollment, FlashcardUsage, InClassActivity, PDFQuotaUsage, ProcessedMaterial,Quiz, QuizAttempt, QuizComment, RubricCriterion, RubricEvaluation, RubricLevel, StudentActivity, StudyGuideUsage, TextChunk, User  # Make sure this is correct
from backend.db.schemas import QueryRequest
from backend.utils.permissions import require_teacher_or_ta  # Optional if you want TA access too
from fastapi.templating import Jinja2Templates
from .text_processing import PDFQuotaConfig, create_modules_from_pdf_analysis, download_file_from_s3, extract_pdf_page_ranges, extract_pdf_pages_to_file, extract_text_from_pdf, get_answer_from_rag_langchain_openai, get_context_for_query, get_course_retriever, get_openai_client, process_materials_in_background, process_submodule_with_quota_check,sanitize_filename, validate_pdf_for_upload
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from langchain.schema import AIMessage, HumanMessage
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import joinedload

ai_router = APIRouter()

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


UPLOAD_DIR = "backend/uploads/assignments"
os.makedirs(UPLOAD_DIR, exist_ok=True)


SECRET_KEY = os.getenv("SECRET_KEY")

templates = Jinja2Templates(directory="backend/templates")


class AIResponse(BaseModel):
    summary: str
    resources: List[str]

class MuddiestPointResponse(AIResponse):
    confusion_areas: List[str]
    review_topics: List[str]

class BeliefEvaluation(BaseModel):
    statement: str
    is_accurate: bool
    explanation: str

class MisconceptionCheckResponse(AIResponse):
    beliefs: List[BeliefEvaluation]





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

    # Define daily limits
    DAILY_FILE_LIMIT = 100  # Max 5 files per day
    DAILY_SIZE_LIMIT = 2 * 1024 * 1024 * 1024  # 100 MB per day
    
    # Get today's upload quota usage
    today = date.today()
    upload_quota = db.query(CourseUploadQuota).filter(
        CourseUploadQuota.course_id == course_id,
        CourseUploadQuota.usage_date == today
    ).first()
    
    # Create new upload quota record if it doesn't exist
    if not upload_quota:
        upload_quota = CourseUploadQuota(
            course_id=course_id,
            usage_date=today,
            files_uploaded=0,
            bytes_uploaded=0
        )
        db.add(upload_quota)
        db.commit()
    
    # Check file count limit
    if upload_quota.files_uploaded >= DAILY_FILE_LIMIT:
        return HTMLResponse(
            content=f"<div class='toast error'>❌ Daily upload limit reached (maximum {DAILY_FILE_LIMIT} files per day)</div>",
            status_code=200
        )
    
    # Read the file content to check size
    contents = await file.read()
    file_size = len(contents)
    
    # Reset file position after reading
    await file.seek(0)
    
    # Check file size limit
    if upload_quota.bytes_uploaded + file_size > DAILY_SIZE_LIMIT:
        remaining_bytes = DAILY_SIZE_LIMIT - upload_quota.bytes_uploaded
        readable_remaining = f"{remaining_bytes / (1024 * 1024):.2f} MB"
        return HTMLResponse(
            content=f"<div class='toast error'>❌ Daily size limit exceeded. Remaining: {readable_remaining}</div>",
            status_code=200
        )
    
    # For PDF files, check page count against quota
    page_count = None
    if file.filename.lower().endswith(".pdf"):
        # Save to temporary file for validation
        temp_dir = "temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = f"{temp_dir}/temp_{file.filename}"
        
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(contents)
            
        # Validate PDF page count
        is_valid, message, page_count = validate_pdf_for_upload(temp_file_path)
        
        # Remove temporary file
        try:
            os.remove(temp_file_path)
        except Exception as e:
            pass  # Continue even if cleanup fails
        
        if not is_valid:
            return HTMLResponse(
                content=f"<div class='toast error'>❌ {message}</div>",
                status_code=200
            )
    
    # Proceed with file upload
    filename = f"{datetime.utcnow().timestamp()}_{sanitize_filename(file.filename)}"
    s3_key = f"course_materials/{course_id}/{filename}"
    
    # Upload file to S3
    upload_success = await upload_file_to_s3(file, s3_key)
    if not upload_success:
        return HTMLResponse("❌ File upload failed", status_code=500)
    
    # Generate a presigned URL that expires in 1 hour
    presigned_url = generate_presigned_url(s3_key)
    
    # Record the material with file size
    material = CourseMaterial(
        course_id=course_id,
        title=file.filename,
        filename=filename,
        filepath=f"uploads/{filename}",
        uploaded_by=user["user_id"],
        file_size=file_size  # Store the file size
    )
    db.add(material)
    
    # Update upload quota usage
    upload_quota.files_uploaded += 1
    upload_quota.bytes_uploaded += file_size
    db.commit()
    
    # Calculate remaining quota
    files_remaining = DAILY_FILE_LIMIT - upload_quota.files_uploaded
    bytes_remaining = DAILY_SIZE_LIMIT - upload_quota.bytes_uploaded
    mb_remaining = bytes_remaining / (1024 * 1024)
    
    # Include page count information in success message if PDF
    pdf_info = f" ({page_count} pages)" if page_count else ""
    
    return HTMLResponse(
        content=f"""
        <div class='toast success'>
            ✅ File uploaded successfully!{pdf_info}
            <div class="text-xs mt-1 text-gray-600">
                Daily quota remaining: {files_remaining} files / {mb_remaining:.2f} MB
            </div>
        </div>
        <script>
            // Refresh the page after successful upload to show the new material in the table
            setTimeout(() => {{
                window.location.reload();
            }}, 2000);
        </script>
        """,
        status_code=200
    )

@ai_router.post("/courses/{course_id}/materials/{material_id}/create_modules", response_class=HTMLResponse)
async def create_modules_from_pdf(
    course_id: int,
    material_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    """Analyze PDF and create suggested module structure"""
    
    material = db.query(CourseMaterial).filter_by(id=material_id, course_id=course_id).first()
    if not material:
        return HTMLResponse("❌ Material not found", status_code=404)
    
    # Check if modules already exist for this material
    existing_modules = db.query(CourseSubmodule).filter_by(material_id=material_id).count()
    if existing_modules > 0:
        return HTMLResponse(
            content=f"<div class='toast error'>❌ Modules already exist for this material</div>",
            status_code=200
        )
    
    try:
        # Get file path (handle S3 download if needed)
        file_path = material.filepath
        if not os.path.exists(file_path):
            s3_key = f"course_materials/{course_id}/{material.filename}"
            temp_dir = "temp_pdfs"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = f"{temp_dir}/{material.filename}"
            
            if not download_file_from_s3(s3_key, file_path):
                return HTMLResponse("❌ Failed to download file", status_code=500)
        
        # Analyze PDF structure
        suggested_modules = extract_pdf_page_ranges(file_path)
        
        if not suggested_modules:
            return HTMLResponse("❌ Could not analyze PDF structure", status_code=500)
        
        # Create modules
        created_count = create_modules_from_pdf_analysis(course_id, material_id, suggested_modules, db)
        
        return HTMLResponse(
            content=f"""
            <div class='toast success'>
                ✅ Created {created_count} modules from PDF analysis!
                <div class="text-xs mt-1 text-gray-600">
                    Refresh the page to see the new modules
                </div>
            </div>
            """,
            status_code=200
        )
        
    except Exception as e:
        logging.error(f"Error creating modules: {e}")
        return HTMLResponse("❌ Error creating modules", status_code=500)

@ai_router.get("/courses/{course_id}/modules", response_class=HTMLResponse)
async def show_course_modules(
    request: Request,
    course_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    """Display course modules and submodules with processing options"""
    
    course = db.query(Course).filter_by(id=course_id).first()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)
    
    # Get all modules with their submodules
    modules = db.query(CourseModule).filter_by(course_id=course_id).order_by(CourseModule.order_index).all()
    quizzes = db.query(Quiz).filter_by(course_id=course_id).order_by(Quiz.created_at.desc()).all()
    quizzes_by_module_id = {}
    for quiz in quizzes:
        quizzes_by_module_id.setdefault(quiz.module_id, []).append(quiz)
    # Get quota information
    today = date.today()
    quota_usage = db.query(PDFQuotaUsage).filter(
        PDFQuotaUsage.course_id == course_id,
        PDFQuotaUsage.usage_date == today
    ).first()
    
    quota_used = quota_usage.pages_processed if quota_usage else 0
    quota_remaining = PDFQuotaConfig.DAILY_PAGE_QUOTA - quota_used
    materials = db.query(CourseMaterial).filter(CourseMaterial.course_id == course_id).all()
    # Get materials that don't have modules yet
    materials_without_modules = db.query(CourseMaterial).filter(
        CourseMaterial.course_id == course_id,
        ~CourseMaterial.id.in_(
            db.query(CourseSubmodule.material_id).filter(CourseSubmodule.material_id.isnot(None))
        )
    ).all()
    
    return templates.TemplateResponse("course_modules.html", {
        "request": request,
        "course": course,
        "modules": modules,
        "materials_without_modules": materials_without_modules,
        "quizzes_by_module_id": quizzes_by_module_id,
        "quota_used": quota_used,
        "quota_remaining": quota_remaining,
        "quota_total": PDFQuotaConfig.DAILY_PAGE_QUOTA,
        "materials": materials,
        "user": user
    })

@ai_router.post("/courses/{course_id}/submodules/{submodule_id}/process", response_class=HTMLResponse)
async def process_submodule(
    course_id: int,
    submodule_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    """Process a specific submodule"""
    
    success, message, pages_used = process_submodule_with_quota_check(course_id, submodule_id, db)
    
    if success:
        return HTMLResponse(
            content=f"""
            <div class='toast success'>
                ✅ {message}
                <div class="text-xs mt-1 text-gray-600">
                    Processed {pages_used} pages
                </div>
            </div>
            """,
            status_code=200
        )
    else:
        return HTMLResponse(
            content=f"<div class='toast error'>❌ {message}</div>",
            status_code=200
        )

@ai_router.post("/courses/{course_id}/modules/{module_id}/process_all", response_class=HTMLResponse)
async def process_all_submodules_in_module(
    course_id: int,
    module_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    """Process all unprocessed submodules in a module"""
    
    module = db.query(CourseModule).filter_by(id=module_id, course_id=course_id).first()
    if not module:
        return HTMLResponse("❌ Module not found", status_code=404)
    
    unprocessed_submodules = db.query(CourseSubmodule).filter_by(
        module_id=module_id, 
        is_processed=False
    ).all()
    
    if not unprocessed_submodules:
        return HTMLResponse(
            content="<div class='toast info'>ℹ️ All submodules in this module are already processed</div>",
            status_code=200
        )
    
    processed_count = 0
    failed_count = 0
    total_pages = 0
    
    for submodule in unprocessed_submodules:
        success, message, pages_used = process_submodule_with_quota_check(course_id, submodule.id, db)
        if success:
            processed_count += 1
            total_pages += pages_used
        else:
            failed_count += 1
            if "quota" in message.lower():
                break  # Stop processing if quota exceeded
    
    return HTMLResponse(
        content=f"""
        <div class='toast success'>
            ✅ Module processing complete!
            <div class="text-xs mt-1 text-gray-600">
                Processed: {processed_count} submodules ({total_pages} pages)
                {f" | Failed: {failed_count}" if failed_count > 0 else ""}
            </div>
        </div>
        """,
        status_code=200
    )

@ai_router.post("/courses/{course_id}/modules/create", response_class=HTMLResponse)
async def create_custom_module(
    course_id: int,
    title: str = Form(...),
    description: str = Form(""),
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    """Create a custom module manually"""
    
    try:
        # Get the next order index
        max_order = db.query(CourseModule.order_index).filter_by(course_id=course_id).order_by(CourseModule.order_index.desc()).first()
        next_order = (max_order[0] + 1) if max_order else 0
        
        module = CourseModule(
            course_id=course_id,
            title=title,
            description=description,
            order_index=next_order
        )
        
        db.add(module)
        db.commit()
        logging.info(f"CREATED MODULE: {module.id} {module.title}")
        print(f"CREATED MODULE: {module.id} {module.title}")
        return HTMLResponse(
            content=f"""
            <div class='toast success'>
                ✅ Module "{title}" created successfully!
            </div>
            """,
            status_code=200
        )
    except Exception as e:
        db.rollback()
        return HTMLResponse(
            content=f"""
            <div class='toast error'>
                ❌ Error creating module: {str(e)}
            </div>
            """,
            status_code=500
        )

@ai_router.get("/student/courses/{course_id}/modules", response_class=HTMLResponse)
async def show_student_modules(
    request: Request,
    course_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    """Display published course modules for students"""
    
    course = db.query(Course).filter_by(id=course_id).first()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)
    
    # Get only published modules with their published submodules
    modules = db.query(CourseModule).filter_by(
        course_id=course_id, 
        is_published=True
    ).order_by(CourseModule.order_index).all()
    published_quizzes = db.query(Quiz)\
        .filter_by(course_id=course_id, is_published=True)\
        .order_by(Quiz.created_at.desc())\
        .all()

    # Group quizzes by module for students
    quizzes_by_module_id = {}
    for quiz in published_quizzes:
        module_id = quiz.module_id or 0
        if module_id not in quizzes_by_module_id:
            quizzes_by_module_id[module_id] = []
        
        # Parse the JSON data and add question count
        try:
            quiz_data = json.loads(quiz.json_data)
            quiz.question_count = len(quiz_data.get('questions', []))
        except:
            quiz.question_count = 0
        
        quizzes_by_module_id[module_id].append(quiz)
    # Filter to only show published submodules and quizzes
    published_modules = []
    for module in modules:
        published_submodules = [sub for sub in module.submodules if sub.is_published]
        published_quizzes = [quiz for quiz in module.quizzes if quiz.is_published]
        
        if published_submodules or published_quizzes:
            module.published_submodules = published_submodules
            module.published_quizzes = published_quizzes
            published_modules.append(module)
    
    return templates.TemplateResponse("student_modules.html", {
        "request": request,
        "course": course,
        "published_modules": published_modules,
        "quizzes_by_module_id": quizzes_by_module_id,
        "user": user
    })


# Add this endpoint to allow students to download published quizzes
@ai_router.get("/courses/{course_id}/quiz/{quiz_id}/download")
async def download_quiz_for_student(
    course_id: int,
    quiz_id: int,
    format: str = Query("pdf", regex="^(pdf|docx)$"),
    include_answers: bool = Query(False),  # Students get version without answers
    user: dict = Depends(require_role("student")),  # Any authenticated user
    db: Session = Depends(get_db)
):
    # Get the specific published quiz
    quiz = db.query(Quiz)\
        .filter_by(id=quiz_id, course_id=course_id, is_published=True)\
        .first()
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found or not published")
    
    # enrollment_check = check_user_enrollment(user.get("user_id"), course_id, db)
    # if not enrollment_check:
    #     raise HTTPException(status_code=403, detail="Not enrolled in this course")
    
    try:
        # Generate the export (students get version without answers)
        s3_key, filename = generate_quiz_export(
            quiz.json_data, 
            course_id, 
            format, 
            include_answers=False  # Force no answers for students
        )
        
        # Return download response
        if s3_key:
            download_url = generate_s3_download_link(s3_key, filename)
            return RedirectResponse(url=download_url)
        else:
            return FileResponse(
                path=filename,
                filename=f"{quiz.topic}_{format}.{format}",
                media_type='application/octet-stream'
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating download: {str(e)}")
# Publish/Unpublish endpoints for teachers
@ai_router.post("/courses/{course_id}/modules/{module_id}/publish")
async def publish_module(
    course_id: int,
    module_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    """Publish a module and all its processed submodules"""
    
    module = db.query(CourseModule).filter_by(id=module_id, course_id=course_id).first()
    if not module:
        return {"error": "Module not found"}, 404
    
    module.is_published = True
    
    # Also publish all processed submodules
    for submodule in module.submodules:
        if submodule.is_processed:
            submodule.is_published = True
    
    db.commit()
    return HTMLResponse(f"""
    <button 
      hx-post="/ai/courses/{course_id}/modules/{module_id}/unpublish"
      hx-target="this"
      hx-swap="outerHTML"
      class="status-toggle status-published">
      ✓ Published
    </button>
    """)

@ai_router.post("/courses/{course_id}/modules/{module_id}/unpublish")
async def unpublish_module(
    course_id: int,
    module_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    """Unpublish a module and all its submodules"""
    
    module = db.query(CourseModule).filter_by(id=module_id, course_id=course_id).first()
    if not module:
        return {"error": "Module not found"}, 404
    
    module.is_published = False
    
    # Also unpublish all submodules
    for submodule in module.submodules:
        submodule.is_published = False
    
    db.commit()
    return HTMLResponse(f"""
    <button 
      hx-post="/ai/courses/{course_id}/modules/{module_id}/publish"
      hx-target="this"
      hx-swap="outerHTML"
      class="status-toggle status-draft">
      📝 Draft
    </button>
    """)

@ai_router.post("/courses/{course_id}/submodules/{submodule_id}/publish")
async def publish_submodule(
    course_id: int,
    submodule_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    """Publish a single submodule"""
    
    submodule = db.query(CourseSubmodule).join(CourseModule).filter(
        CourseSubmodule.id == submodule_id,
        CourseModule.course_id == course_id
    ).first()
    
    if not submodule:
        return {"error": "Submodule not found"}, 404
    
    if not submodule.is_processed:
        return {"error": "Cannot publish unprocessed content"}, 400
    
    submodule.is_published = True
    db.commit()
    return HTMLResponse(f"""
    <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
      <div>
        <span class="font-medium text-gray-900">{submodule.title}</span>
        {f'<span class="text-sm text-gray-500 ml-2">(Pages {submodule.page_range})</span>' if submodule.page_range else ''}
        <span class="ml-2 text-green-600 text-sm">✓ Ready</span>
      </div>
      <div class="flex items-center space-x-2">
        <button 
          hx-post="/ai/courses/{course_id}/submodules/{submodule_id}/unpublish"
          hx-target="closest .bg-gray-50"
          hx-swap="outerHTML"
          class="text-xs bg-orange-500 text-white px-2 py-1 rounded">
          Unpublish
        </button>
      </div>
    </div>
    """)

@ai_router.post("/courses/{course_id}/submodules/{submodule_id}/unpublish")
async def unpublish_submodule(
    course_id: int,
    submodule_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    """Unpublish a single submodule"""
    
    submodule = db.query(CourseSubmodule).join(CourseModule).filter(
        CourseSubmodule.id == submodule_id,
        CourseModule.course_id == course_id
    ).first()
    
    if not submodule:
        return {"error": "Submodule not found"}, 404
    
    submodule.is_published = False
    db.commit()
    return HTMLResponse(f"""
    <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
      <div>
        <span class="font-medium text-gray-900">{submodule.title}</span>
        {f'<span class="text-sm text-gray-500 ml-2">(Pages {submodule.page_range})</span>' if submodule.page_range else ''}
        <span class="ml-2 text-green-600 text-sm">✓ Ready</span>
      </div>
      <div class="flex items-center space-x-2">
        <button 
          hx-post="/ai/courses/{course_id}/submodules/{submodule_id}/publish"
          hx-target="closest .bg-gray-50"
          hx-swap="outerHTML"
          class="text-xs bg-blue-500 text-white px-2 py-1 rounded">
          Publish
        </button>
      </div>
    </div>
    """)

@ai_router.post("/courses/{course_id}/quiz/{quiz_id}/publish")
async def publish_quiz(
    course_id: int,
    quiz_id: int,
    user: dict = Depends(require_role("teacher")),
    db: Session = Depends(get_db)
):
    teacher_id = user.get("user_id", "")
    
    quiz = db.query(Quiz)\
        .filter_by(id=quiz_id, course_id=course_id, teacher_id=teacher_id)\
        .first()
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz.is_published = True
    db.commit()
    
    return HTMLResponse(f"""
        <button
          hx-post="/ai/courses/{course_id}/quiz/{quiz_id}/unpublish"
          hx-target="this"
          hx-swap="outerHTML"
          class="status-toggle status-published">
          ✓ Published
        </button>
    """)

@ai_router.post("/courses/{course_id}/quiz/{quiz_id}/unpublish")
async def unpublish_quiz(
    course_id: int,
    quiz_id: int,
    user: dict = Depends(require_role("teacher")),
    db: Session = Depends(get_db)
):
    teacher_id = user.get("user_id", "")
    
    quiz = db.query(Quiz)\
        .filter_by(id=quiz_id, course_id=course_id, teacher_id=teacher_id)\
        .first()
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz.is_published = False
    db.commit()
    
    return HTMLResponse(f"""
        <button
          hx-post="/ai/courses/{course_id}/quiz/{quiz_id}/publish"
          hx-target="this"
          hx-swap="outerHTML"
          class="status-toggle status-draft">
          📝 Draft
        </button>
    """)

@ai_router.post("/courses/{course_id}/modules/{module_id}/submodules/create", response_class=HTMLResponse)
async def create_custom_submodule(
    course_id: int,
    module_id: int,
    title: str = Form(...),
    description: str = Form(""),
    material_id: Optional[int] = Form(None),
    page_range: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    """Create a custom submodule manually"""
    
    try:
        module = db.query(CourseModule).filter_by(id=module_id, course_id=course_id).first()
        if not module:
            return HTMLResponse(
                content="<div class='toast error'>❌ Module not found</div>", 
                status_code=404
            )
        
        # Validate page range if provided
        if page_range and material_id:
            try:
                start_page, end_page = map(int, page_range.split('-'))
                if start_page > end_page or start_page < 1:
                    return HTMLResponse(
                        content="<div class='toast error'>❌ Invalid page range</div>", 
                        status_code=400
                    )
            except ValueError:
                return HTMLResponse(
                    content="<div class='toast error'>❌ Page range must be in format 'start-end'</div>", 
                    status_code=400
                )
        
        # Get the next order index
        max_order = db.query(CourseSubmodule.order_index).filter_by(module_id=module_id).order_by(CourseSubmodule.order_index.desc()).first()
        next_order = (max_order[0] + 1) if max_order else 0
        
        submodule = CourseSubmodule(
            module_id=module_id,
            material_id=material_id,
            title=title,
            description=description,
            page_range=page_range,
            order_index=next_order
        )
        
        db.add(submodule)
        db.commit()
        
        return HTMLResponse(
            content=f"""
            <div class='toast success'>
                ✅ Submodule "{title}" created successfully!
            </div>
            """,
            status_code=200
        )
    except Exception as e:
        db.rollback()
        return HTMLResponse(
            content=f"""
            <div class='toast error'>
                ❌ Error creating submodule: {str(e)}
            </div>
            """,
            status_code=500
        )

@ai_router.get("/courses/{course_id}/submodules/{submodule_id}/download")
async def download_submodule_pages(
    course_id: int,
    submodule_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))  # Students can download published content
):
    """Download specific pages of a published submodule as PDF"""
    
    # Get the submodule with its module and material
    submodule = db.query(CourseSubmodule).join(CourseModule).filter(
        CourseSubmodule.id == submodule_id,
        CourseModule.course_id == course_id,
        CourseSubmodule.is_published == True  # Only published submodules
    ).first()
    
    if not submodule:
        return HTMLResponse("❌ Submodule not found or not published", status_code=404)
    
    if not submodule.page_range:
        return HTMLResponse("❌ No page range specified for this submodule", status_code=400)
    
    if not submodule.material:
        return HTMLResponse("❌ No source material linked to this submodule", status_code=400)
    
    try:
        # Parse page range
        start_page, end_page = map(int, submodule.page_range.split('-'))
        
        # Get the source material file
        material = submodule.material
        file_path = material.filepath
        
        # Handle S3 download if needed
        if not os.path.exists(file_path):
            s3_key = f"course_materials/{course_id}/{material.filename}"
            temp_dir = "temp_pdfs"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = f"{temp_dir}/{material.filename}"
            
            if not download_file_from_s3(s3_key, file_path):
                return HTMLResponse("❌ Failed to access source file", status_code=500)
        
        # Extract specific pages and create new PDF
        output_pdf_path = extract_pdf_pages_to_file(
            file_path, 
            start_page, 
            end_page, 
            submodule.title
        )
        
        if not output_pdf_path or not os.path.exists(output_pdf_path):
            return HTMLResponse("❌ Failed to extract pages", status_code=500)
        
        # Create safe filename
        safe_filename = f"{submodule.title}_pages_{start_page}-{end_page}.pdf"
        safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
        
        # Return the PDF file
        return FileResponse(
            output_pdf_path,
            media_type='application/pdf',
            filename=safe_filename,
            headers={"Content-Disposition": f"attachment; filename={safe_filename}"}
        )
        
    except ValueError:
        return HTMLResponse("❌ Invalid page range format", status_code=400)
    except Exception as e:
        logging.error(f"Error downloading submodule pages: {e}")
        return HTMLResponse("❌ Error generating download", status_code=500)


@ai_router.get("/courses/{course_id}/submodules/{submodule_id}/download/preview")
async def download_submodule_pages_teacher(
    course_id: int,
    submodule_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())  # Teachers can preview any submodule
):
    """Download specific pages of a submodule as PDF (teacher preview)"""
    
    # Similar logic but without the is_published check for teachers
    submodule = db.query(CourseSubmodule).join(CourseModule).filter(
        CourseSubmodule.id == submodule_id,
        CourseModule.course_id == course_id
    ).first()
    
    if not submodule:
        return HTMLResponse("❌ Submodule not found", status_code=404)
    
    if not submodule.page_range:
        return HTMLResponse("❌ No page range specified for this submodule", status_code=400)
    
    if not submodule.material:
        return HTMLResponse("❌ No source material linked to this submodule", status_code=400)
    
    try:
        # Parse page range
        start_page, end_page = map(int, submodule.page_range.split('-'))
        
        # Get the source material file
        material = submodule.material
        file_path = material.filepath
        
        # Handle S3 download if needed
        if not os.path.exists(file_path):
            s3_key = f"course_materials/{course_id}/{material.filename}"
            temp_dir = "temp_pdfs"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = f"{temp_dir}/{material.filename}"
            
            if not download_file_from_s3(s3_key, file_path):
                return HTMLResponse("❌ Failed to access source file", status_code=500)
        
        # Extract specific pages and create new PDF
        output_pdf_path = extract_pdf_pages_to_file(
            file_path, 
            start_page, 
            end_page, 
            f"PREVIEW_{submodule.title}"
        )
        
        if not output_pdf_path or not os.path.exists(output_pdf_path):
            return HTMLResponse("❌ Failed to extract pages", status_code=500)
        
        # Create safe filename
        safe_filename = f"PREVIEW_{submodule.title}_pages_{start_page}-{end_page}.pdf"
        safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
        
        # Return the PDF file
        return FileResponse(
            output_pdf_path,
            media_type='application/pdf',
            filename=safe_filename,
            headers={"Content-Disposition": f"attachment; filename={safe_filename}"}
        )
        
    except ValueError:
        return HTMLResponse("❌ Invalid page range format", status_code=400)
    except Exception as e:
        logging.error(f"Error downloading submodule pages for teacher: {e}")
        return HTMLResponse("❌ Error generating download", status_code=500)
    

@ai_router.post("/courses/{course_id}/modules/delete-all", response_class=HTMLResponse)
async def delete_all_modules_with_post(
    course_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    method_override = request.query_params.get("_method")
    if method_override != "DELETE":
        return HTMLResponse("Invalid method", status_code=400)
   
    # Get all modules for this course
    modules = db.query(CourseModule).filter_by(course_id=course_id).all()
    if not modules:
        return HTMLResponse("❌ No modules found for this course", status_code=404)
   
    # Delete all modules (this will cascade delete submodules if FK constraint is set up properly)
    # If not using CASCADE, you'll need to delete submodules first:
    
    # Option 1: If you have CASCADE DELETE set up in your database
    for module in modules:
        db.delete(module)
    
    # Option 2: If you need to manually delete submodules first (uncomment if needed)
    # for module in modules:
    #     # Delete all submodules for this module first
    #     submodules = db.query(CourseSubmodule).filter_by(module_id=module.id).all()
    #     for submodule in submodules:
    #         db.delete(submodule)
    #     # Then delete the module
    #     db.delete(module)
    
    db.commit()
    
    modules_count = len(modules)
    return HTMLResponse(
        content=f"✅ Successfully deleted {modules_count} module(s)",
        status_code=200
    )

@ai_router.post("/courses/{course_id}/modules/{module_id}", response_class=HTMLResponse)
async def delete_module_with_post(
    course_id: int,
    module_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    method_override = request.query_params.get("_method")
    if method_override != "DELETE":
        return HTMLResponse("Invalid method", status_code=400)
    
    module = db.query(CourseModule).filter_by(id=module_id, course_id=course_id).first()
    if not module:
        return HTMLResponse("❌ Module not found", status_code=404)
    
    # Check if any submodules are processed
    processed_submodules = db.query(CourseSubmodule).filter_by(module_id=module_id, is_processed=True).count()
    if processed_submodules > 0:
        return HTMLResponse("❌ Cannot delete module with processed submodules", status_code=400)
    
    db.delete(module)
    db.commit()

    return HTMLResponse(
        content="",
        status_code=200
    )



@ai_router.get("/courses/{course_id}/upload_materials", response_class=HTMLResponse)
async def show_upload_form(request: Request, course_id: int, user: dict = Depends(require_teacher_or_ta()), db: Session = Depends(get_db)):
    # Get course materials as before
    materials = db.query(CourseMaterial).filter(CourseMaterial.course_id == course_id).order_by(CourseMaterial.uploaded_at.desc()).all()
    teacher_id = user["user_id"]
    courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()
    
    # Constants for upload size limit
    DAILY_UPLOAD_SIZE_LIMIT = 2 * 1024  * 1024 * 1024  # 100 MB in bytes
    
    # Get today's date
    today = date.today()
    
    # Get processing quota information for today
    quota_usage = db.query(PDFQuotaUsage).filter(
        PDFQuotaUsage.course_id == course_id,
        PDFQuotaUsage.usage_date == today
    ).first()
    
    # Set processing quota values - using our centralized config
    quota_used = quota_usage.pages_processed if quota_usage else 0
    quota_remaining = PDFQuotaConfig.DAILY_PAGE_QUOTA - quota_used
    
    # Get upload quota information for today
    upload_quota = db.query(CourseUploadQuota).filter(
        CourseUploadQuota.course_id == course_id,
        CourseUploadQuota.usage_date == today
    ).first()
    
    # Set upload quota values
    upload_files_used = upload_quota.files_uploaded if upload_quota else 0
    upload_bytes_used = upload_quota.bytes_uploaded if upload_quota else 0
    
    # Return template with quota information
    return templates.TemplateResponse("upload_materials.html", {
        "request": request,
        "course_id": course_id,
        "role": user["role"],
        "materials": materials,
        "courses": courses,
        "quota_used": quota_used,
        "quota_remaining": quota_remaining,
        "quota_total": PDFQuotaConfig.DAILY_PAGE_QUOTA,
        "upload_files_used": upload_files_used,
        "upload_bytes_used": upload_bytes_used,
        "upload_bytes_total": DAILY_UPLOAD_SIZE_LIMIT
    })

@ai_router.delete("/courses/{course_id}/materials/{material_id}", response_class=HTMLResponse)
async def delete_course_material(
    course_id: int,
    material_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())  # ✅ Only teachers/TAs
):
    # Verify course exists
    course = db.query(Course).filter_by(id=course_id).first()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)
    
    # Get the material
    material = db.query(CourseMaterial).filter(
        CourseMaterial.id == material_id,
        CourseMaterial.course_id == course_id
    ).first()
    
    if not material:
        return HTMLResponse("❌ Material not found", status_code=404)
    
    # Check if material has already been processed
    processed_material = db.query(ProcessedMaterial).filter(
        ProcessedMaterial.material_id == material_id
    ).first()
    
    if processed_material:
        return HTMLResponse(
            content="<div class='toast error'>❌ Cannot delete processed materials. Material has already been processed.</div>",
            status_code=400
        )
    
    # Delete from S3 (if using S3 storage)
    s3_key = f"course_materials/{course_id}/{material.filename}"
    try:
        # Assuming you have a delete_file_from_s3 function
        delete_success = await delete_file_from_s3(s3_key)
        if not delete_success:
            return HTMLResponse(
                content="<div class='toast error'>❌ Failed to delete file from storage</div>",
                status_code=500
            )
    except Exception as e:
        # Log the error but continue with database deletion
        print(f"Warning: Failed to delete file from S3: {e}")
    
    # Update upload quota (reduce the counts since we're deleting)
    today = date.today()
    upload_quota = db.query(CourseUploadQuota).filter(
        CourseUploadQuota.course_id == course_id,
        CourseUploadQuota.usage_date == today
    ).first()
    
    if upload_quota:
        # Get file size from the stored material record
        file_size = material.file_size if material.file_size else 0
        
        # Reduce quota usage
        upload_quota.files_uploaded = max(0, upload_quota.files_uploaded - 1)
        upload_quota.bytes_uploaded = max(0, upload_quota.bytes_uploaded - file_size)
    
    # Delete any related text chunks (if they exist)
    db.query(TextChunk).filter(TextChunk.material_id == material_id).delete()
    
    # Delete the material record
    db.delete(material)
    db.commit()
    
    return HTMLResponse(
        content="""
        <div class='toast success'>
            ✅ Material deleted successfully!
            <div class="text-xs mt-1 text-gray-600">
                Upload quota has been restored.
            </div>
        </div>
        """,
        status_code=200
    )


@ai_router.get("/courses/{course_id}/materials/refresh", response_class=HTMLResponse)
async def refresh_materials_list(
    course_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    """
    Returns just the materials table body HTML for HTMX updates
    """
    course = db.query(Course).filter_by(id=course_id).first()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)
    
    # Get materials for this course
    materials = db.query(CourseMaterial).filter(
        CourseMaterial.course_id == course_id
    ).order_by(CourseMaterial.uploaded_at.desc()).all()
    
    # Generate table rows HTML
    if not materials:
        table_html = '''
        <tr id="no-materials-row">
            <td colspan="3" class="text-center text-gray-400 py-4">No materials uploaded yet.</td>
        </tr>
        '''
    else:
        table_html = ""
        for material in materials:
            # Check if material is processed
            is_processed = len(material.processed_materials) > 0
            
            if is_processed:
                action_html = '''
                <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    ✓ Processed
                </span>
                '''
            else:
                action_html = f'''
                <button
                    onclick="confirmDelete({material.id}, '{material.title or material.filename}')"
                    class="inline-flex items-center px-3 py-1 text-xs font-medium text-red-600 bg-red-50 border border-red-200 rounded-md hover:bg-red-100 transition-colors"
                    title="Delete material">
                    🗑️ Delete
                </button>
                '''
            
            table_html += f'''
            <tr class="border-b" id="material-row-{material.id}">
                <td class="px-4 py-2">
                    <a href="/ai/courses/{course_id}/materials/{material.id}/download" target="_blank"
                    class="text-blue-700 underline font-medium hover:text-blue-900">
                    📄 {material.title or material.filename}
                    </a>
                </td>
                <td class="px-4 py-2">{material.uploaded_at.strftime('%Y-%m-%d %H:%M')}</td>
                <td class="px-4 py-2">
                    {action_html}
                </td>
            </tr>
            '''
    
    return HTMLResponse(content=table_html, status_code=200)


@ai_router.post("/courses/{course_id}/process_materials")
async def process_materials(course_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # Get today's quota usage
    today = date.today()
    quota_usage = db.query(PDFQuotaUsage).filter(
        PDFQuotaUsage.course_id == course_id,
        PDFQuotaUsage.usage_date == today
    ).first()
    
    used_pages = quota_usage.pages_processed if quota_usage else 0
    remaining_quota = PDFQuotaConfig.DAILY_PAGE_QUOTA - used_pages
    
    # Process materials in foreground instead of background for immediate feedback
    # This change gives instant feedback to the user
    result = process_materials_in_background(course_id, db)
    
    # After processing, get updated quota information
    quota_usage = db.query(PDFQuotaUsage).filter(
        PDFQuotaUsage.course_id == course_id,
        PDFQuotaUsage.usage_date == today
    ).first()
    
    used_pages = quota_usage.pages_processed if quota_usage else 0
    remaining_quota = PDFQuotaConfig.DAILY_PAGE_QUOTA - used_pages
    
    return HTMLResponse(
        content=f"""
        <div class="bg-blue-50 border border-blue-100 rounded-lg p-4 mb-4 mt-4">
            <h3 class="font-semibold text-blue-700">📊 Processing Results</h3>
            <ul class="mt-2 text-sm text-blue-600 space-y-1">
                <li>✅ Processed: {result['processed']} materials</li>
                <li>⏭️ Skipped: {result['skipped']} (already processed)</li>
                <li>⚠️ Quota exceeded: {result['quota_exceeded']} materials</li>
            </ul>
            <p class="mt-2 text-xs text-blue-700">
                Daily quota remaining: {remaining_quota} pages
            </p>
            <script>
                // Trigger a page refresh to update the quota bar
                setTimeout(function() {{
                    window.location.reload();
                }}, 3000);
            </script>
        </div>
        """,
        status_code=200
    )

@ai_router.get("/courses/{course_id}/materials/{material_id}/download")
async def download_material(course_id: int, material_id: int, db: Session = Depends(get_db)):
    # Get the material info from DB
    material = db.query(CourseMaterial).filter_by(id=material_id, course_id=course_id).first()
    if not material:
        return HTMLResponse("❌ Material not found", status_code=404)
    # Reconstruct the S3 key (must match upload!)
    s3_key = f"course_materials/{course_id}/{material.filename}"

    # Get a presigned URL from S3
    url = generate_presigned_url(s3_key, expiration=300)  # 5 min expiry
    if not url:
        return HTMLResponse("❌ Could not generate download link", status_code=500)
    return RedirectResponse(url)

@ai_router.post("/ask_tutor", response_class=HTMLResponse)
async def ask_tutor(
    query: str = Form(...),
    course_id: int = Form(...),
    module_id: str = Form(""),  # Changed to str to handle empty values
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    try:
        student_id = str(user["user_id"])
        session_id = f"{student_id}_{course_id}"
        
        # Convert module_id from string to int or None
        parsed_module_id = None
        if module_id and module_id.strip() and module_id != "":
            try:
                parsed_module_id = int(module_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid module ID format")
        
        print(f"User ID: {user['user_id']}, Course ID: {course_id}, Module ID: {parsed_module_id}")
        current_time = datetime.now()
        print(f"Current server time: {current_time}")
        
        # Check daily message limit PER COURSE (100 messages per day per course)
        today = current_time.date()
        today_start = datetime.combine(today, time.min)
        today_end = datetime.combine(today, time.max)
        print(f"Date range: {today_start} to {today_end}")

        message_count = db.query(ChatHistory).filter(
            ChatHistory.user_id == user["user_id"],
            ChatHistory.course_id == course_id,
            ChatHistory.sender == "student",
            ChatHistory.timestamp >= today_start,
            ChatHistory.timestamp <= today_end
        ).count()
        
        print(f"Current message count: {message_count} for course {course_id}")
        if message_count >= 100:
            return HTMLResponse(
                content="""
                <div class="chat-bubble bg-amber-50 border border-amber-200 text-amber-900 px-6 py-4 rounded-2xl self-start max-w-2xl shadow text-lg font-medium leading-relaxed whitespace-pre-line">
                    ⚠️ You've reached your daily message limit (100 messages) for this course. Please try again tomorrow.
                </div>
                <script>
                    document.body.dispatchEvent(new CustomEvent('ai-response-complete'));
                </script>
                """,
                status_code=429,
                headers={"HX-Trigger": "ai-response-complete"}
            )
        
        # Check if course exists
        course = db.query(Course).filter_by(id=course_id).first()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        
        # Validate module_id if provided (ensure it's published and belongs to the course)
        if parsed_module_id:
            module = db.query(CourseModule).filter_by(
                id=parsed_module_id, 
                course_id=course_id, 
                is_published=True
            ).first()
            if not module:
                raise HTTPException(status_code=400, detail="Invalid or unpublished module selected")
            
        # Save student message to DB
        student_message = ChatHistory(
            user_id=user["user_id"], 
            course_id=course_id, 
            sender="student", 
            message=query,
            timestamp=current_time
        )
        db.add(student_message)
        db.commit()
        
        # Get answer using enhanced RAG system
        answer = get_answer_from_rag_langchain_openai(query, course_id, student_id, parsed_module_id)
        
        # Save AI response to DB
        ai_message = ChatHistory(
            user_id=user["user_id"], 
            course_id=course_id, 
            sender="ai", 
            message=answer,
            timestamp=datetime.now()
        )
        db.add(ai_message)
        db.commit()
        
        safe_query = html.escape(query)
        
        # Save to chat history
        history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string="sqlite:///chat_history.db"
        )
        history.add_user_message(query)
        history.add_ai_message(answer)
        
        return HTMLResponse(
            content=f"""
            <div class="chat-bubble bg-blue-600 text-white px-6 py-4 rounded-2xl self-end max-w-2xl ml-auto shadow text-lg font-semibold whitespace-pre-line">
                🧑‍🎓 {safe_query}
            </div>
            <div class="chat-bubble bg-indigo-50 border border-indigo-200 text-indigo-900 px-6 py-4 rounded-2xl self-start max-w-2xl shadow text-lg font-medium leading-relaxed whitespace-pre-line">
                💡 {answer}
            </div>
            <script>
                document.body.dispatchEvent(new CustomEvent('ai-response-complete'));
            </script>
            """, 
            status_code=200,
            headers={"HX-Trigger": "ai-response-complete"}
        )
       
    except HTTPException as e:
        logging.error(f"HTTP Exception: {e.detail}")
        return HTMLResponse(
            content=f"""
            <div class='toast error'>❌ {e.detail}</div>
            <script>
                document.body.dispatchEvent(new CustomEvent('ai-response-complete'));
            </script>
            """, 
            status_code=e.status_code,
            headers={"HX-Trigger": "ai-response-complete"}
        )
    except Exception as e:
        import traceback
        logging.error(f"Error in ask_tutor: {str(e)}")
        logging.error(traceback.format_exc())
        return HTMLResponse(
            content=f"""
            <div class='toast error'>❌ Something went wrong. Please try again later.</div>
            <script>
                document.body.dispatchEvent(new CustomEvent('ai-response-complete'));
            </script>
            """, 
            status_code=500,
            headers={"HX-Trigger": "ai-response-complete"}
        )

@ai_router.get("/courses/{course_id}/tutor", response_class=HTMLResponse)
async def show_student_tutor(
    request: Request, 
    course_id: int, 
    db: Session = Depends(get_db), 
    user=Depends(require_role("student"))
):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Get only published course modules for selection dropdown
    modules = db.query(CourseModule).filter_by(
        course_id=course_id, 
        is_published=True
    ).order_by(CourseModule.order_index).all()
    
    messages = db.query(ChatHistory).filter_by(user_id=user["user_id"], course_id=course_id).order_by(ChatHistory.timestamp).all()
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
    
    # Get remaining messages for today for THIS COURSE
    today = datetime.now().date()
    today_start = datetime.combine(today, time.min)
    today_end = datetime.combine(today, time.max)
    
    message_count = db.query(ChatHistory).filter(
        ChatHistory.user_id == user["user_id"],
        ChatHistory.course_id == course_id,
        ChatHistory.sender == "student",
        ChatHistory.timestamp >= today_start,
        ChatHistory.timestamp <= today_end
    ).count()
    
    remaining_messages = 100 - message_count if message_count < 100 else 0

    print(f"{remaining_messages} to get!")
    print("COURSE ID:", course_id)
    
    return templates.TemplateResponse("student_ai_tutor.html", {
        "request": request,
        "course": course,
        "course_id": course_id,
        "modules": modules,  # Pass modules instead of materials
        "messages": messages,
        "courses": courses,
        "remaining_messages": remaining_messages
    })

@ai_router.post("/process_materials")
def process_materials(action: str, db: Session = Depends(get_db)):
    return cleanup_or_reset_processed_materials(db, action)

def cleanup_or_reset_processed_materials(db: Session, action: str):
    try:
        if action == "clean":
            # Remove all records
            db.query(ProcessedMaterial).delete()
            db.commit()
            return {"status": "success", "message": "All processed materials have been cleaned."}
        elif action == "reset":
            # Reset status to 'pending'
            db.query(ProcessedMaterial).update({"status": "pending"})
            db.commit()
            return {"status": "success", "message": "Processed material statuses have been reset."}
        else:
            return {"status": "error", "message": "Invalid action. Use 'clean' or 'reset'."}
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}

'''
@ai_router.get("/study", response_class=HTMLResponse)
async def study_page(
    request: Request,
    course_id: int = Query(...),
    topic: Optional[str] = Query(None),
    material_type: Optional[str] = Query(None)
):

    session_material_type = request.session.get("material_type")
    materials_json = request.session.get("current_materials")

    # Render the materials in Python
    if session_material_type == "flashcards" and materials_json:
        study_material_html = render_flashcards_htmx(materials_json)
    elif session_material_type == "quiz" and materials_json:
        study_material_html = render_quiz_htmx(materials_json)
    elif session_material_type == "study_guide" and materials_json:
        study_material_html = render_study_guide_htmx(materials_json)
    else:
        study_material_html = """
            <div class="welcome-container">
                <h3>Enter a topic and select a material type to get started</h3>
                <p>Lumi will create personalized study materials based on your course content.</p>
            </div>
        """
    print("SESSION (in study page):", request.session)
    print("HTML to render:", study_material_html[:500])
    return templates.TemplateResponse(
        "study.html",
        {
            "request": request,
            "course_id": course_id,
            "study_material_html": study_material_html,
        }
    )


@ai_router.post("/study/generate", response_class=HTMLResponse)
async def generate_study_material(
    request: Request,
    topic: str = Form(...),
    material_type: str = Form(...),
    course_id: int = Form(...),
    student_id: str = Form(...)
):
    materials_json = generate_study_materials(topic, material_type, course_id, student_id)
    
    # Store in session for state persistence (add these lines)
    request.session["material_type"] = material_type
    request.session["current_materials"] = materials_json
    
    # Select renderer:
    if material_type == "flashcards":
        study_material_html = render_flashcards_htmx(materials_json)
    elif material_type == "quiz":
        study_material_html = render_quiz_htmx(materials_json)
    elif material_type == "study_guide":
        study_material_html = render_study_guide_htmx(materials_json)
    else:
        study_material_html = "<div>Unknown material type</div>"
    return templates.TemplateResponse(
        "study.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "material_type": material_type,
            "study_material_html": study_material_html,
            "student_id": student_id,
        }
    )


@ai_router.get("/study/flip-card/{card_id}", response_class=HTMLResponse)
async def flip_card(request: Request, card_id: int, show_answer: bool = False):
    """
    Flip a flashcard to show question or answer
    """
    print("DEBUG FLIP:", dict(request.session))
    materials_json = request.session.get("current_materials")
    material_type = request.session.get("material_type")
    print(f"DEBUG FLIP - Card ID: {card_id}, Show Answer: {show_answer}")
    print(f"DEBUG FLIP - Material Type: {material_type}")
    print(f"DEBUG FLIP - Materials JSON exists: {'Yes' if materials_json else 'No'}")
    if not materials_json or material_type != "flashcards":
        return HTMLResponse('<div class="toast error">No flashcards available.</div>')
    return HTMLResponse(render_flip_card_htmx(card_id, materials_json, show_answer))
    
@ai_router.post("/study/check-answer/{question_id}", response_class=HTMLResponse)
async def check_answer(
    request: Request,
    question_id: int,
    answer: int = Form(...),
    correct_answer: int = Form(...),
    explanation: str = Form(...)
):
        """
        Check a quiz answer and return the result
        """
        return render_question_result_htmx(question_id, answer, correct_answer, explanation)
'''

@ai_router.delete("/ai/clear_history/{student_id}/{course_id}")
def clear_chat_history(student_id: str, course_id: int, db: Session = Depends(get_db)):
    session_id = f"{student_id}_{course_id}"
    try:
        # 1. Delete from main app DB (ChatHistory table)
        db.query(ChatHistory)\
            .filter(ChatHistory.user_id == int(student_id), ChatHistory.course_id == int(course_id))\
            .delete()
        db.commit()

        # 2. Delete from message_store in chat_history.db
        engine = create_engine("sqlite:///chat_history.db")
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM message_store WHERE session_id = :sid"),
                {"sid": session_id}
            )
        return {"message": f"🧹 Cleared chat history for session '{session_id}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}")
    

# Main study landing page
@ai_router.get("/study", response_class=HTMLResponse)
async def study_landing_page(
    request: Request,
    course_id: int = Query(...),
    topic: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
    return templates.TemplateResponse(
        "study_landing.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "student_id": request.session.get("student_id", ""),
            "courses":courses
        }
    )

# Flashcards specific page
@ai_router.get("/study/flashcards", response_class=HTMLResponse)
async def flashcards_page(
    request: Request,
    course_id: int = Query(...),
    topic: Optional[str] = Query(None),
    module_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    materials_json = request.session.get("flashcards_materials")
    study_material_html = render_flashcards_htmx(materials_json) if materials_json else ""
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
    
    # Get only published modules for the course
    modules = db.query(CourseModule).filter_by(
        course_id=course_id, 
        is_published=True
    ).order_by(CourseModule.order_index).all()
    
    return templates.TemplateResponse(
        "flashcards.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "module_id": module_id,
            "modules": modules,
            "study_material_html": study_material_html,
            "student_id": request.session.get("student_id", ""),
            "courses": courses
        }
    )

# Quiz specific page
@ai_router.get("/study/quiz", response_class=HTMLResponse)
async def quiz_page(
    request: Request,
    course_id: int = Query(...),
    topic: Optional[str] = Query(None),
):
    materials_json = request.session.get("quiz_materials")
    study_material_html = render_quiz_htmx(materials_json) if materials_json else ""
    
    return templates.TemplateResponse(
        "quiz.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "study_material_html": study_material_html,
            "student_id": request.session.get("student_id", "")
        }
    )

# Study guide specific page
@ai_router.get("/study/guide", response_class=HTMLResponse)
async def study_guide_page(
    request: Request,
    course_id: int = Query(...),
    topic: Optional[str] = Query(None),
    module_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    materials_json = request.session.get("study_guide_materials")
    study_material_html = render_study_guide_htmx(materials_json) if materials_json else ""
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
    
    # Get only published modules for the course
    modules = db.query(CourseModule).filter_by(
        course_id=course_id, 
        is_published=True
    ).order_by(CourseModule.order_index).all()
    
    return templates.TemplateResponse(
        "study_guide.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "module_id": module_id,
            "modules": modules,
            "study_material_html": study_material_html,
            "student_id": request.session.get("student_id", ""),
            "courses": courses
        }
    )

# Separate endpoints for generating each type of material
@ai_router.post("/study/flashcards/generate", response_class=HTMLResponse)
async def generate_flashcards(
    request: Request,
    topic: str = Form(...),
    course_id: int = Form(...),
    student_id: str = Form(...),
    module_id: str = Form(""),  # Changed to str to handle empty values
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    # Convert module_id from string to int or None
    parsed_module_id = None
    if module_id and module_id.strip() and module_id != "":
        try:
            parsed_module_id = int(module_id)
        except ValueError:
            # Return error for invalid module format
            enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
            courses = [enroll.course for enroll in enrollments]
            modules = db.query(CourseModule).filter_by(
                course_id=course_id, 
                is_published=True
            ).order_by(CourseModule.order_index).all()
            
            return templates.TemplateResponse(
                "flashcards.html",
                {
                    "request": request,
                    "course_id": course_id,
                    "topic": topic,
                    "module_id": None,
                    "modules": modules,
                    "study_material_html": "",
                    "student_id": student_id,
                    "courses": courses,
                    "error_message": "Invalid module ID format."
                }
            )
    
    # Validate module_id if provided (ensure it's published and belongs to the course)
    if parsed_module_id:
        module = db.query(CourseModule).filter_by(
            id=parsed_module_id, 
            course_id=course_id, 
            is_published=True
        ).first()
        if not module:
            # Return error for invalid module
            enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
            courses = [enroll.course for enroll in enrollments]
            modules = db.query(CourseModule).filter_by(
                course_id=course_id, 
                is_published=True
            ).order_by(CourseModule.order_index).all()
            
            return templates.TemplateResponse(
                "flashcards.html",
                {
                    "request": request,
                    "course_id": course_id,
                    "topic": topic,
                    "module_id": parsed_module_id,
                    "modules": modules,
                    "study_material_html": "",
                    "student_id": student_id,
                    "courses": courses,
                    "error_message": "Invalid or unpublished module selected."
                }
            )
    
    # Check today's usage for this student and course
    today = date.today()
    usage = db.query(FlashcardUsage).filter(
        FlashcardUsage.student_id == student_id,
        FlashcardUsage.course_id == course_id,
        FlashcardUsage.usage_date == today
    ).first()
    
    # Set the daily limit
    daily_limit = 5
    
    # Check if the user has exceeded their daily quota
    if usage and usage.count >= daily_limit:
        # Return the page with an error message
        enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
        courses = [enroll.course for enroll in enrollments]
        modules = db.query(CourseModule).filter_by(
            course_id=course_id, 
            is_published=True
        ).order_by(CourseModule.order_index).all()
        
        return templates.TemplateResponse(
            "flashcards.html",
            {
                "request": request,
                "course_id": course_id,
                "topic": topic,
                "module_id": parsed_module_id,
                "modules": modules,
                "study_material_html": "",
                "student_id": student_id,
                "courses": courses,
                "error_message": f"Daily limit reached. You can generate up to {daily_limit} flashcards per course per day."
            }
        )
    
    # Update usage record or create a new one
    if usage:
        usage.count += 1
    else:
        new_usage = FlashcardUsage(
            student_id=student_id,
            course_id=course_id,
            usage_date=today,
            count=1
        )
        db.add(new_usage)
    
    # Commit changes to the database
    db.commit()
    
    # Generate the flashcards with optional module focus
    materials_json = generate_study_material(topic, "flashcards", course_id, student_id, parsed_module_id)
    request.session["flashcards_materials"] = materials_json
    study_material_html = render_flashcards_htmx(materials_json)
    
    # Get courses for the sidebar
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
    modules = db.query(CourseModule).filter_by(
        course_id=course_id, 
        is_published=True
    ).order_by(CourseModule.order_index).all()
    
    return templates.TemplateResponse(
        "flashcards.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "module_id": parsed_module_id,
            "modules": modules,
            "study_material_html": study_material_html,
            "student_id": student_id,
            "courses": courses,
        }
    )
@ai_router.post("/study/quiz/generate", response_class=HTMLResponse)
async def generate_quiz(
    request: Request,
    topic: str = Form(...),
    course_id: int = Form(...),
    student_id: str = Form(...)
):
    materials_json = generate_study_material(topic, "quiz", course_id, student_id)
    request.session["quiz_materials"] = materials_json
    study_material_html = render_quiz_htmx(materials_json)
    
    return templates.TemplateResponse(
        "quiz.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "study_material_html": study_material_html,
            "student_id": student_id,
        }
    )

@ai_router.post("/study/guide/generate", response_class=HTMLResponse)
async def generate_study_guide(
    request: Request,
    topic: str = Form(...),
    course_id: int = Form(...),
    student_id: str = Form(...),
    module_id: str = Form(""),  # Changed to str to handle empty values
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    # Convert module_id from string to int or None
    parsed_module_id = None
    if module_id and module_id.strip() and module_id != "":
        try:
            parsed_module_id = int(module_id)
        except ValueError:
            # Return error for invalid module format
            enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
            courses = [enroll.course for enroll in enrollments]
            modules = db.query(CourseModule).filter_by(
                course_id=course_id, 
                is_published=True
            ).order_by(CourseModule.order_index).all()
            
            return templates.TemplateResponse(
                "study_guide.html",
                {
                    "request": request,
                    "course_id": course_id,
                    "topic": topic,
                    "module_id": None,
                    "modules": modules,
                    "study_material_html": "",
                    "student_id": student_id,
                    "courses": courses,
                    "error_message": "Invalid module ID format."
                }
            )
    
    # Validate module_id if provided (ensure it's published and belongs to the course)
    if parsed_module_id:
        module = db.query(CourseModule).filter_by(
            id=parsed_module_id, 
            course_id=course_id, 
            is_published=True
        ).first()
        if not module:
            # Return error for invalid module
            enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
            courses = [enroll.course for enroll in enrollments]
            modules = db.query(CourseModule).filter_by(
                course_id=course_id, 
                is_published=True
            ).order_by(CourseModule.order_index).all()
            
            return templates.TemplateResponse(
                "study_guide.html",
                {
                    "request": request,
                    "course_id": course_id,
                    "topic": topic,
                    "module_id": parsed_module_id,
                    "modules": modules,
                    "study_material_html": "",
                    "student_id": student_id,
                    "courses": courses,
                    "error_message": "Invalid or unpublished module selected."
                }
            )
    
    today = date.today()
    usage = db.query(StudyGuideUsage).filter(
        StudyGuideUsage.student_id == student_id,
        StudyGuideUsage.course_id == course_id,
        StudyGuideUsage.usage_date == today
    ).first()
    
    # Set the daily limit
    daily_limit = 5
    if usage and usage.count >= daily_limit:
        # Return the page with an error message
        enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
        courses = [enroll.course for enroll in enrollments]
        modules = db.query(CourseModule).filter_by(
            course_id=course_id, 
            is_published=True
        ).order_by(CourseModule.order_index).all()
        
        return templates.TemplateResponse(
            "study_guide.html",
            {
                "request": request,
                "course_id": course_id,
                "topic": topic,
                "module_id": parsed_module_id,
                "modules": modules,
                "study_material_html": "",
                "student_id": student_id,
                "courses": courses,
                "error_message": f"Daily limit reached. You can generate up to {daily_limit} Study Guides per course per day."
            }
        )
    
    if usage:
        usage.count += 1
    else:
        new_usage = StudyGuideUsage(
            student_id=student_id,
            course_id=course_id,
            usage_date=today,
            count=1
        )
        db.add(new_usage)
    
    # Commit changes to the database
    db.commit()
    print(f"Study guides remaining {usage and usage.count}")
    
    # Generate study material with optional module focus
    materials_json = generate_study_material(topic, "study_guide", course_id, student_id, parsed_module_id)
    request.session["study_guide_materials"] = materials_json
    study_material_html = render_study_guide_htmx(materials_json)
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
    modules = db.query(CourseModule).filter_by(
        course_id=course_id, 
        is_published=True
    ).order_by(CourseModule.order_index).all()
    
    return templates.TemplateResponse(
        "study_guide.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "module_id": parsed_module_id,
            "modules": modules,
            "study_material_html": study_material_html,
            "student_id": student_id,
            "courses": courses,
        }
    )
# Modified route handlers to support rubrics

@ai_router.get("/courses/{course_id}/create-assignment", response_class=HTMLResponse)
async def show_create_assignment_form(
    request: Request, 
    course_id: int, 
    db: Session = Depends(get_db), 
    user=Depends(require_role("teacher"))
):
    modules = db.query(CourseModule).filter_by(course_id=course_id).order_by(CourseModule.order_index).all()
    materials = db.query(CourseMaterial).filter(CourseMaterial.course_id == course_id).order_by(CourseMaterial.uploaded_at.desc()).all()
    course = db.query(Course).filter(Course.id == course_id).first()
    teacher_id = user["user_id"]
    courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)
    return templates.TemplateResponse("create_assignment.html", {"request": request, "course_id": course_id, "course":course, "materials": materials, "courses": courses, "modules": modules})

@ai_router.post("/courses/{course_id}/create-assignment", response_class=HTMLResponse)
async def create_assignment(
    request: Request,
    course_id: int,
    title: str = Form(...),
    description: str = Form(...),
    module_id: str = Form(""),
    deadline: str = Form(...),
    db: Session = Depends(get_db),
    user = Depends(require_teacher_or_ta()),
    material_ids: List[int] = Form(None),
):
    # Restriction: Only 3 assignments per day (24h window), per teacher per course
    now = datetime.utcnow()
    window_start = now - timedelta(days=1)
    teacher_id = user["user_id"]

    # Count assignments for this teacher & course in the last 24h
    assignment_count = db.query(Assignment).filter(
        Assignment.teacher_id == teacher_id,
        Assignment.course_id == course_id,
        Assignment.created_at >= window_start
    ).count()
    if assignment_count >= 3:
        raise HTTPException(
            status_code=400,
            detail="You can only create up to 3 assignments per day for this course."
        )
    parsed_module_id = None
    if module_id and module_id.strip():
        try:
            parsed_module_id = int(module_id)
        except ValueError:
            parsed_module_id = None
    deadline_dt = datetime.fromisoformat(deadline)
    assignment = Assignment(
        course_id=course_id,
        title=title,
        description=description,
        deadline=deadline_dt,
        teacher_id=teacher_id,
        module_id=parsed_module_id
    )

    if material_ids:
        assignment.materials = db.query(CourseMaterial).filter(CourseMaterial.id.in_(material_ids)).all()

    db.add(assignment)
    db.commit()
    db.refresh(assignment)

    form_data = await request.form()
    # Process rubrics
    rubric_data = {}
    for key, value in form_data.items():
        if key.startswith('rubric['):
            parts = key.rstrip(']').split('[')
            criterion_index = int(parts[1].rstrip(']'))
            if criterion_index not in rubric_data:
                rubric_data[criterion_index] = {'levels': {}}
            if len(parts) == 3:
                field_name = parts[2].rstrip(']')
                rubric_data[criterion_index][field_name] = value
            elif len(parts) == 5:
                level_index = int(parts[3].rstrip(']'))
                field_name = parts[4].rstrip(']')
                if level_index not in rubric_data[criterion_index]['levels']:
                    rubric_data[criterion_index]['levels'][level_index] = {}
                rubric_data[criterion_index]['levels'][level_index][field_name] = value

    for criterion_data in rubric_data.values():
        criterion = RubricCriterion(
            assignment_id=assignment.id,
            name=criterion_data.get('name', ''),
            weight=int(criterion_data.get('weight', 10))
        )
        db.add(criterion)
        db.commit()
        db.refresh(criterion)

        for level_data in criterion_data['levels'].values():
            level = RubricLevel(
                criterion_id=criterion.id,
                description=level_data.get('description', ''),
                points=float(level_data.get('points', 0))
            )
            db.add(level)

    db.commit()

    return RedirectResponse(f"/ai/teacher/{course_id}/assignments", status_code=303)

@ai_router.get("/assignments/{assignment_id}/submit", response_class=HTMLResponse)
async def show_submit_assignment_form(request: Request, assignment_id: int, db: Session = Depends(get_db),user: dict = Depends(require_role("student"))):
    assignment = db.query(Assignment).filter_by(id=assignment_id).first()
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]

    rubric_criteria = db.query(RubricCriterion).filter(
        RubricCriterion.assignment_id == assignment_id
    ).options(joinedload(RubricCriterion.levels)).all()
    return templates.TemplateResponse("submit_assignment.html", {"request": request, "assignment": assignment, "courses":courses, "rubric_criteria": rubric_criteria})

@ai_router.post("/assignments/{assignment_id}/submit", response_class=HTMLResponse)
async def submit_assignment(
    assignment_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    student_id = user["user_id"]
    assignment = db.query(Assignment).filter_by(id=assignment_id).first()
    if not assignment:
        return HTMLResponse("❌ Assignment not found", status_code=404)

    # Save file
    filename = f"{uuid4()}_{sanitize_filename(file.filename)}"
    s3_key = f"assignment_submissions/{assignment_id}/{student_id}/{filename}"

    upload_success = await upload_file_to_s3(file, s3_key)

    if not upload_success:
        return HTMLResponse("❌ Upload to cloud storage failed!", status_code=500)

    # Check for existing submission
    existing_submission = (
        db.query(AssignmentSubmission)
        .filter_by(assignment_id=assignment_id, student_id=student_id)
        .first()
    )

    # Extract text & AI grade (with/without rubric) - only if allowed
    ai_score = None
    ai_feedback = None
    ai_criteria_eval = None
    perform_ai_evaluation = False
    
    if existing_submission:
        # Use existing submission
        submission = existing_submission
        submission.file_path = s3_key
        submission.submitted_at = datetime.utcnow()
        
        # Handle case where ai_evaluation_count might be None
        current_count = submission.ai_evaluation_count or 0
        
        # Check if we can still use AI evaluation (less than 2 times)
        if current_count < 2:
            perform_ai_evaluation = True
            submission.ai_evaluation_count = current_count + 1
    else:
        # Create new submission
        submission = AssignmentSubmission(
            assignment_id=assignment_id,
            student_id=student_id,
            file_path=s3_key,
            submitted_at=datetime.utcnow(),
            ai_evaluation_count=1  # First evaluation
        )
        db.add(submission)
        perform_ai_evaluation = True  # New submission gets one evaluation
    
    # Perform AI evaluation if allowed
    if perform_ai_evaluation:
        try:
            s3_key = submission.file_path  # e.g., "assignment_submissions/{assignment_id}/{student_id}/{filename}"
            file_ext = os.path.splitext(s3_key)[-1] or ".pdf"
            s3_client = get_s3_client()
            # ---- FIX: Use delete=False, work outside the block ----
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmpfile:
                s3_client.download_fileobj(S3_BUCKET_NAME, s3_key, tmpfile)
                tmpfile_path = tmpfile.name  # save name for after block

            try:
                raw_text = extract_text_from_pdf(tmpfile_path)
            finally:
                try:
                    os.remove(tmpfile_path)
                except Exception:
                    pass
            
            # Try to get rubric criteria (with all levels loaded)
            rubric_criteria = db.query(RubricCriterion).filter(
                RubricCriterion.assignment_id == assignment.id
            ).options(joinedload(RubricCriterion.levels)).all()
            ai_rubric = prepare_rubric_for_ai(rubric_criteria) if rubric_criteria else None
            ai_score, ai_feedback, ai_criteria_eval = evaluate_assignment_text(
                text=raw_text,
                assignment_title=assignment.title,
                assignment_description=assignment.description,
                rubric_criteria=ai_rubric
            )
            # Update submission with AI evaluation results
            submission.ai_score = ai_score
            submission.ai_feedback = ai_feedback

        except Exception as e:
            logging.exception(f"AI Evaluation failed: {e}")
            ai_score, ai_feedback, ai_criteria_eval = None, None, None

    # Commit changes to submission
    db.flush()  # so submission.id is available
    db.commit()

    # Save rubric evaluations per-criterion (only if AI evaluation was performed)
    if perform_ai_evaluation and ai_criteria_eval:
        # Delete previous AI evaluations for this submission (if any)
        db.query(RubricEvaluation).filter(
            RubricEvaluation.submission_id == submission.id,
            RubricEvaluation.graded_by_ai == True
        ).delete(synchronize_session=False)
        
        # Create new evaluations
        for crit in ai_criteria_eval:
            criterion_id = crit.get('criterion_id')
            level_id = crit.get('selected_level_id')
            points = crit.get('points_awarded')
            criterion_feedback = crit.get('feedback')
            evaluation = RubricEvaluation(
                submission_id=submission.id,
                criterion_id=criterion_id,
                level_id=level_id,
                points_awarded=points,
                feedback=criterion_feedback,
                graded_by_ai=True,
                created_at=datetime.utcnow()
            )
            db.add(evaluation)
        db.commit()

    # Generate HTML feedback (overall + rubric, if any):
    rubric_html = ""
    if ai_criteria_eval:
        rubric_html += "<h4 class='font-bold mt-2'>Rubric Evaluation:</h4><ul class='m-2 p-2'>"
        for crit in ai_criteria_eval:
            rubric_html += (
                f"<li><b>{crit.get('name')}</b>: {crit.get('feedback')} "
                f"(Points: {crit.get('points_awarded', '—')})</li>"
            )
        rubric_html += "</ul>"

    # Create appropriate response message based on whether AI evaluation was performed
    evaluation_status = ""
    if not perform_ai_evaluation:
        evaluation_status = "<p class='text-amber-600'>AI evaluation limit reached (max 2 per submission).</p>"
    
    return HTMLResponse(
        f"""<div class='toast success'>
                ✅ Submitted! {evaluation_status}
                {"AI Score: " + str(ai_score) if ai_score is not None else ""}
                <br>
                <small>{ai_feedback if ai_feedback else ""}</small>
                {rubric_html}
            </div>""",
        status_code=200
    )

@ai_router.get("/assignments/submission/{submission_id}/download")
async def download_submission_file(
    submission_id: int, 
    db: Session = Depends(get_db)
):
    submission = db.query(AssignmentSubmission).filter_by(id=submission_id).first()
    if not submission or not submission.file_path:
        return HTMLResponse("❌ Submission not found", status_code=404)
    filename = submission.file_path.split("/")[-1]
    s3_key = submission.file_path

    presigned_url = generate_presigned_url(s3_key, expiration=600)
    if not presigned_url:
        return HTMLResponse("❌ Unable to generate download link.", status_code=404)
    
    return RedirectResponse(presigned_url)

#Function to get into student assignment page
@ai_router.get("/student/courses/{course_id}/assignments", response_class=HTMLResponse)
async def student_course_assignments(
    request: Request,
    course_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    student_id = user["user_id"]
    # Ensure student is enrolled & accepted in this course
    enrollment = (
        db.query(Enrollment)
        .filter_by(student_id=student_id, course_id=course_id, is_accepted=True)
        .first()
    )
    if not enrollment:
        return HTMLResponse("❌ You are not enrolled in this course", status_code=403)
    
    # Get the course info
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
    # Get all assignments for this course
    assignments = (
        db.query(Assignment)
        .filter_by(course_id=course_id)
        .order_by(Assignment.deadline)
        .all()
    )
    
    assignment_ids = [a.id for a in assignments]
    
    # Get submissions with eager loading of comments and user info
    submissions = (
        db.query(AssignmentSubmission)
        .filter(AssignmentSubmission.assignment_id.in_(assignment_ids))
        .filter_by(student_id=student_id)
        .options(
            joinedload(AssignmentSubmission.comments).joinedload(AssignmentComment.user)
        )
        .all()
    )
    
    # Debug log to check for comments
    for s in submissions:
        logging.info(f"Submission ID: {s.id}, Assignment ID: {s.assignment_id}, Comments: {len(s.comments)}")
        for comment in s.comments:
            logging.info(f"Comment: {comment.message} by User: {comment.user.id if comment.user else 'Unknown'}")
    
    # Create a dictionary for easier lookup in template
    submission_dict = {s.assignment_id: s for s in submissions}
    criteria_by_assignment = {}
    levels_by_criterion = {}
    for assignment in assignments:
        criteria = db.query(RubricCriterion).filter_by(assignment_id=assignment.id).all()
        criteria_by_assignment[assignment.id] = criteria
        for crit in criteria:
            levels_by_criterion[crit.id] = db.query(RubricLevel).filter_by(criterion_id=crit.id).order_by(RubricLevel.points.desc()).all()

    # For each submission: get rubric evaluations
    evaluations_by_submission = {}
    for s in submissions:
        evals = db.query(RubricEvaluation).filter_by(submission_id=s.id).all()
        evaluations_by_submission[s.id] = evals
    
    return templates.TemplateResponse(
        "student_assignments.html",
        {
            "request": request,
            "assignments": assignments,
            "submissions": submissions,
            "submission_dict": submission_dict,
            "course": course,
            "courses":courses,
            "criteria_by_assignment": criteria_by_assignment,
            "levels_by_criterion": levels_by_criterion,
            "evaluations_by_submission": evaluations_by_submission,
        },
    )
@ai_router.post("/assignments/{assignment_id}/grade/{submission_id}")
async def grade_submission(
    assignment_id: int,
    submission_id: int,
    teacher_score: Optional[int] = Form(None),
    comment: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher")),
    request: Request = None,
):
    submission = db.query(AssignmentSubmission).join(Assignment).filter(
        Assignment.id == assignment_id,
        Assignment.teacher_id == user["user_id"],
        AssignmentSubmission.id == submission_id
    ).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    # Update the teacher's score
    if teacher_score is not None:
        submission.teacher_score = teacher_score

    # Rubric evaluation handling
    form = await request.form()
    print("==== GRADING SUBMISSION DEBUG ====")
    print("Form fields received:")
    for k, v in form.items():
        print(f"  {k}: {v}")

    all_criteria = db.query(RubricCriterion).filter_by(assignment_id=assignment_id).all()

    for crit in all_criteria:
        crit_id = crit.id
        level_id_raw = form.get(f"rubric_crit_{crit_id}_level")
        feedback = form.get(f"rubric_crit_{crit_id}_feedback", "").strip()
        print(f"\n-- Processing criterion {crit_id} ({crit.name})")
        print(f"   Level: {level_id_raw}")
        print(f"   Feedback: '{feedback}'")
        if not level_id_raw:
            print("   [SKIPPED: No level selected]")
            continue
        chosen_level_id = int(level_id_raw)
        level = db.query(RubricLevel).filter_by(id=chosen_level_id).first()
        print(f"   RubricLevel object: {level} (desc: {level.description if level else 'N/A'})")
        eval = db.query(RubricEvaluation).filter_by(submission_id=submission_id, criterion_id=crit_id).first()
        print(f"   RubricEvaluation found: {bool(eval)}")
        if not eval:
            eval = RubricEvaluation(
                submission_id=submission_id,
                criterion_id=crit_id,
                level_id=chosen_level_id,
                points_awarded=level.points if level else None,
                feedback=feedback,
                graded_by_user_id=user["user_id"]
            )
            db.add(eval)
            print("   [CREATED NEW Eval]")
        else:
            eval.level_id = chosen_level_id
            eval.points_awarded = level.points if level else None
            eval.feedback = feedback
            eval.graded_by_user_id = user["user_id"]
            print("   [UPDATED Eval]")
        print(f"   Saved: level_id={eval.level_id}, points={eval.points_awarded}, feedback='{eval.feedback}'")

    # Add a general comment if provided
    if comment and comment.strip():
        logging.info(f"Adding comment to submission {submission_id}: '{comment}' by user {user['user_id']}")
        new_comment = AssignmentComment(
            submission_id=submission_id,
            user_id=user["user_id"],
            message=comment
        )
        db.add(new_comment)
    db.commit()
    print("==== END GRADING DEBUG ====")
    print(form)
    return RedirectResponse(
        url=f"/ai/assignments/{assignment_id}/submissions",
        status_code=303
    )

@ai_router.get("/assignments/{assignment_id}/submissions", response_class=HTMLResponse)
async def view_submissions(
    assignment_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    assignment = db.query(Assignment).filter_by(id=assignment_id, teacher_id=user["user_id"]).first()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    submissions = db.query(AssignmentSubmission).filter_by(assignment_id=assignment.id).all()
    teacher_id = user["user_id"]
    courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()

    # RUBRICS ADDED
    criteria = db.query(RubricCriterion).filter_by(assignment_id=assignment.id).all()
    levels_by_criterion = {}
    for crit in criteria:
        levels_by_criterion[crit.id] = db.query(RubricLevel).filter_by(criterion_id=crit.id).order_by(RubricLevel.points.desc()).all()
    evals_by_submission = {}
    for s in submissions:
        evals_by_submission[s.id] = db.query(RubricEvaluation).filter_by(submission_id=s.id).all()

    return templates.TemplateResponse("teacher_assignment_submissions.html", {
        "request": request,
        "assignment": assignment,
        "submissions": submissions,
        "courses": courses,
        "criteria": criteria,
        "levels_by_criterion": levels_by_criterion,
        "evals_by_submission": evals_by_submission,
    })

@ai_router.get("/assignments/{assignment_id}/export")
async def export_submissions_to_excel(
    assignment_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    assignment = db.query(Assignment).filter_by(id=assignment_id, teacher_id=user["user_id"]).first()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    submissions = db.query(AssignmentSubmission).filter_by(assignment_id=assignment.id).all()

    data = []
    for s in submissions:
        # Get the latest teacher comment, or join all comments if you prefer
        latest_comment = s.comments[-1].message if s.comments else ""
        data.append({
            "Student": f"{s.student.f_name} {s.student.l_name}",
            "Email": getattr(s.student, "email", ""),
            "Submitted At": s.submitted_at,
            "AI Score": s.ai_score,
            "AI Feedback": s.ai_feedback,
            "Teacher Score": s.teacher_score,
            "Teacher Comment": latest_comment,
            "File": s.file_path
        })

    df = pd.DataFrame(data)
    path = f"exports/assignment_{assignment_id}_submissions.xlsx"
    os.makedirs("exports", exist_ok=True)
    df.to_excel(path, index=False)

    return FileResponse(path, filename=f"submissions_assignment_{assignment_id}.xlsx")


@ai_router.get("/student/assignments", response_class=HTMLResponse)
async def student_assignments(request: Request, db: Session = Depends(get_db), user: dict = Depends(require_role("student"))):
    assignments = db.query(Assignment).all()
    submissions = db.query(AssignmentSubmission).filter_by(student_id=user["user_id"]).all()
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
    print("courses list:", courses)
    return templates.TemplateResponse("student_assignments.html", {
        "request": request,
        "assignments": assignments,
        "submissions": submissions,
        "courses":courses
    })


@ai_router.get("/teacher/{course_id}/assignments", response_class=HTMLResponse)
async def teacher_assignments(request: Request, course_id: int, db: Session = Depends(get_db), user: dict = Depends(require_role("teacher"))):
    # Changed the order to sort by created_at descending to show newest first
    assignments = (
        db.query(Assignment)
        .filter_by(teacher_id=user["user_id"], course_id=course_id)
        .order_by(Assignment.created_at.desc())
        .all()
    )
    modules = db.query(CourseModule).filter_by(course_id=course_id).order_by(CourseModule.order_index).all()
    course = db.query(Course).filter(Course.id == course_id).first()
    teacher_id = user["user_id"]
    courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)
    return templates.TemplateResponse("teacher_assignments.html", {
        "request": request,
        "assignments": assignments,
        "course": course,
        "courses": courses,
        "modules": modules
    })

# Add a new route for editing assignments
@ai_router.get("/assignments/{assignment_id}/edit", response_class=HTMLResponse)
async def edit_assignment_form(request: Request, assignment_id: int, db: Session = Depends(get_db), user: dict = Depends(require_role("teacher"))):
    assignment = db.query(Assignment).filter(Assignment.id == assignment_id).first()
    
    if not assignment:
        return HTMLResponse("❌ Assignment not found", status_code=404)
    
    # Ensure the teacher owns this assignment
    if assignment.teacher_id != user["user_id"]:
        return HTMLResponse("❌ You don't have permission to edit this assignment", status_code=403)
    
    course = db.query(Course).filter(Course.id == assignment.course_id).first()
    courses = db.query(Course).filter(Course.teacher_id == user["user_id"]).all()
    modules = db.query(CourseModule).filter_by(course_id=assignment.course_id).order_by(CourseModule.order_index).all()
    return templates.TemplateResponse("edit_assignment.html", {
        "request": request,
        "assignment": assignment,
        "course": course,
        "courses": courses,
        "modules": modules
    })

# Add a route to handle the assignment update
@ai_router.post("/assignments/{assignment_id}/update", response_class=HTMLResponse)
async def update_assignment(
    request: Request,
    assignment_id: int,
    title: str = Form(...),
    description: str = Form(...),
    deadline: str = Form(None),
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    assignment = db.query(Assignment).filter(Assignment.id == assignment_id).first()
    form_data = await request.form()
    module_id_str = form_data.get("module_id", "")
    module_id = int(module_id_str) if module_id_str and module_id_str.strip() else None
    if not assignment:
        return HTMLResponse("❌ Assignment not found", status_code=404)
    
    # Ensure the teacher owns this assignment
    if assignment.teacher_id != user["user_id"]:
        return HTMLResponse("❌ You don't have permission to edit this assignment", status_code=403)
    
    # Update assignment details
    assignment.title = title
    assignment.description = description
    assignment.module_id = module_id 
    if deadline:
        try:
            deadline_dt = datetime.strptime(deadline, "%Y-%m-%dT%H:%M")
            assignment.deadline = deadline_dt
        except ValueError:
            # If deadline format is invalid, keep the existing one
            pass
    else:
        assignment.deadline = None
    
    db.commit()
    
    # Redirect back to assignments page
    return RedirectResponse(url=f"/ai/teacher/{assignment.course_id}/assignments", status_code=303)

# Add a route to handle assignment deletion
@ai_router.post("/assignments/{assignment_id}/delete", response_class=HTMLResponse)
async def delete_assignment(
    request: Request,
    assignment_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    assignment = db.query(Assignment).filter(Assignment.id == assignment_id).first()
    
    if not assignment:
        return HTMLResponse("❌ Assignment not found", status_code=404)
    
    # Ensure the teacher owns this assignment
    if assignment.teacher_id != user["user_id"]:
        return HTMLResponse("❌ You don't have permission to delete this assignment", status_code=403)
    
    course_id = assignment.course_id
    
    # Delete the assignment
    db.delete(assignment)
    db.commit()
    
    # Redirect back to assignments page
    return RedirectResponse(url=f"/ai/teacher/{course_id}/assignments", status_code=303)

@ai_router.get("/courses/{course_id}/quiz/create", response_class=HTMLResponse)
async def quiz_creation_page(
    request: Request,
    course_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    teacher_id = user["user_id"]
    courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()
   
    # Check quota and get remaining count
    quota_exceeded, remaining = check_quiz_quota(db, teacher_id, course_id)
    
    # Get only modules where ALL submodules are processed
    eligible_modules = []
    all_modules = db.query(CourseModule).filter_by(
        course_id=course_id
    ).order_by(CourseModule.order_index).all()
    
    for module in all_modules:
        # Get all submodules for this module
        submodules = db.query(CourseSubmodule).filter_by(
            module_id=module.id
        ).all()
        
        # Check if module has submodules and all are processed
        if submodules:  # Module has submodules
            all_processed = all(sub.is_processed for sub in submodules)
            if all_processed:
                # Add processing status info to module object
                module.submodule_count = len(submodules)
                module.processed_count = len([s for s in submodules if s.is_processed])
                eligible_modules.append(module)
        # If module has no submodules, we might want to exclude it or handle differently
        # For now, excluding modules without submodules from quiz generation
    
    # Render form to create a quiz for this specific course
    return templates.TemplateResponse(
        "quiz_creator.html",
        {
            "request": request,
            "course_id": course_id,
            "modules": eligible_modules,  # Only fully processed modules
            "topic": request.query_params.get("topic", ""),
            "study_material_html": "",
            "teacher_id": user.get("user_id", ""),
            "question_types": ["mcq"],  # Default
            "num_questions": 10,
            "courses": courses,
            "remaining_quota": remaining,
            "quota_exceeded": quota_exceeded,
            "total_modules": len(all_modules),
            "eligible_modules_count": len(eligible_modules)
        }
    )
def get_template_context(request, course_id, teacher_id, db, **kwargs):
    """Helper to get consistent template context"""
    try:
        _, remaining = check_quiz_quota(db, teacher_id, course_id)
        modules = db.query(CourseModule).filter(CourseModule.course_id == course_id).all()
        courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()
    except:
        remaining = 0
        modules = []
        courses = []
    
    base_context = {
        "request": request,
        "course_id": course_id,
        "teacher_id": teacher_id,
        "remaining_quota": remaining,
        "quota_exceeded": remaining <= 0,
        "modules": modules,
        "courses": courses
    }
    base_context.update(kwargs)
    return base_context

@ai_router.post("/courses/{course_id}/quiz/generate", response_class=HTMLResponse)
async def generate_quiz(
    request: Request,
    course_id: int,
    topic: str = Form(""),  # Changed from Form(...) to Form("") to allow empty values
    module_id: str = Form(""), 
    difficulty: str = Form("medium"), 
    question_types: List[str] = Form(...),
    num_questions: int = Form(10),
    user: dict = Depends(require_role("teacher")),
    db: Session = Depends(get_db),
):
    teacher_id = user.get("user_id", "")
    MAX_QUESTIONS = 20
    if num_questions > MAX_QUESTIONS:
        num_questions = MAX_QUESTIONS
    
    parsed_module_id = None
    if module_id and module_id.strip() and module_id != "":
        try:
            parsed_module_id = int(module_id)
        except ValueError:
            return templates.TemplateResponse(
                "quiz_creator.html",
                get_template_context(
                    request, course_id, teacher_id, db,
                    topic=topic,
                    error="Invalid module selection. Please try again.",
                    question_types=question_types,
                    num_questions=num_questions
                )
            )
    
    # If module is selected but no topic is provided, use module-based quiz generation
    if parsed_module_id and (not topic or not topic.strip()):
        try:
            module = db.query(CourseModule).filter_by(id=parsed_module_id, course_id=course_id).first()
            if module:
                topic = f"Module: {module.title}"
            else:
                return templates.TemplateResponse(
                "quiz_creator.html",
                get_template_context(
                    request, course_id, teacher_id, db,
                    topic=topic,
                    error="Selected module not found.",
                    question_types=question_types,
                    num_questions=num_questions
                )
            )
        except Exception as e:
            return templates.TemplateResponse(
                "quiz_creator.html",
                get_template_context(
                    request, course_id, teacher_id, db,
                    topic=topic,
                    error=f"Error retrieving module information: {str(e)}",
                    question_types=question_types,
                    num_questions=num_questions
                )
            )
    
    # Validate that we have either a topic or a module selected
    if (not topic or not topic.strip()) and not parsed_module_id:
        return templates.TemplateResponse(
            "quiz_creator.html",
            get_template_context(
                request, course_id, teacher_id, db,
                topic=topic,
                error="Please provide a topic or select a module for the quiz.",
                question_types=question_types,
                num_questions=num_questions
            )
        )
    
    try:
        # Check if quota is exceeded
        quota_exceeded, remaining = check_quiz_quota(db, teacher_id, course_id)
        
        if quota_exceeded:
            return templates.TemplateResponse(
            "quiz_creator.html",
            get_template_context(
                request, course_id, teacher_id, db,
                topic=topic,
                error="Daily quiz creation limit reached (5 quizzes per course). Please try again tomorrow.",
                question_types=question_types,
                num_questions=num_questions
            )
        )
                    
        # Generate the quiz materials
        materials_json = generate_study_material_quiz(
            query=topic,
            material_type="quiz",
            course_id=course_id,
            teacher_id=teacher_id,
            question_types=question_types,
            num_questions=num_questions,
            module_id=parsed_module_id,     
            difficulty=difficulty 
        )
        print("Generated materials_json:", materials_json)

        try:
            existing = db.query(Quiz).filter_by(
                course_id=course_id,
                teacher_id=teacher_id,
                topic=topic
            ).first()
            if existing:
                print("Found existing quiz, updating.")
                existing.json_data = materials_json
                quiz = existing
            else:
                print("Creating new quiz for this course/teacher/topic.")
                quiz = Quiz(
                    course_id=course_id,
                    teacher_id=teacher_id,
                    topic=topic,
                    json_data=materials_json,
                    module_id=parsed_module_id,     
                    difficulty=difficulty 

                )
                db.add(quiz)
            db.commit()
            
            # Increment the quota since we successfully created a quiz
            increment_quiz_quota(db, teacher_id, course_id)
            # Get updated remaining count
            _, remaining = check_quiz_quota(db, teacher_id, course_id)
            
        except Exception as db_exc:
            print("DB error:", db_exc)
            return templates.TemplateResponse(
            "quiz_creator.html",
            get_template_context(
                request, course_id, teacher_id, db,
                topic=topic,
                error=f"Database error: {db_exc}",
                remaining_quota=remaining
            )
        )
            
        study_material_html = render_quiz_htmx(materials_json)
        return templates.TemplateResponse(
            "quiz_creator.html",
            {
                "request": request,
                "course_id": course_id,
                "topic": topic,
                "study_material_html": study_material_html,
                "teacher_id": teacher_id,
                "question_types": question_types,
                "num_questions": num_questions,
                "remaining_quota": remaining,
                "difficulty":difficulty,
                "module_id":parsed_module_id,
                "quota_success": True,
                "courses": db.query(Course).filter(Course.teacher_id == teacher_id).all()
            }
        )
    except Exception as e:
        print("Main error:", e)
        error_message = f"An unexpected error occurred: {str(e)}"
        return templates.TemplateResponse(
            "quiz_creator.html",
            get_template_context(
                request, course_id, teacher_id, db,
                topic=topic if 'topic' in locals() else "",
                error=error_message
            )
        )
@ai_router.get("/courses/{course_id}/quiz/export", response_class=HTMLResponse)
async def export_quiz(
    request: Request,
    course_id: int,
    format: str = Query("pdf"),
    include_answers: bool = Query(False),
    user: dict = Depends(require_role("teacher")),
    db: Session = Depends(get_db)      # ⏩ DB!
):
    materials_json = request.session.get("quiz_materials")
    if not materials_json:
        quiz = (db.query(Quiz)
            .filter_by(course_id=course_id, teacher_id=user.get("user_id", ""))
            .order_by(Quiz.created_at.desc())
            .first())
        if not quiz:
            return HTMLResponse("<div>No quiz found. Please generate a quiz first.</div>")
        materials_json = quiz.json_data

    # continue with your export logic!
    s3_key, filename = generate_quiz_export(materials_json, course_id, format, include_answers)
    download_url = f"/ai/courses/{course_id}/quiz/download/{filename}"
    return templates.TemplateResponse(
        "quiz_export.html",
        {
            "request": request,
            "course_id": course_id,
            "export_url": download_url,
            "format": format,
            "filename":filename
        }
    )

@ai_router.get("/courses/{course_id}/quiz/previous", response_class=HTMLResponse)
async def get_previous_quizzes(
    request: Request,
    course_id: int,
    user: dict = Depends(require_role("teacher")),
    db: Session = Depends(get_db)
):
    teacher_id = user.get("user_id", "")
    
    # Get all previous quizzes for this course and teacher
    quizzes = db.query(Quiz)\
        .filter_by(course_id=course_id, teacher_id=teacher_id)\
        .order_by(Quiz.created_at.desc())\
        .all()
    
    if not quizzes:
        return HTMLResponse("""
            <div class="text-center py-8 text-gray-500">
                <div class="text-4xl mb-3">📝</div>
                <p class="font-medium">No previous quizzes found</p>
                <p class="text-sm">Create your first quiz using the form below!</p>
            </div>
        """)
    
    # Group quizzes by module
    quizzes_by_module = {}
    for quiz in quizzes:
        module_name = "All Course Content"
        if quiz.module_id and quiz.module:
            module_name = quiz.module.title
        
        if module_name not in quizzes_by_module:
            quizzes_by_module[module_name] = []
        quizzes_by_module[module_name].append(quiz)
    
    # Generate HTML
    html_parts = []
    
    for module_name, module_quizzes in quizzes_by_module.items():
        html_parts.append(f"""
            <div class="mb-6">
                <h3 class="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
                    📂 {module_name}
                    <span class="text-sm font-normal text-gray-500">({len(module_quizzes)} quiz{'es' if len(module_quizzes) != 1 else ''})</span>
                </h3>
                <div class="grid gap-3">
        """)
        
        for quiz in module_quizzes:
            # Parse quiz data to get question count
            try:
                import json
                quiz_data = json.loads(quiz.json_data)
                question_count = len(quiz_data.get('questions', []))
            except:
                question_count = 'N/A'
            
            # Format date
            created_date = quiz.created_at.strftime("%b %d, %Y at %I:%M %p")
            
            # Difficulty badge color
            difficulty_colors = {
                'easy': 'bg-green-100 text-green-800',
                'medium': 'bg-yellow-100 text-yellow-800', 
                'hard': 'bg-red-100 text-red-800'
            }
            difficulty_color = difficulty_colors.get(quiz.difficulty, 'bg-gray-100 text-gray-800')
            
            html_parts.append(f"""
                <div class="bg-gray-50 border border-gray-200 rounded-lg p-4 hover:bg-gray-100 transition-colors">
                    <div class="flex items-start justify-between">
                        <div class="flex-1">
                            <div class="flex items-center gap-3 mb-2">
                                <h4 class="font-medium text-gray-900">{quiz.topic}</h4>
                                <span class="px-2 py-1 text-xs font-medium rounded-full {difficulty_color}">
                                    {quiz.difficulty.title()}
                                </span>
                            </div>
                            <div class="text-sm text-gray-600 mb-2">
                                <span class="inline-flex items-center gap-1">
                                    📊 {question_count} questions
                                </span>
                                <span class="mx-2">•</span>
                                <span class="inline-flex items-center gap-1">
                                    🕒 {created_date}
                                </span>
                            </div>
                        </div>
                        <div class="flex gap-2 ml-4">
                            <button
                                onclick="downloadQuiz({quiz.id}, '{quiz.topic}', 'pdf')"
                                data-quiz-id="{quiz.id}"
                                class="px-3 py-1.5 text-xs font-medium bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                                title="Download as PDF"
                            >
                                📄 PDF
                            </button>
                            <button
                                onclick="deleteQuiz({quiz.id}, '{quiz.topic}')"
                                data-quiz-id="{quiz.id}"
                                data-action="delete"
                                class="px-3 py-1.5 text-xs font-medium bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
                                title="Delete Quiz"
                            >
                                🗑️ Delete
                            </button>
                        </div>
                    </div>
                </div>
            """)
        
        html_parts.append("</div></div>")
    
    return HTMLResponse("".join(html_parts))


@ai_router.get("/courses/{course_id}/quiz/{quiz_id}/download")
async def download_quiz_by_id(
    course_id: int,
    quiz_id: int,
    format: str = Query("pdf", regex="^(pdf|docx)$"),
    include_answers: bool = Query(True),
    user: dict = Depends(require_role("teacher")),
    db: Session = Depends(get_db)
):
    teacher_id = user.get("user_id", "")
    
    # Get the specific quiz
    quiz = db.query(Quiz)\
        .filter_by(id=quiz_id, course_id=course_id, teacher_id=teacher_id)\
        .first()
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    try:
        # Generate the export using your existing function
        s3_key, filename = generate_quiz_export(
            quiz.json_data, 
            course_id, 
            format, 
            include_answers
        )
        
        # Return download response (adjust based on your file storage setup)
        if s3_key:
            # If using S3, redirect to signed URL
            download_url = generate_s3_download_link(s3_key, filename)
            return RedirectResponse(url=download_url)
        else:
            # If storing locally, return file response
            return FileResponse(
                path=filename,
                filename=f"{quiz.topic}_{format}.{format}",
                media_type='application/octet-stream'
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating download: {str(e)}")

@ai_router.get("/courses/{course_id}/quiz/download/{filename}", response_class=FileResponse)
async def download_quiz_file(
    course_id: int,
    filename: str,
    user: dict = Depends(require_role("teacher"))
):
    filename = os.path.basename(filename)      # Security: strip directory
    s3_key = f"quiz_exports/{course_id}/{filename}"
    print("---- DOWNLOAD DEBUG ----")
    print("Current working directory:", os.getcwd())
    print("-------------------------")
    download_link = generate_s3_download_link(s3_key, filename)
    if not download_link:
        return HTMLResponse(
            f"<div class='bg-red-100 text-red-700 p-4 rounded-lg'>Quiz download failed. File not found in cloud storage.</div>",
            status_code=404
        )
    return RedirectResponse(download_link)

@ai_router.delete("/courses/{course_id}/quiz/{quiz_id}/delete")
async def delete_quiz_by_id(
    course_id: int,
    quiz_id: int,
    user: dict = Depends(require_role("teacher")),
    db: Session = Depends(get_db)
):
    teacher_id = user.get("user_id", "")
    
    # Get the specific quiz to ensure it belongs to this teacher and course
    quiz = db.query(Quiz)\
        .filter_by(id=quiz_id, course_id=course_id, teacher_id=teacher_id)\
        .first()
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    try:
        # Delete the quiz from database
        db.delete(quiz)
        db.commit()
        
        # Optionally: Clean up any associated files (if stored locally)
        # You might want to delete exported files or other related data here
        
        return {"message": "Quiz deleted successfully", "quiz_id": quiz_id}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting quiz: {str(e)}")

@ai_router.get("/student/courses/{course_id}/engagement", response_class=HTMLResponse)
async def student_engagement_activities(
    request: Request,
    course_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    enrollment = db.query(Enrollment).filter_by(
        student_id=user["user_id"],
        course_id=course_id,
        is_accepted=True
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="You are not enrolled in this course")
    
    course = db.query(Course).filter_by(id=course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    student_activities = db.query(StudentActivity).filter_by(
        student_id=user["user_id"], course_id=course_id
    ).order_by(StudentActivity.created_at.desc()).all()
    
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
    
    # Get only published modules for the course
    modules = db.query(CourseModule).filter_by(
        course_id=course_id, 
        is_published=True
    ).order_by(CourseModule.order_index).all()
    
    # Calculate how many activities were done today
    today = datetime.utcnow().date()
    activities_today = sum(1 for activity in student_activities if activity.created_at.date() == today)
    daily_limit_reached = activities_today >= 5
    
    return templates.TemplateResponse("student_engagement.html", {
        "request": request,
        "course": course,
        "courses": courses,
        "modules": modules,
        "student_activities": student_activities,
        "activities_today": activities_today,
        "daily_limit": 5,
        "daily_limit_reached": daily_limit_reached
    })

# Enhanced helper function to get context with module support
def get_engagement_context(course_id: int, query: str, module_id: Optional[int] = None) -> str:
    """Get context for engagement activities with optional module focus."""
    try:
        context_result = retrieve_course_context(course_id, query, module_id)
        
        # Check if we have an error (no content available)
        if isinstance(context_result, dict) and "error" in context_result:
            # Return empty context if no specific content found
            return ""
        
        return context_result if isinstance(context_result, str) else ""
    except Exception as e:
        print(f"Error retrieving engagement context: {e}")
        return ""
# Helper function to check daily activity limit
def check_daily_activity_limit(db: Session, student_id: int, course_id: int):
    today = datetime.utcnow().date()
    today_start = datetime.combine(today, datetime.min.time())
    today_end = datetime.combine(today, datetime.max.time())
    
    # Count activities created today for this course
    activities_today = db.query(StudentActivity).filter(
        StudentActivity.student_id == student_id,
        StudentActivity.course_id == course_id,
        StudentActivity.created_at >= today_start,
        StudentActivity.created_at <= today_end
    ).count()
    
    return activities_today, activities_today >= 5

# Muddiest Point endpoint
# Fixed Muddiest Point Endpoint
@ai_router.post("/student/courses/{course_id}/muddiest-point", response_class=HTMLResponse)
async def process_muddiest_point(
    request: Request,
    course_id: int,
    topic: str = Form(...),
    confusion: str = Form(...),
    module_id: str = Form(""),  # Changed to str to handle empty values
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    enrollment = db.query(Enrollment).filter_by(
        student_id=user["user_id"], course_id=course_id, is_accepted=True
    ).first()
    if not enrollment:
        return JSONResponse(content={"error": "You are not enrolled in this course"}, status_code=403)
    
    # Convert module_id from string to int or None
    parsed_module_id = None
    if module_id and module_id.strip() and module_id != "":
        try:
            parsed_module_id = int(module_id)
        except ValueError:
            return JSONResponse(content={"error": "Invalid module ID format"}, status_code=400)
    
    # Validate module_id if provided
    if parsed_module_id:
        module = db.query(CourseModule).filter_by(
            id=parsed_module_id, 
            course_id=course_id, 
            is_published=True
        ).first()
        if not module:
            return JSONResponse(content={"error": "Invalid or unpublished module selected"}, status_code=400)
    
    # Check daily activity limit
    activities_today, limit_reached = check_daily_activity_limit(db, user["user_id"], course_id)
    if limit_reached:
        return JSONResponse(
            content={"error": f"Daily limit of 5 engagement activities per course reached. Please try again tomorrow."},
            status_code=429
        )
    
    query = f"Topic: {topic}. Confusion: {confusion}"
    try:
        # Get context with optional module focus
        context = get_engagement_context(course_id, query, parsed_module_id)
        chat = get_openai_client()
        
        # Determine context scope for prompt
        context_scope = "the specific module content" if parsed_module_id else "the course materials"
        context_note = f" (focusing on module content)" if parsed_module_id else " (using all available course content)"
        
        system_prompt = f"""
You are an educational AI tutor analyzing a student's confusion about a topic. Use the provided {context_scope} to:
1. Identify the core confusion areas
2. Suggest specific review topics that would help clarify understanding
3. Provide a list of key resources or concepts to focus on
Base your analysis ONLY on the context provided{context_note}.
CONTEXT FROM COURSE MATERIALS: {context}
        """
        
        user_message = f"""
Topic: {topic}
Student's confusion: {confusion}
Analyze this confusion, identify the core confusion areas, and provide suggestions for review.
Return your response as valid JSON with the following structure:
{{ "summary": "...", "confusion_areas": [ ... ], "review_topics": [ ... ], "resources": [ ... ] }}
        """

        response = chat.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ])
        
        ai_text = getattr(response, "content", response)
        
        try:
            ai_response = json.loads(ai_text)
        except Exception as e:
            return JSONResponse(content={"error": "Failed to parse AI response: " + str(e)}, status_code=500)

        # Save activity with module information
        new_activity = StudentActivity(
            student_id=user["user_id"],
            course_id=course_id,
            activity_type="Muddiest Point",
            topic=topic,
            user_input=confusion,
            ai_response=json.dumps(ai_response),
            module_id=parsed_module_id,  # Store module reference
            created_at=datetime.utcnow()
        )
        db.add(new_activity)
        db.commit()

        return templates.TemplateResponse("muddiest_point_response.html", {
            "request": request,
            "ai_response": ai_response,
            "activities_today": activities_today + 1,
            "daily_limit": 5,
            "module_id": parsed_module_id
        })
    except HTTPException as e:
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Fixed Misconception Check Endpoint
@ai_router.post("/student/courses/{course_id}/misconception-check", response_class=HTMLResponse)
async def process_misconception_check(
    request: Request,
    course_id: int,
    topic: str = Form(...),
    beliefs: str = Form(...),
    module_id: str = Form(""),  # Changed to str to handle empty values
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    enrollment = db.query(Enrollment).filter_by(
        student_id=user["user_id"], course_id=course_id, is_accepted=True
    ).first()
    if not enrollment:
        return JSONResponse(content={"error": "You are not enrolled in this course"}, status_code=403)
    
    # Convert module_id from string to int or None
    parsed_module_id = None
    if module_id and module_id.strip() and module_id != "":
        try:
            parsed_module_id = int(module_id)
        except ValueError:
            return JSONResponse(content={"error": "Invalid module ID format"}, status_code=400)
    
    # Validate module_id if provided
    if parsed_module_id:
        module = db.query(CourseModule).filter_by(
            id=parsed_module_id, 
            course_id=course_id, 
            is_published=True
        ).first()
        if not module:
            return JSONResponse(content={"error": "Invalid or unpublished module selected"}, status_code=400)
    
    # Check daily activity limit
    activities_today, limit_reached = check_daily_activity_limit(db, user["user_id"], course_id)
    if limit_reached:
        return JSONResponse(
            content={"error": f"Daily limit of 5 engagement activities per course reached. Please try again tomorrow."},
            status_code=429
        )
    
    query = f"Topic: {topic}. Student beliefs: {beliefs}"
    try:
        # Get context with optional module focus
        context = get_engagement_context(course_id, query, parsed_module_id)
        chat = get_openai_client()

        # Determine context scope for prompt
        context_scope = "the specific module content" if parsed_module_id else "the course materials"
        context_note = f" (focusing on module content)" if parsed_module_id else " (using all available course content)"

        system_prompt = f"""
You are an educational AI tutor analyzing a student's beliefs or understanding about a topic. Use the provided {context_scope} to:
1. Compare the student's stated beliefs with accurate information
2. Identify any misconceptions
3. Provide helpful corrections and explanations
4. Suggest resources for further learning
Base your analysis ONLY on the context provided{context_note}.
CONTEXT FROM COURSE MATERIALS: {context}
        """
        
        user_message = f"""
Topic: {topic}
Student's beliefs: {beliefs}
Analyze these beliefs, identify what's accurate and what may be misconceptions.
Return your response as valid JSON with the following structure:
{{ "summary": "...", "beliefs": [{{ "statement": "...", "is_accurate": true/false, "explanation": "..." }}], "resources": [ ... ] }}
        """

        response = chat.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ])
        
        ai_text = getattr(response, "content", response)
        
        try:
            ai_response = json.loads(ai_text)
        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to parse AI response: {e}\nRaw response: {ai_text}"},
                status_code=500
            )
            
        # Save activity with module information
        new_activity = StudentActivity(
            student_id=user["user_id"],
            course_id=course_id,
            activity_type="Misconception Check",
            topic=topic,
            user_input=beliefs,
            ai_response=json.dumps(ai_response),
            module_id=parsed_module_id,  # Store module reference
            created_at=datetime.utcnow()
        )
        db.add(new_activity)
        db.commit()
        
        return templates.TemplateResponse("misconception_response.html", {
            "request": request,
            "ai_response": ai_response,
            "activities_today": activities_today + 1,
            "daily_limit": 5,
            "module_id": parsed_module_id
        })
    except HTTPException as e:
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

# View activity detail
@ai_router.get("/student/activities/{activity_id}", response_class=HTMLResponse)
async def view_activity_detail(
    request: Request,
    activity_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    activity = db.query(StudentActivity).filter_by(
        id=activity_id, student_id=user["user_id"]
    ).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found or you don't have access")
    ai_response = json.loads(activity.ai_response)
    if activity.activity_type == "Muddiest Point":
        # Render muddiest point response component as HTML for detail page
        ai_response_html = templates.get_template("muddiest_point_response.html").render(
            request=request,
            ai_response=ai_response
        )
    elif activity.activity_type == "Misconception Check":
        # Render misconception response component as HTML for detail page
        ai_response_html = templates.get_template("misconception_response.html").render(
            request=request,
            ai_response=ai_response
        )
    else:
        ai_response_html = "<div class='p-4 bg-gray-100 rounded text-gray-600'>No AI response data available for this activity.</div>"

    return templates.TemplateResponse("activity_detail.html", {
        "request": request,
        "activity": activity,
        "ai_response_html": ai_response_html
    })

def expire_activity_if_due(activity, db):
    if activity.is_active and activity.started_at:
        now = datetime.utcnow()
        expected_end = activity.started_at + timedelta(minutes=activity.duration_minutes or 0)
        if now >= expected_end:
            activity.is_active = False
            activity.ended_at = expected_end
            db.commit()

@ai_router.get("/teacher/courses/{course_id}/inclass-activities", response_class=HTMLResponse)
async def teacher_inclass_activities(
    request: Request,
    course_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    course = db.query(Course).filter_by(id=course_id, teacher_id=user["user_id"]).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Get course modules for selection
    modules = db.query(CourseModule).filter_by(course_id=course_id, is_active=True).order_by(CourseModule.order_index).all()
    
    # Get existing activities
    activities = db.query(InClassActivity).filter_by(course_id=course_id).order_by(InClassActivity.created_at.desc()).all()
    for act in activities:
        expire_activity_if_due(act, db)
    
    return templates.TemplateResponse("teacher_inclass_activities.html", {
        "request": request,
        "course": course,
        "modules": modules,
        "activities": activities
    })

@ai_router.post("/teacher/courses/{course_id}/inclass-activities/create")
async def create_inclass_activity(
    request: Request,
    course_id: int,
    activity_name: str = Form(...),
    activity_type: str = Form(...),
    participation_type: str = Form(...),
    complexity: str = Form(...),
    module_id: int = Form(None),
    duration_minutes: int = Form(15),
    instructions: str = Form(""),
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    course = db.query(Course).filter_by(id=course_id, teacher_id=user["user_id"]).first()
    if not course:
        return JSONResponse(content={"error": "Course not found"}, status_code=404)
    
    # Create activity
    activity = InClassActivity(
        course_id=course_id,
        module_id=module_id if module_id else None,
        teacher_id=user["user_id"],
        activity_name=activity_name,
        activity_type=activity_type,
        participation_type=participation_type,
        complexity=complexity,
        duration_minutes=duration_minutes,
        instructions=instructions,
        is_active=False
    )
    
    db.add(activity)
    db.commit()
    db.refresh(activity)
    
    activities = db.query(InClassActivity).filter_by(course_id=course_id).order_by(InClassActivity.created_at.desc()).all()
    
    activity_html = ""
    for act in activities:
        status_badge = ""
        action_buttons = ""
        timer_display = ""
        
        if act.is_active:
            status_badge = '''
            <div class="text-sm">
                <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    🟢 Active
                </span>
                <span class="ml-2 text-gray-500" id="timer-{0}">
                    Time remaining: <span class="font-mono"></span>
                </span>
            </div>
            <script>startTimer({0}, {1}, '{2}Z');</script>
            '''.format(act.id, act.duration_minutes, act.started_at.isoformat() if act.started_at else '')
            
            action_buttons = f'''
            <button hx-post="/ai/teacher/activities/{act.id}/end"
                    hx-target="#activity-{act.id}"
                    hx-swap="outerHTML"
                    class="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600 transition">
                End Activity
            </button>
            <a href="/ai/teacher/activities/{act.id}/monitor"
               class="bg-green-500 text-white px-3 py-1 rounded text-sm hover:bg-green-600 transition inline-block">
                Monitor
            </a>
            '''
        else:
            action_buttons = f'''
            <button hx-post="/ai/teacher/activities/{act.id}/start"
                    hx-target="#activity-{act.id}"
                    hx-swap="outerHTML"
                    class="bg-blue-500 text-white px-3 py-1 rounded text-sm hover:bg-blue-600 transition">
                Start Activity
            </button>
            '''
        
        module_text = f"• {act.module.title}" if act.module else ""
        
        activity_html += f'''
        <div class="border rounded-lg p-4" id="activity-{act.id}">
          <div class="flex justify-between items-start">
            <div class="flex-1">
              <div class="flex items-center gap-2 mb-2">
                <h3 class="font-semibold text-lg" id="activity-name-{act.id}">{act.activity_name}</h3>
                <button onclick="editActivityName({act.id}, '{act.activity_name}')" 
                        class="text-blue-500 hover:text-blue-700 text-sm">
                  ✏️ Edit
                </button>
              </div>
              <p class="text-sm text-gray-600 mb-1">
                {act.activity_type.title()} • {act.participation_type.title()} • {act.complexity.title()}
                {module_text}
              </p>
              <p class="text-sm text-gray-600 mb-1">
                Duration: {act.duration_minutes} minutes
              </p>
              {status_badge}
              <p class="text-xs text-gray-500 mt-1">
                Created {act.created_at.strftime('%Y-%m-%d %H:%M')}
              </p>
            </div>
            <div class="flex gap-2 ml-4">
              {action_buttons}
              <button hx-delete="/ai/teacher/activities/{act.id}/delete"
                      hx-target="#activity-{act.id}"
                      hx-swap="outerHTML"
                      hx-confirm="Are you sure you want to delete this activity?"
                      class="bg-gray-500 text-white px-3 py-1 rounded text-sm hover:bg-gray-600 transition">
                Delete
              </button>
            </div>
          </div>
        </div>
        '''
    
    if not activity_html:
        activity_html = '<p class="text-gray-500 text-center py-8">No activities created yet.</p>'
    
    return HTMLResponse(content=activity_html)

@ai_router.post("/teacher/activities/{activity_id}/start")
async def start_activity(
    activity_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    activity = db.query(InClassActivity).filter_by(id=activity_id, teacher_id=user["user_id"]).first()
    if not activity:
        return HTMLResponse(content="<div class='text-red-500'>Activity not found</div>", status_code=404)
    
    activity.started_at = datetime.utcnow()
    activity.is_active = True
    db.commit()
    
    # Return updated activity HTML
    module_text = f"• {activity.module.title}" if activity.module else ""
    
    return HTMLResponse(f'''
    <div class="border rounded-lg p-4" id="activity-{activity.id}">
      <div class="flex justify-between items-start">
        <div class="flex-1">
          <div class="flex items-center gap-2 mb-2">
            <h3 class="font-semibold text-lg" id="activity-name-{activity.id}">{activity.activity_name}</h3>
            <button onclick="editActivityName({activity.id}, '{activity.activity_name}')" 
                    class="text-blue-500 hover:text-blue-700 text-sm">
              ✏️ Edit
            </button>
          </div>
          <p class="text-sm text-gray-600 mb-1">
            {activity.activity_type.title()} • {activity.participation_type.title()} • {activity.complexity.title()}
            {module_text}
          </p>
          <p class="text-sm text-gray-600 mb-1">
            Duration: {activity.duration_minutes} minutes
          </p>
          <div class="text-sm">
            <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
              🟢 Active
            </span>
            <span class="ml-2 text-gray-500" id="timer-{activity.id}">
              Time remaining: <span class="font-mono"></span>
            </span>
          </div>
          <p class="text-xs text-gray-500 mt-1">
            Created {activity.created_at.strftime('%Y-%m-%d %H:%M')}
          </p>
        </div>
        <div class="flex gap-2 ml-4">
          <button hx-post="/ai/teacher/activities/{activity.id}/end"
                  hx-target="#activity-{activity.id}"
                  hx-swap="outerHTML"
                  class="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600 transition">
            End Activity
          </button>
          <a href="/ai/teacher/activities/{activity.id}/monitor"
             class="bg-green-500 text-white px-3 py-1 rounded text-sm hover:bg-green-600 transition inline-block">
            Monitor
          </a>
          <button hx-delete="/ai/teacher/activities/{activity.id}/delete"
                  hx-target="#activity-{activity.id}"
                  hx-swap="outerHTML"
                  hx-confirm="Are you sure you want to delete this activity?"
                  class="bg-gray-500 text-white px-3 py-1 rounded text-sm hover:bg-gray-600 transition">
            Delete
          </button>
        </div>
      </div>
    </div>
    <script>startTimer({activity.id}, {activity.duration_minutes}, '{activity.started_at.isoformat()}Z');</script>
    ''')

@ai_router.post("/teacher/activities/{activity_id}/end")
async def end_activity(
    activity_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    activity = db.query(InClassActivity).filter_by(id=activity_id, teacher_id=user["user_id"]).first()
    if not activity:
        return HTMLResponse(content="<div class='text-red-500'>Activity not found</div>", status_code=404)
    
    activity.ended_at = datetime.utcnow()
    activity.is_active = False
    db.commit()
    
    # Return updated activity HTML
    module_text = f"• {activity.module.title}" if activity.module else ""
    
    return HTMLResponse(f'''
    <div class="border rounded-lg p-4" id="activity-{activity.id}">
      <div class="flex justify-between items-start">
        <div class="flex-1">
          <div class="flex items-center gap-2 mb-2">
            <h3 class="font-semibold text-lg" id="activity-name-{activity.id}">{activity.activity_name}</h3>
            <button onclick="editActivityName({activity.id}, '{activity.activity_name}')" 
                    class="text-blue-500 hover:text-blue-700 text-sm">
              ✏️ Edit
            </button>
          </div>
          <p class="text-sm text-gray-600 mb-1">
            {activity.activity_type.title()} • {activity.participation_type.title()} • {activity.complexity.title()}
            {module_text}
          </p>
          <p class="text-sm text-gray-600 mb-1">
            Duration: {activity.duration_minutes} minutes
          </p>
          <p class="text-xs text-gray-500 mt-1">
            Created {activity.created_at.strftime('%Y-%m-%d %H:%M')}
          </p>
        </div>
        <div class="flex gap-2 ml-4">
          <button hx-post="/ai/teacher/activities/{activity.id}/start"
                  hx-target="#activity-{activity.id}"
                  hx-swap="outerHTML"
                  class="bg-blue-500 text-white px-3 py-1 rounded text-sm hover:bg-blue-600 transition">
            Start Activity
          </button>
          <button hx-delete="/ai/teacher/activities/{activity.id}/delete"
                  hx-target="#activity-{activity.id}"
                  hx-swap="outerHTML"
                  hx-confirm="Are you sure you want to delete this activity?"
                  class="bg-gray-500 text-white px-3 py-1 rounded text-sm hover:bg-gray-600 transition">
            Delete
          </button>
        </div>
      </div>
    </div>
    ''')

@ai_router.delete("/teacher/activities/{activity_id}/delete")
async def delete_activity(
    activity_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    activity = db.query(InClassActivity).filter_by(id=activity_id, teacher_id=user["user_id"]).first()
    if not activity:
        return HTMLResponse(content="<div class='text-red-500'>Activity not found</div>", status_code=404)
    
    # Delete related records first (if any)
    db.query(ActivityParticipation).filter_by(activity_id=activity_id).delete()
    db.query(ActivityGroup).filter_by(activity_id=activity_id).delete()
    
    # Delete the activity
    db.delete(activity)
    db.commit()
    
    # Return empty content (this will remove the activity from the DOM)
    return HTMLResponse(content="")

@ai_router.post("/teacher/activities/{activity_id}/update-name")
async def update_activity_name(
    activity_id: int,
    activity_name: str = Form(...),
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    activity = db.query(InClassActivity).filter_by(id=activity_id, teacher_id=user["user_id"]).first()
    if not activity:
        return JSONResponse(content={"success": False, "error": "Activity not found"}, status_code=404)
    
    activity.activity_name = activity_name
    db.commit()
    
    return JSONResponse(content={"success": True})

@ai_router.get("/teacher/activities/{activity_id}/monitor", response_class=HTMLResponse)
async def monitor_activity(
    request: Request,
    activity_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    activity = db.query(InClassActivity).filter_by(id=activity_id, teacher_id=user["user_id"]).first()
    expire_activity_if_due(activity, db)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    
    # Get participations with student info
    participations = db.query(ActivityParticipation).filter_by(activity_id=activity_id).all()
    
    # Get groups if group activity
    groups = []
    if activity.participation_type == "group":
        groups = db.query(ActivityGroup).filter_by(activity_id=activity_id).all()
    
    return templates.TemplateResponse("activity_monitor.html", {
        "request": request,
        "activity": activity,
        "participations": participations,
        "groups": groups
    })


@ai_router.get("/student/courses/{course_id}/active-activities", response_class=HTMLResponse)
async def student_active_activities(
    request: Request,
    course_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    # Check enrollment
    enrollment = db.query(Enrollment).filter_by(
        student_id=user["user_id"], course_id=course_id, is_accepted=True
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in course")
    
    # Get active activities
    active_activities = db.query(InClassActivity).filter_by(
        course_id=course_id, is_active=True
    ).filter(InClassActivity.started_at.isnot(None)).all()
    
    return templates.TemplateResponse("student_active_activities.html", {
        "request": request,
        "course": enrollment.course,
        "activities": active_activities
    })

@ai_router.post("/student/activities/{activity_id}/join")
async def join_activity_fixed(
    activity_id: int,
    group_name: str = Form(None),
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    activity = db.query(InClassActivity).filter_by(id=activity_id, is_active=True).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    
    # Check if already participating
    existing = db.query(ActivityParticipation).filter_by(
        activity_id=activity_id, student_id=user["user_id"]
    ).first()
    if existing:
        return RedirectResponse(url=f"/ai/student/activities/{activity_id}/work", status_code=303)
    
    group_id = None
    if activity.participation_type == "group":
        if not group_name:
            raise HTTPException(status_code=400, detail="Group name required")
        
        # Find or create group
        group = db.query(ActivityGroup).filter_by(
            activity_id=activity_id, group_name=group_name
        ).first()
        if not group:
            group = ActivityGroup(activity_id=activity_id, group_name=group_name)
            db.add(group)
            db.flush()
        group_id = group.id
    
    # Create participation
    participation = ActivityParticipation(
        activity_id=activity_id,
        student_id=user["user_id"],
        group_id=group_id
    )
    
    db.add(participation)
    db.commit()
    
    return RedirectResponse(url=f"/ai/student/activities/{activity_id}/work", status_code=303)

@ai_router.get("/student/activities/{activity_id}/groups")
async def get_activity_groups(
    activity_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    activity = db.query(InClassActivity).filter_by(id=activity_id, is_active=True).first()
    if not activity:
        return JSONResponse(content={"error": "Activity not found"}, status_code=404)
    
    groups = db.query(ActivityGroup).filter_by(activity_id=activity_id).all()
    
    group_data = []
    for group in groups:
        member_count = db.query(ActivityParticipation).filter_by(
            activity_id=activity_id, group_id=group.id
        ).count()
        
        group_data.append({
            "id": group.id,
            "name": group.group_name,
            "member_count": member_count
        })
    
    return JSONResponse(content={"groups": group_data})

async def check_and_end_expired_activities():
    """Background task to automatically end activities that have exceeded their duration."""
    while True:
        try:
            db = next(get_db())
            current_time = datetime.utcnow()
            
            # Find active activities that should be ended
            active_activities = db.query(InClassActivity).filter_by(is_active=True).all()
            
            for activity in active_activities:
                if activity.started_at:
                    # Calculate when the activity should end
                    end_time = activity.started_at + timedelta(minutes=activity.duration_minutes)
                    
                    if current_time >= end_time:
                        # Auto-end the activity
                        activity.ended_at = current_time
                        activity.is_active = False
                        db.commit()
                        print(f"Auto-ended activity {activity.id}: {activity.activity_name}")
            
            db.close()
            
        except Exception as e:
            print(f"Error in background task: {e}")
        
        # Check every 30 seconds
        await asyncio.sleep(30)
# Peer Quiz Builder Activity
@ai_router.post("/student/activities/{activity_id}/peer-quiz/submit")
async def submit_peer_quiz(
    request: Request,
    activity_id: int,
    questions: str = Form(...),  # JSON string of questions
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    participation = db.query(ActivityParticipation).filter_by(
        activity_id=activity_id, student_id=user["user_id"]
    ).first()
    if not participation:
        raise HTTPException(status_code=404, detail="Not participating in activity")
    
    try:
        quiz_data = json.loads(questions)
        
        # Validate quiz data
        if not isinstance(quiz_data, list) or len(quiz_data) != 5:
            raise HTTPException(status_code=400, detail="Quiz must have exactly 5 questions")
        
        for i, question in enumerate(quiz_data, 1):
            if not all(key in question for key in ['question', 'options', 'correct']):
                raise HTTPException(status_code=400, detail=f"Question {i} missing required fields")
            if not all(option in question['options'] for option in ['a', 'b', 'c', 'd']):
                raise HTTPException(status_code=400, detail=f"Question {i} missing options")
        
        # AI evaluation of quiz quality
        activity = db.query(InClassActivity).filter_by(id=activity_id).first()
        if not activity:
            raise HTTPException(status_code=404, detail="Activity not found")
        module_id = activity.module_id
        course_id = activity.course_id
        # Retrieve context for the module if available, or for the whole course
        context = retrieve_course_context(course_id, "Quiz questions about course material", module_id=module_id)
        if isinstance(context, dict) and "error" in context:
            raise HTTPException(status_code=404, detail=context["error"])
        chat = get_openai_client()
        
        # Format questions for AI review
        questions_text = ""
        for i, q in enumerate(quiz_data, 1):
            questions_text += f"Question {i}: {q['question']}\n"
            questions_text += f"A) {q['options']['a']}\n"
            questions_text += f"B) {q['options']['b']}\n"
            questions_text += f"C) {q['options']['c']}\n"
            questions_text += f"D) {q['options']['d']}\n"
            questions_text += f"Correct: {q['correct'].upper()}\n\n"
        
        system_prompt = f"""
You are evaluating student-created quiz questions for educational quality.
Analyze the questions for:
1. Clarity and appropriate difficulty level
2. Alignment with learning objectives
3. Question type variety and educational value
4. Accuracy of content and correct answers
5. Distractors quality (incorrect options should be plausible)

Provide specific feedback for improvement and highlight strengths.
Context from course materials: {context}
        """
        
        response = chat.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Evaluate these quiz questions:\n\n{questions_text}"}
        ])
        
        ai_feedback_text = response.content if hasattr(response, "content") else str(response)
        
        # Update participation
        participation.submission_data = quiz_data
        participation.ai_feedback = {"feedback": ai_feedback_text, "score": "pending"}
        participation.status = "submitted"
        participation.submitted_at = datetime.utcnow()
        
        db.commit()
        
        # Redirect back to work page to show results
        return RedirectResponse(
            url=f"/ai/student/activities/{activity_id}/work", 
            status_code=303
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid quiz data format")
    except Exception as e:
        #logger.error(f"Error submitting peer quiz: {str(e)}")
        print(f"Error submitting peer quiz: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing quiz submission")
    
@ai_router.get("/teacher/activities/{activity_id}/participation/{participation_id}/ai-feedback", response_class=HTMLResponse)
async def teacher_view_ai_feedback(
    request: Request,
    activity_id: int,
    participation_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    # Optionally, check the teacher owns this activity/course for security!
    participation = db.query(ActivityParticipation).filter_by(id=participation_id, activity_id=activity_id).first()
    if not participation or not participation.ai_feedback:
        return HTMLResponse("<div class='p-6 bg-gray-100 rounded'>No AI feedback available.</div>")
    feedback = participation.ai_feedback
    if isinstance(feedback, dict):
        feedback_text = feedback.get("feedback", "")
    else:
        # fallback in case it's stored as string
        feedback_text = feedback

    return templates.TemplateResponse(
        "partial_ai_feedback_modal.html",  # name your modal partial here
        {"request": request, "student": participation.student, "feedback_text": feedback_text}
    )

# Concept Mapping Activity  
@ai_router.post("/student/activities/{activity_id}/concept-map/submit")
async def submit_concept_map(
    activity_id: int,
    concepts: str = Form(...),
    connections: str = Form(...),
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    participation = db.query(ActivityParticipation).filter_by(
        activity_id=activity_id, student_id=user["user_id"]
    ).first()
    if not participation:
        return JSONResponse(content={"error": "Not participating in activity"}, status_code=404)
    
    try:
        map_data = {
            "concepts": json.loads(concepts),
            "connections": json.loads(connections)
        }
        
        # AI evaluation of concept map
        retriever = get_course_retriever(participation.activity.course_id)
        context = get_context_for_query(retriever, f"Concept map: {concepts} {connections}")
        chat = get_openai_client()
        
        system_prompt = f"""
You are evaluating a student-created concept map for understanding.
Analyze for:
1. Concept accuracy and relevance
2. Connection quality and logic
3. Completeness of understanding
4. Missing key concepts or relationships
CONTEXT: {context}
        """
        
        response = chat.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Evaluate this concept map - Concepts: {concepts}, Connections: {connections}"}
        ])
        
        ai_feedback = json.loads(response.content if hasattr(response, "content") else response)
        
        # Update participation
        participation.submission_data = map_data
        participation.ai_feedback = ai_feedback
        participation.status = "submitted"
        participation.submitted_at = datetime.utcnow()
        
        db.commit()
        
        return templates.TemplateResponse("concept_map_feedback.html", {
            "request": request,
            "ai_feedback": ai_feedback,
            "map_data": map_data
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@ai_router.get("/student/activities/{activity_id}/join-page", response_class=HTMLResponse)
async def activity_join_page(
    request: Request,
    activity_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    activity = db.query(InClassActivity).filter_by(id=activity_id, is_active=True).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    
    # Check if already participating
    existing_participation = db.query(ActivityParticipation).filter_by(
        activity_id=activity_id, student_id=user["user_id"]
    ).first()
    
    # Get all groups for this activity
    groups = db.query(ActivityGroup).filter_by(activity_id=activity_id).all()
    group_data = []
    for group in groups:
        member_count = db.query(ActivityParticipation).filter_by(
            activity_id=activity_id, group_id=group.id
        ).count()
        
        # Get member names
        members = db.query(ActivityParticipation, User).join(User).filter(
            ActivityParticipation.activity_id == activity_id,
            ActivityParticipation.group_id == group.id
        ).all()
        
        group_data.append({
            "group": group,
            "member_count": member_count,
            "members": [member.User.f_name for member in members],
            "is_owner": any(member.ActivityParticipation.student_id == user["user_id"] for member in members)
        })
    
    return templates.TemplateResponse("activity_join.html", {
        "request": request,
        "activity": activity,
        "existing_participation": existing_participation,
        "groups": group_data,
        "user_id": user["user_id"]
    })

# Group management routes
@ai_router.post("/student/groups/{group_id}/rename")
async def rename_group(
    group_id: int,
    new_name: str = Form(...),
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    # Check if user is in this group
    participation = db.query(ActivityParticipation).filter_by(
        group_id=group_id, student_id=user["user_id"]
    ).first()
    if not participation:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    group = db.query(ActivityGroup).filter_by(id=group_id).first()
    group.group_name = new_name
    db.commit()
    
    return RedirectResponse(url=f"/ai/student/activities/{group.activity_id}/join-page", status_code=303)

@ai_router.post("/student/groups/{group_id}/delete")
async def delete_group(
    group_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    # Check if user is in this group
    participation = db.query(ActivityParticipation).filter_by(
        group_id=group_id, student_id=user["user_id"]
    ).first()
    if not participation:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    group = db.query(ActivityGroup).filter_by(id=group_id).first()
    activity_id = group.activity_id
    
    # Delete all participations first
    db.query(ActivityParticipation).filter_by(group_id=group_id).delete()
    # Delete group
    db.delete(group)
    db.commit()
    
    return RedirectResponse(url=f"/ai/student/activities/{activity_id}/join-page", status_code=303)

@ai_router.post("/student/groups/{group_id}/leave")
async def leave_group(
    group_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    participation = db.query(ActivityParticipation).filter_by(
        group_id=group_id, student_id=user["user_id"]
    ).first()
    if participation:
        activity_id = participation.activity.id
        db.delete(participation)
        db.commit()
        return RedirectResponse(url=f"/ai/student/activities/{activity_id}/join-page", status_code=303)
    
    raise HTTPException(status_code=404, detail="Participation not found")

@ai_router.get("/student/activities/{activity_id}/work", response_class=HTMLResponse)
async def activity_work_page(
    request: Request,
    activity_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    # Check participation
    participation = db.query(ActivityParticipation).filter_by(
        activity_id=activity_id, student_id=user["user_id"]
    ).first()
    if not participation:
        return RedirectResponse(url=f"/ai/student/activities/{activity_id}/join-page")
    
    activity = participation.activity
    
    return templates.TemplateResponse("activity_work.html", {
        "request": request,
        "activity": activity,
        "participation": participation
    })

@ai_router.get("/student/activities/{activity_id}/peer-quiz/solve", response_class=HTMLResponse)
async def peer_quiz_solve_page(
    request: Request,
    activity_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    # Check participation
    participation = db.query(ActivityParticipation).filter_by(
        activity_id=activity_id, student_id=user["user_id"]
    ).first()
    if not participation:
        return RedirectResponse(url=f"/ai/student/activities/{activity_id}/join-page")
    
    activity = participation.activity
    
    # Get all groups with submitted quizzes
    submitted_quizzes = db.query(ActivityParticipation, ActivityGroup, User).join(
        ActivityGroup, ActivityParticipation.group_id == ActivityGroup.id, isouter=True
    ).join(User).filter(
        ActivityParticipation.activity_id == activity_id,
        ActivityParticipation.status == "submitted",
        ActivityParticipation.submission_data.isnot(None)
    ).all()
    
    # Get quiz attempts by current user/group
    user_attempts = db.query(QuizAttempt).filter_by(
        activity_id=activity_id,
        solver_id=user["user_id"]
    ).all()
    
    return templates.TemplateResponse("peer_quiz_solve.html", {
        "request": request,
        "activity": activity,
        "participation": participation,
        "submitted_quizzes": submitted_quizzes,
        "user_attempts": user_attempts
    })

# 2. Route to take a specific quiz
@ai_router.get("/student/activities/{activity_id}/quiz/{creator_participation_id}/take", response_class=HTMLResponse)
async def take_peer_quiz(
    request: Request,
    activity_id: int,
    creator_participation_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    # Check participation
    participation = db.query(ActivityParticipation).filter_by(
        activity_id=activity_id, student_id=user["user_id"]
    ).first()
    if not participation:
        raise HTTPException(status_code=404, detail="Not participating in activity")
    
    # Get the quiz to solve
    quiz_creator = db.query(ActivityParticipation).filter_by(
        id=creator_participation_id,
        activity_id=activity_id,
        status="submitted"
    ).first()
    if not quiz_creator or not quiz_creator.submission_data:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    # Check if already attempted
    existing_attempt = db.query(QuizAttempt).filter_by(
        activity_id=activity_id,
        creator_participation_id=creator_participation_id,
        solver_id=user["user_id"]
    ).first()
    
    return templates.TemplateResponse("take_peer_quiz.html", {
        "request": request,
        "activity": participation.activity,
        "participation": participation,
        "quiz_creator": quiz_creator,
        "quiz_data": quiz_creator.submission_data,
        "existing_attempt": existing_attempt
    })

# 3. Route to submit quiz answers
@ai_router.post("/student/activities/{activity_id}/quiz/{creator_participation_id}/submit")
async def submit_quiz_answers(
    request: Request,
    activity_id: int,
    creator_participation_id: int,
    answers: str = Form(...),  # JSON string of answers
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    participation = db.query(ActivityParticipation).filter_by(
        activity_id=activity_id, student_id=user["user_id"]
    ).first()
    if not participation:
        raise HTTPException(status_code=404, detail="Not participating in activity")
    
    # Get the original quiz
    quiz_creator = db.query(ActivityParticipation).filter_by(
        id=creator_participation_id,
        activity_id=activity_id,
        status="submitted"
    ).first()
    if not quiz_creator:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    try:
        answer_data = json.loads(answers)
        quiz_questions = quiz_creator.submission_data
        
        # Calculate score
        correct_count = 0
        total_questions = len(quiz_questions)
        
        for i, question in enumerate(quiz_questions):
            user_answer = answer_data.get(f"question_{i+1}")
            if user_answer and user_answer.lower() == question['correct'].lower():
                correct_count += 1
        
        score = (correct_count / total_questions) * 100 if total_questions > 0 else 0
        
        # Check if already attempted
        existing_attempt = db.query(QuizAttempt).filter_by(
            activity_id=activity_id,
            creator_participation_id=creator_participation_id,
            solver_id=user["user_id"]
        ).first()
        
        if existing_attempt:
            # Update existing attempt
            existing_attempt.answers = answer_data
            existing_attempt.score = score
            existing_attempt.correct_count = correct_count
            existing_attempt.total_questions = total_questions
            existing_attempt.completed_at = datetime.utcnow()
        else:
            # Create new attempt
            attempt = QuizAttempt(
                activity_id=activity_id,
                creator_participation_id=creator_participation_id,
                solver_id=user["user_id"],
                solver_group_id=participation.group_id,
                answers=answer_data,
                score=score,
                correct_count=correct_count,
                total_questions=total_questions,
                completed_at=datetime.utcnow()
            )
            db.add(attempt)
        
        db.commit()
        
        return RedirectResponse(
            url=f"/ai/student/activities/{activity_id}/quiz/{creator_participation_id}/results",
            status_code=303
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid answer format")
    except Exception as e:
        print(f"Error submitting quiz answers: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing quiz submission")

# 4. Route to view quiz results and add comments
@ai_router.get("/student/activities/{activity_id}/quiz/{creator_participation_id}/results", response_class=HTMLResponse)
async def quiz_results_page(
    request: Request,
    activity_id: int,
    creator_participation_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    participation = db.query(ActivityParticipation).filter_by(
        activity_id=activity_id, student_id=user["user_id"]
    ).first()
    if not participation:
        raise HTTPException(status_code=404, detail="Not participating in activity")
    
    # Get quiz attempt
    attempt = db.query(QuizAttempt).filter_by(
        activity_id=activity_id,
        creator_participation_id=creator_participation_id,
        solver_id=user["user_id"]
    ).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Quiz attempt not found")
    
    # Get quiz creator info
    quiz_creator = db.query(ActivityParticipation, User, ActivityGroup).join(
        User
    ).join(ActivityGroup, ActivityParticipation.group_id == ActivityGroup.id, isouter=True).filter(
        ActivityParticipation.id == creator_participation_id
    ).first()
    
    # Get comments for this quiz
    comments = db.query(QuizComment, User).join(User).filter(
        QuizComment.creator_participation_id == creator_participation_id
    ).order_by(QuizComment.created_at.desc()).all()
    
    return templates.TemplateResponse("quiz_results.html", {
        "request": request,
        "activity": participation.activity,
        "participation": participation,
        "attempt": attempt,
        "quiz_creator": quiz_creator,
        "quiz_data": quiz_creator.ActivityParticipation.submission_data,
        "comments": comments
    })

# 5. Route to add comments to quizzes
@ai_router.post("/student/activities/{activity_id}/quiz/{creator_participation_id}/comment")
async def add_quiz_comment(
    activity_id: int,
    creator_participation_id: int,
    comment_text: str = Form(...),
    comment_type: str = Form("general"),  # general, question_specific, improvement
    question_number: int = Form(None),
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    participation = db.query(ActivityParticipation).filter_by(
        activity_id=activity_id, student_id=user["user_id"]
    ).first()
    if not participation:
        raise HTTPException(status_code=404, detail="Not participating in activity")
    
    # Validate comment
    if not comment_text.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")
    
    comment = QuizComment(
        creator_participation_id=creator_participation_id,
        commenter_id=user["user_id"],
        commenter_group_id=participation.group_id,
        comment_text=comment_text.strip(),
        comment_type=comment_type,
        question_number=question_number,
        created_at=datetime.utcnow()
    )
    
    db.add(comment)
    db.commit()
    
    return RedirectResponse(
        url=f"/ai/student/activities/{activity_id}/quiz/{creator_participation_id}/results",
        status_code=303
    )

# 6. Route to view all quiz attempts for your created quiz
@ai_router.get("/student/activities/{activity_id}/my-quiz/attempts", response_class=HTMLResponse)
async def my_quiz_attempts(
    request: Request,
    activity_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    participation = db.query(ActivityParticipation).filter_by(
        activity_id=activity_id, 
        student_id=user["user_id"],
        status="submitted"
    ).first()
    if not participation:
        raise HTTPException(status_code=404, detail="No submitted quiz found")
    
    # Get all attempts on this user's quiz
    attempts = db.query(QuizAttempt, User, ActivityGroup).join(
        User, QuizAttempt.solver_id == User.id
    ).join(
        ActivityGroup, QuizAttempt.solver_group_id == ActivityGroup.id, isouter=True
    ).filter(
        QuizAttempt.creator_participation_id == participation.id
    ).order_by(QuizAttempt.completed_at.desc()).all()
    
    # Get comments on this quiz
    comments = db.query(QuizComment, User).join(User).filter(
        QuizComment.creator_participation_id == participation.id
    ).order_by(QuizComment.created_at.desc()).all()
    
    return templates.TemplateResponse("my_quiz_attempts.html", {
        "request": request,
        "activity": participation.activity,
        "participation": participation,
        "attempts": attempts,
        "comments": comments,
        "quiz_data": participation.submission_data
    })

@ai_router.get("/teacher/activities/{activity_id}/quiz-overview", response_class=HTMLResponse)
async def teacher_quiz_overview(
    request: Request,
    activity_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    activity = db.query(InClassActivity).filter_by(
        id=activity_id, 
        teacher_id=user["user_id"],
        activity_type="peer_quiz"
    ).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Quiz activity not found")
    
    # Get all submitted quizzes with group info
    submitted_quizzes = db.query(ActivityParticipation, User, ActivityGroup).join(
        User
    ).join(ActivityGroup, ActivityParticipation.group_id == ActivityGroup.id, isouter=True).filter(
        ActivityParticipation.activity_id == activity_id,
        ActivityParticipation.status == "submitted",
        ActivityParticipation.submission_data.isnot(None)
    ).all()
    
    # Get all quiz attempts
    all_attempts = db.query(QuizAttempt, User, ActivityGroup).join(
        User, QuizAttempt.solver_id == User.id
    ).join(
        ActivityGroup, QuizAttempt.solver_group_id == ActivityGroup.id, isouter=True
    ).filter(
        QuizAttempt.activity_id == activity_id
    ).all()
    
    # Get all comments
    all_comments = db.query(QuizComment, User, ActivityParticipation).join(
        User, QuizComment.commenter_id == User.id
    ).join(
        ActivityParticipation, QuizComment.creator_participation_id == ActivityParticipation.id
    ).filter(
        ActivityParticipation.activity_id == activity_id
    ).order_by(QuizComment.created_at.desc()).all()
    
    return templates.TemplateResponse("teacher_quiz_overview.html", {
        "request": request,
        "activity": activity,
        "submitted_quizzes": submitted_quizzes,
        "all_attempts": all_attempts,
        "all_comments": all_comments
    })

@ai_router.get("/teacher/activities/{activity_id}/quiz/{participation_id}/view", response_class=HTMLResponse)
async def teacher_view_quiz(
    request: Request,
    activity_id: int,
    participation_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    activity = db.query(InClassActivity).filter_by(
        id=activity_id, 
        teacher_id=user["user_id"]
    ).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    
    # Get the quiz participation
    quiz_participation = db.query(ActivityParticipation, User, ActivityGroup).join(
        User
    ).join(ActivityGroup, ActivityParticipation.group_id == ActivityGroup.id, isouter=True).filter(
        ActivityParticipation.id == participation_id,
        ActivityParticipation.activity_id == activity_id,
        ActivityParticipation.status == "submitted"
    ).first()
    
    if not quiz_participation:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    # Get all attempts on this quiz
    quiz_attempts = db.query(QuizAttempt, User, ActivityGroup).join(
        User, QuizAttempt.solver_id == User.id
    ).join(
        ActivityGroup, QuizAttempt.solver_group_id == ActivityGroup.id, isouter=True
    ).filter(
        QuizAttempt.creator_participation_id == participation_id
    ).order_by(QuizAttempt.completed_at.desc()).all()
    
    # Get comments on this quiz
    quiz_comments = db.query(QuizComment, User, ActivityGroup).join(
        User, QuizComment.commenter_id == User.id
    ).join(
        ActivityGroup, QuizComment.commenter_group_id == ActivityGroup.id, isouter=True
    ).filter(
        QuizComment.creator_participation_id == participation_id
    ).order_by(QuizComment.created_at.desc()).all()
    
    return templates.TemplateResponse("teacher_quiz_detail.html", {
        "request": request,
        "activity": activity,
        "quiz_participation": quiz_participation,
        "quiz_data": quiz_participation[0].submission_data,
        "quiz_attempts": quiz_attempts,
        "quiz_comments": quiz_comments
    })

@ai_router.get("/teacher/activities/{activity_id}/peer-analytics", response_class=HTMLResponse)
async def teacher_peer_analytics(
    request: Request,
    activity_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    activity = db.query(InClassActivity).filter_by(
        id=activity_id, 
        teacher_id=user["user_id"],
        activity_type="peer_quiz"
    ).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Quiz activity not found")
    
    # Get comprehensive analytics data
    submitted_quizzes = db.query(ActivityParticipation).filter_by(
        activity_id=activity_id,
        status="submitted"
    ).count()
    
    total_attempts = db.query(QuizAttempt).filter_by(activity_id=activity_id).count()
    total_comments = db.query(QuizComment).join(ActivityParticipation).filter(
        ActivityParticipation.activity_id == activity_id
    ).count()
    
    # Average scores by group
    group_performance = db.query(
        ActivityGroup.group_name,
        func.avg(QuizAttempt.score).label('avg_score'),
        func.count(QuizAttempt.id).label('attempt_count')
    ).join(
        QuizAttempt, ActivityGroup.id == QuizAttempt.solver_group_id
    ).filter(
        QuizAttempt.activity_id == activity_id
    ).group_by(ActivityGroup.id, ActivityGroup.group_name).all()
    
    # Most active commenters
    top_commenters = db.query(
        User.f_name,
        func.count(QuizComment.id).label('comment_count')
    ).join(
        QuizComment, User.id == QuizComment.commenter_id
    ).join(
        ActivityParticipation, QuizComment.creator_participation_id == ActivityParticipation.id
    ).filter(
        ActivityParticipation.activity_id == activity_id
    ).group_by(User.id, User.f_name).order_by(
        func.count(QuizComment.id).desc()
    ).limit(10).all()
    
    return templates.TemplateResponse("teacher_peer_analytics.html", {
        "request": request,
        "activity": activity,
        "submitted_quizzes": submitted_quizzes,
        "total_attempts": total_attempts,
        "total_comments": total_comments,
        "group_performance": group_performance,
        "top_commenters": top_commenters
    })

# Add this route to fetch recent quiz activity for the monitor page

@ai_router.get("/teacher/activities/{activity_id}/quiz-activity", response_class=HTMLResponse)
async def get_recent_quiz_activity(
    request: Request,
    activity_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    activity = db.query(InClassActivity).filter_by(
        id=activity_id, 
        teacher_id=user["user_id"]
    ).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    
    # Get recent quiz attempts (last 10)
    recent_attempts = db.query(QuizAttempt, User, ActivityParticipation, ActivityGroup).join(
        User, QuizAttempt.solver_id == User.id
    ).join(
        ActivityParticipation, QuizAttempt.creator_participation_id == ActivityParticipation.id
    ).join(
        User, ActivityParticipation.student_id == User.id
    ).join(
        ActivityGroup, QuizAttempt.solver_group_id == ActivityGroup.id, isouter=True
    ).filter(
        QuizAttempt.activity_id == activity_id
    ).order_by(QuizAttempt.completed_at.desc()).limit(10).all()
    
    # Get recent comments (last 10)
    recent_comments = db.query(QuizComment, User, ActivityParticipation, User).join(
        User, QuizComment.commenter_id == User.id
    ).join(
        ActivityParticipation, QuizComment.creator_participation_id == ActivityParticipation.id
    ).join(
        User, ActivityParticipation.student_id == User.id
    ).filter(
        ActivityParticipation.activity_id == activity_id
    ).order_by(QuizComment.created_at.desc()).limit(10).all()
    
    # Combine and sort by timestamp
    activity_feed = []
    
    for attempt, solver, creator_participation, creator_user, solver_group in recent_attempts:
        activity_feed.append({
            'type': 'attempt',
            'timestamp': attempt.completed_at,
            'solver_name': solver.f_name,
            'solver_group': solver_group.group_name if solver_group else None,
            'creator_name': creator_user.f_name,
            'creator_group': creator_participation.group.group_name if creator_participation.group else None,
            'score': attempt.score,
            'participation_id': creator_participation.id
        })
    
    for comment, commenter, creator_participation, creator_user in recent_comments:
        activity_feed.append({
            'type': 'comment',
            'timestamp': comment.created_at,
            'commenter_name': commenter.f_name,
            'creator_name': creator_user.f_name,
            'creator_group': creator_participation.group.group_name if creator_participation.group else None,
            'comment_type': comment.comment_type,
            'comment_text': comment.comment_text[:50] + "..." if len(comment.comment_text) > 50 else comment.comment_text,
            'participation_id': creator_participation.id
        })
    
    # Sort by timestamp (most recent first)
    activity_feed.sort(key=lambda x: x['timestamp'], reverse=True)
    activity_feed = activity_feed[:15]  # Keep only 15 most recent
    
    return templates.TemplateResponse("quiz_activity_feed.html", {
        "request": request,
        "activity": activity,
        "activity_feed": activity_feed
    })

# Add route to get AI feedback as JSON for modal display
@ai_router.get("/teacher/activities/{activity_id}/participation/{participation_id}/ai-feedback-json")
async def get_ai_feedback_json(
    activity_id: int,
    participation_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    participation = db.query(ActivityParticipation).filter_by(
        id=participation_id, 
        activity_id=activity_id
    ).first()
    
    if not participation or not participation.ai_feedback:
        return JSONResponse(content={"error": "No AI feedback available"}, status_code=404)
    
    feedback = participation.ai_feedback
    if isinstance(feedback, dict):
        feedback_text = feedback.get("feedback", "")
    else:
        feedback_text = str(feedback)
    
    return JSONResponse(content={
        "feedback": feedback_text,
        "student_name": participation.student.f_name if participation.student else "Unknown",
        "group_name": participation.group.group_name if participation.group else None
    })