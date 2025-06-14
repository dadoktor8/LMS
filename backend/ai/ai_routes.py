# backend/ai/ai_routes.py
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
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import shutil
import os
from datetime import date, datetime, time, timedelta

import urllib
from backend.ai.ai_grader import evaluate_assignment_text, prepare_rubric_for_ai
from backend.ai.aws_ai import S3_BUCKET_NAME, delete_file_from_s3, generate_presigned_url, generate_s3_download_link, get_s3_client, upload_file_to_s3
from backend.ai.open_notes_system import check_quiz_quota, generate_quiz_export, generate_study_material, generate_study_material_quiz, increment_quiz_quota, render_flashcards_htmx, render_quiz_htmx, render_study_guide_htmx
from backend.auth.routes import require_role
from backend.db.database import engine,get_db
from backend.db.models import Assignment, AssignmentComment, AssignmentSubmission, ChatHistory, Course,CourseMaterial, CourseUploadQuota, Enrollment, FlashcardUsage, PDFQuotaUsage, ProcessedMaterial,Quiz, RubricCriterion, RubricEvaluation, RubricLevel, StudentActivity, StudyGuideUsage, TextChunk  # Make sure this is correct
from backend.db.schemas import QueryRequest
from backend.utils.permissions import require_teacher_or_ta  # Optional if you want TA access too
from fastapi.templating import Jinja2Templates
from .text_processing import PDFQuotaConfig, extract_text_from_pdf, get_answer_from_rag_langchain_openai, get_context_for_query, get_course_retriever, get_openai_client, process_materials_in_background,sanitize_filename, validate_pdf_for_upload
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
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    try:
        student_id = str(user["user_id"])
        session_id = f"{student_id}_{course_id}"
        print(f"User ID: {user['user_id']}, Course ID: {course_id}")
        current_time = datetime.now()
        print(f"Current server time: {current_time}")
        # Check daily message limit PER COURSE (100 messages per day per course)
        today = current_time.date()
        today_start = datetime.combine(today, time.min)
        today_end = datetime.combine(today, time.max)
        print(f"Date range: {today_start} to {today_end}")

        all_messages = db.query(ChatHistory).filter(
            ChatHistory.user_id == user["user_id"],
            ChatHistory.course_id == course_id,
            ChatHistory.sender == "student"
        ).all()
        
        print(f"Total messages ever sent in this course: {len(all_messages)}")
        if all_messages:
            print(f"First message timestamp: {all_messages[0].timestamp}")
            print(f"Last message timestamp: {all_messages[-1].timestamp}")

        message_count = db.query(ChatHistory).filter(
            ChatHistory.user_id == user["user_id"],
            ChatHistory.course_id == course_id,  # Added course_id filter to restrict count to current course
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
                status_code=429,  # Too Many Requests
                headers={"HX-Trigger": "ai-response-complete"}
            )
        
        # Ensure materials exist
        course_materials = db.query(CourseMaterial).filter_by(course_id=course_id).all()
        if not course_materials:
            raise HTTPException(status_code=404, detail="No course materials found")
            
        processed_materials = db.query(ProcessedMaterial).filter_by(course_id=course_id).all()
        if not processed_materials:
            raise HTTPException(status_code=404, detail="Course materials haven't been processed yet")
            
        faiss_index_dir = f"faiss_index_{course_id}"
        if not os.path.exists(faiss_index_dir):
            raise HTTPException(status_code=404, detail="FAISS index not found")

            
        # Save student message to DB
        student_message = ChatHistory(
            user_id=user["user_id"], 
            course_id=course_id, 
            sender="student", 
            message=query,
            timestamp=current_time  # Explicitly set the current time
        )
        db.add(student_message)
        db.commit()
        
        # Get answer using LangChain RAG
        answer = get_answer_from_rag_langchain_openai(query, course_id, student_id)
        
        # Save AI response to DB
        ai_message = ChatHistory(
            user_id=user["user_id"], 
            course_id=course_id, 
            sender="ai", 
            message=answer,
            timestamp=datetime.now()  # Use fresh timestamp
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
                // Dispatch a custom event to signal that the AI response is complete
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
    materials = db.query(CourseMaterial).filter_by(course_id=course_id).order_by(CourseMaterial.uploaded_at.desc()).all()
    messages = db.query(ChatHistory).filter_by(user_id=user["user_id"], course_id=course_id).order_by(ChatHistory.timestamp).all()
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
    
    # Get remaining messages for today for THIS COURSE
    today = datetime.now().date()
    today_start = datetime.combine(today, time.min)
    today_end = datetime.combine(today, time.max)
    
    message_count = db.query(ChatHistory).filter(
        ChatHistory.user_id == user["user_id"],
        ChatHistory.course_id == course_id,  # Added course_id filter to count messages per course
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
        "materials": materials,
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
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    materials_json = request.session.get("flashcards_materials")
    study_material_html = render_flashcards_htmx(materials_json) if materials_json else ""
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
   
    return templates.TemplateResponse(
        "flashcards.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
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
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    materials_json = request.session.get("study_guide_materials")
    study_material_html = render_study_guide_htmx(materials_json) if materials_json else ""
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
    
    return templates.TemplateResponse(
        "study_guide.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "study_material_html": study_material_html,
            "student_id": request.session.get("student_id", ""),
            "courses":courses
        }
    )

# Separate endpoints for generating each type of material
@ai_router.post("/study/flashcards/generate", response_class=HTMLResponse)
async def generate_flashcards(
    request: Request,
    topic: str = Form(...),
    course_id: int = Form(...),
    student_id: str = Form(...),
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
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
        
        return templates.TemplateResponse(
            "flashcards.html",
            {
                "request": request,
                "course_id": course_id,
                "topic": topic,
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
    
    # Generate the flashcards
    materials_json = generate_study_material(topic, "flashcards", course_id, student_id)
    request.session["flashcards_materials"] = materials_json
    study_material_html = render_flashcards_htmx(materials_json)
    
    # Get courses for the sidebar
    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
   
    return templates.TemplateResponse(
        "flashcards.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
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
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
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
        
        return templates.TemplateResponse(
            "study_guide.html",
            {
                "request": request,
                "course_id": course_id,
                "topic": topic,
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
    materials_json = generate_study_material(topic, "study_guide", course_id, student_id)
    request.session["study_guide_materials"] = materials_json
    study_material_html = render_study_guide_htmx(materials_json)

    enrollments = db.query(Enrollment).filter_by(student_id=user["user_id"], is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
    
    return templates.TemplateResponse(
        "study_guide.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
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
    materials = db.query(CourseMaterial).filter(CourseMaterial.course_id == course_id).order_by(CourseMaterial.uploaded_at.desc()).all()
    course = db.query(Course).filter(Course.id == course_id).first()
    teacher_id = user["user_id"]
    courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)
    return templates.TemplateResponse("create_assignment.html", {"request": request, "course_id": course_id, "course":course, "materials": materials, "courses": courses})

@ai_router.post("/courses/{course_id}/create-assignment", response_class=HTMLResponse)
async def create_assignment(
    request: Request,
    course_id: int,
    title: str = Form(...),
    description: str = Form(...),
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

    deadline_dt = datetime.fromisoformat(deadline)
    assignment = Assignment(
        course_id=course_id,
        title=title,
        description=description,
        deadline=deadline_dt,
        teacher_id=teacher_id
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
    course = db.query(Course).filter(Course.id == course_id).first()
    teacher_id = user["user_id"]
    courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)
    return templates.TemplateResponse("teacher_assignments.html", {
        "request": request,
        "assignments": assignments,
        "course": course,
        "courses": courses
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
    
    return templates.TemplateResponse("edit_assignment.html", {
        "request": request,
        "assignment": assignment,
        "course": course,
        "courses": courses
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
    
    if not assignment:
        return HTMLResponse("❌ Assignment not found", status_code=404)
    
    # Ensure the teacher owns this assignment
    if assignment.teacher_id != user["user_id"]:
        return HTMLResponse("❌ You don't have permission to edit this assignment", status_code=403)
    
    # Update assignment details
    assignment.title = title
    assignment.description = description
    
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
    db : Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
):
    teacher_id = user["user_id"]
    courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()
    
    # Check quota and get remaining count
    quota_exceeded, remaining = check_quiz_quota(db, teacher_id, course_id)
    
    # Render form to create a quiz for this specific course
    return templates.TemplateResponse(
        "quiz_creator.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": "",
            "study_material_html": "",
            "teacher_id": user.get("user_id", ""),
            "question_types": ["mcq"],  # Default
            "num_questions": 10,
            "courses": courses,
            "remaining_quota": remaining,
            "quota_exceeded": quota_exceeded
        }
    )

@ai_router.post("/courses/{course_id}/quiz/generate", response_class=HTMLResponse)
async def generate_quiz(
    request: Request,
    course_id: int,
    topic: str = Form(...),
    question_types: List[str] = Form(...),
    num_questions: int = Form(10),
    user: dict = Depends(require_role("teacher")),
    db: Session = Depends(get_db),
):
    teacher_id = user.get("user_id", "")
    MAX_QUESTIONS = 20
    if num_questions > MAX_QUESTIONS:
        num_questions = MAX_QUESTIONS
    
    try:
        # Check if quota is exceeded
        quota_exceeded, remaining = check_quiz_quota(db, teacher_id, course_id)
        
        if quota_exceeded:
            return templates.TemplateResponse(
                "quiz_creator.html",
                {
                    "request": request,
                    "course_id": course_id,
                    "topic": topic,
                    "error": "Daily quiz creation limit reached (5 quizzes per course). Please try again tomorrow.",
                    "teacher_id": teacher_id,
                    "question_types": question_types,
                    "num_questions": num_questions,
                    "remaining_quota": 0,
                    "quota_exceeded": True,
                    "courses": db.query(Course).filter(Course.teacher_id == teacher_id).all()
                }
            )
            
        # Generate the quiz materials
        materials_json = generate_study_material_quiz(
            query=topic,
            material_type="quiz",
            course_id=course_id,
            teacher_id=teacher_id,
            question_types=question_types,
            num_questions=num_questions
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
                    json_data=materials_json
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
                {
                    "request": request,
                    "course_id": course_id,
                    "topic": topic,
                    "error": f"Database error: {db_exc}",
                    "teacher_id": teacher_id,
                    "courses": db.query(Course).filter(Course.teacher_id == teacher_id).all(),
                    "remaining_quota": remaining
                }
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
                "quota_success": True,
                "courses": db.query(Course).filter(Course.teacher_id == teacher_id).all()
            }
        )
    except Exception as e:
        print("Main error:", e)
        error_message = f"An unexpected error occurred: {str(e)}"
        return templates.TemplateResponse(
            "quiz_creator.html",
            {
                "request": request,
                "course_id": course_id,
                "topic": topic if 'topic' in locals() else "",
                "error": error_message,
                "teacher_id": teacher_id,
                "courses": db.query(Course).filter(Course.teacher_id == teacher_id).all()
            }
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
    
    # Calculate how many activities were done today
    today = datetime.utcnow().date()
    activities_today = sum(1 for activity in student_activities if activity.created_at.date() == today)
    daily_limit_reached = activities_today >= 5
    
    return templates.TemplateResponse("student_engagement.html", {
        "request": request,
        "course": course,
        "courses": courses,
        "student_activities": student_activities,
        "activities_today": activities_today,
        "daily_limit": 5,
        "daily_limit_reached": daily_limit_reached
    })

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
@ai_router.post("/student/courses/{course_id}/muddiest-point", response_class=HTMLResponse)
async def process_muddiest_point(
    request: Request,
    course_id: int,
    topic: str = Form(...),
    confusion: str = Form(...),
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    enrollment = db.query(Enrollment).filter_by(
        student_id=user["user_id"], course_id=course_id, is_accepted=True
    ).first()
    if not enrollment:
        return JSONResponse(content={"error": "You are not enrolled in this course"}, status_code=403)
    
    # Check daily activity limit
    activities_today, limit_reached = check_daily_activity_limit(db, user["user_id"], course_id)
    if limit_reached:
        return JSONResponse(
            content={"error": f"Daily limit of 5 engagement activities per course reached. Please try again tomorrow."},
            status_code=429  # Too Many Requests
        )
    
    query = f"Topic: {topic}. Confusion: {confusion}"
    try:
        retriever = get_course_retriever(course_id)
        context = get_context_for_query(retriever, query)
        chat = get_openai_client()
        
        system_prompt = f"""
You are an educational AI tutor analyzing a student's confusion about a topic. Use the provided course materials to:
1. Identify the core confusion areas
2. Suggest specific review topics that would help clarify understanding
3. Provide a list of key resources or concepts to focus on
Base your analysis ONLY on the context provided.
CONTEXT FROM COURSE MATERIALS: {context}
        """
        user_message = f"""
Topic: {topic}
Student's confusion: {confusion}
Analyze this confusion, identify the core confusion areas, and provide suggestions for review.
Return your response as valid JSON with the following structure:
{{ "summary": "...", "confusion_areas": [ ... ], "review_topics": [ ... ], "resources": [ ... ] }}
        """

        # Modern Langchain/OpenAI expects a message list for chat models
        response = chat.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ])
        
        # response.content if using Langchain's OpenAI chat model (extract string)
        if hasattr(response, "content"):
            ai_text = response.content
        else:
            ai_text = response  # fallback
        
        try:
            ai_response = json.loads(ai_text)
        except Exception as e:
            return JSONResponse(content={"error": "Failed to parse AI response: " + str(e)}, status_code=500)

        # Save activity
        new_activity = StudentActivity(
            student_id=user["user_id"],
            course_id=course_id,
            activity_type="Muddiest Point",
            topic=topic,
            user_input=confusion,
            ai_response=json.dumps(ai_response),
            created_at=datetime.utcnow()
        )
        db.add(new_activity)
        db.commit()

        # Render response using your template/component
        return templates.TemplateResponse("muddiest_point_response.html", {
            "request": request,
            "ai_response": ai_response,
            "activities_today": activities_today + 1,
            "daily_limit": 5
        })
    except HTTPException as e:
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Misconception Check endpoint
@ai_router.post("/student/courses/{course_id}/misconception-check", response_class=HTMLResponse)
async def process_misconception_check(
    request: Request,
    course_id: int,
    topic: str = Form(...),
    beliefs: str = Form(...),
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("student"))
):
    enrollment = db.query(Enrollment).filter_by(
        student_id=user["user_id"], course_id=course_id, is_accepted=True
    ).first()
    if not enrollment:
        return JSONResponse(content={"error": "You are not enrolled in this course"}, status_code=403)
    
    # Check daily activity limit
    activities_today, limit_reached = check_daily_activity_limit(db, user["user_id"], course_id)
    if limit_reached:
        return JSONResponse(
            content={"error": f"Daily limit of 5 engagement activities per course reached. Please try again tomorrow."},
            status_code=429  # Too Many Requests
        )
    
    query = f"Topic: {topic}. Student beliefs: {beliefs}"
    try:
        retriever = get_course_retriever(course_id)
        context = get_context_for_query(retriever, query)
        chat = get_openai_client()

        system_prompt = f"""
You are an educational AI tutor analyzing a student's beliefs or understanding about a topic. Use the provided course materials to:
1. Compare the student's stated beliefs with accurate information
2. Identify any misconceptions
3. Provide helpful corrections and explanations
4. Suggest resources for further learning
Base your analysis ONLY on the context provided.
CONTEXT FROM COURSE MATERIALS: {context}
        """
        user_message = f"""
Topic: {topic}
Student's beliefs: {beliefs}
Analyze these beliefs, identify what's accurate and what may be misconceptions.
Return your response as valid JSON with the following structure:
{{ "summary": "...", "beliefs": [{{ "statement": "...", "is_accurate": true/false, "explanation": "..." }}], "resources": [ ... ] }}
        """

        # UPDATED: Use .invoke with a message list!
        response = chat.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ])
        # For langchain.chat_models.base.BaseChatModel, .content contains the response text.
        ai_text = getattr(response, "content", response)
        try:
            ai_response = json.loads(ai_text)
        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to parse AI response: {e}\nRaw response: {ai_text}"},
                status_code=500
            )
        new_activity = StudentActivity(
            student_id=user["user_id"],
            course_id=course_id,
            activity_type="Misconception Check",
            topic=topic,
            user_input=beliefs,
            ai_response=json.dumps(ai_response),
            created_at=datetime.utcnow()
        )
        db.add(new_activity)
        db.commit()
        return templates.TemplateResponse("misconception_response.html", {
            "request": request,
            "ai_response": ai_response,
            "activities_today": activities_today + 1,
            "daily_limit": 5
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