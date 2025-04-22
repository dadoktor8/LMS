# backend/ai/ai_routes.py
import html
import json
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Depends, Request, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import shutil
import os
from datetime import datetime

import urllib
from backend.ai.ai_grader import evaluate_assignment_text
from backend.ai.open_notes_system import generate_study_material, render_flashcards_htmx, render_quiz_htmx, render_study_guide_htmx
from backend.auth.routes import require_role
from backend.db.database import engine,get_db
from backend.db.models import Assignment, AssignmentComment, AssignmentSubmission, ChatHistory, Course,CourseMaterial, Enrollment, ProcessedMaterial  # Make sure this is correct
from backend.db.schemas import QueryRequest
from backend.utils.permissions import require_teacher_or_ta  # Optional if you want TA access too
from fastapi.templating import Jinja2Templates
from .text_processing import extract_text_from_pdf, chunk_text, embed_chunks, get_answer_from_rag_langchain_openai, process_materials_in_background, save_embeddings_to_faiss,sanitize_filename
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

        # Ensure materials exist
        course_materials = db.query(CourseMaterial).filter_by(course_id=course_id).all()
        if not course_materials:
            raise HTTPException(status_code=404, detail="No course materials found")

        processed_materials = db.query(ProcessedMaterial).filter_by(course_id=course_id).all()
        if not processed_materials:
            raise HTTPException(status_code=404, detail="Course materials haven't been processed yet")

        faiss_index_path = f"faiss_index_{course_id}.index"
        if not os.path.exists(faiss_index_path):
            raise HTTPException(status_code=404, detail="FAISS index not found")
        db.add(ChatHistory(user_id=user["user_id"], course_id=course_id, sender="student", message=query))
        # Get answer using LangChain RAG
        answer = get_answer_from_rag_langchain_openai(query, course_id, student_id)

        db.add(ChatHistory(user_id=user["user_id"], course_id=course_id, sender="ai", message=answer))
        db.commit()
        safe_query = html.escape(query)
        # Save to chat history
        history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string="sqlite:///chat_history.db"
        )
        history.add_user_message(query)
        history.add_ai_message(answer)

        return HTMLResponse(content=f"""
        <div class="chat-bubble bg-blue-600 text-white px-6 py-4 rounded-2xl self-end max-w-2xl ml-auto shadow text-lg font-semibold whitespace-pre-line">
            🧑‍🎓 {safe_query}
        </div>
        <div class="chat-bubble bg-indigo-50 border border-indigo-200 text-indigo-900 px-6 py-4 rounded-2xl self-start max-w-2xl shadow text-lg font-medium leading-relaxed whitespace-pre-line">
            💡 {answer}
        </div>
        """, status_code=200)
        
    except HTTPException as e:
        logging.error(f"HTTP Exception: {e.detail}")
        return HTMLResponse(content=f"<div class='toast error'>❌ {e.detail}</div>", status_code=e.status_code)
    except Exception as e:
        import traceback
        logging.error(f"Error in ask_tutor: {str(e)}")
        logging.error(traceback.format_exc())
        return HTMLResponse(content=f"<div class='toast error'>❌ Something went wrong. Please try again later.</div>", status_code=500)


@ai_router.get("/courses/{course_id}/tutor", response_class=HTMLResponse)
async def show_student_tutor(request: Request, course_id: int, db: Session = Depends(get_db), user=Depends(require_role("student"))):
    course = db.query(Course).filter(Course.id == course_id).first()
    materials = db.query(CourseMaterial).filter_by(course_id=course_id).order_by(CourseMaterial.uploaded_at.desc()).all()
    messages = db.query(ChatHistory).filter_by(user_id=user["user_id"], course_id=course_id).order_by(ChatHistory.timestamp).all()

    return templates.TemplateResponse("student_ai_tutor.html", {
        "request": request,
        "course": course,
        "materials": materials,
        "messages": messages
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
):
    return templates.TemplateResponse(
        "study_landing.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "student_id": request.session.get("student_id", "")
        }
    )

# Flashcards specific page
@ai_router.get("/study/flashcards", response_class=HTMLResponse)
async def flashcards_page(
    request: Request,
    course_id: int = Query(...),
    topic: Optional[str] = Query(None),
):
    materials_json = request.session.get("flashcards_materials")
    study_material_html = render_flashcards_htmx(materials_json) if materials_json else ""
    
    return templates.TemplateResponse(
        "flashcards.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "study_material_html": study_material_html,
            "student_id": request.session.get("student_id", "")
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
):
    materials_json = request.session.get("study_guide_materials")
    study_material_html = render_study_guide_htmx(materials_json) if materials_json else ""
    
    return templates.TemplateResponse(
        "study_guide.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "study_material_html": study_material_html,
            "student_id": request.session.get("student_id", "")
        }
    )

# Separate endpoints for generating each type of material
@ai_router.post("/study/flashcards/generate", response_class=HTMLResponse)
async def generate_flashcards(
    request: Request,
    topic: str = Form(...),
    course_id: int = Form(...),
    student_id: str = Form(...)
):
    materials_json = generate_study_material(topic, "flashcards", course_id, student_id)
    request.session["flashcards_materials"] = materials_json
    study_material_html = render_flashcards_htmx(materials_json)
    
    return templates.TemplateResponse(
        "flashcards.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "study_material_html": study_material_html,
            "student_id": student_id,
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
    student_id: str = Form(...)
):
    materials_json = generate_study_material(topic, "study_guide", course_id, student_id)
    request.session["study_guide_materials"] = materials_json
    study_material_html = render_study_guide_htmx(materials_json)
    
    return templates.TemplateResponse(
        "study_guide.html",
        {
            "request": request,
            "course_id": course_id,
            "topic": topic,
            "study_material_html": study_material_html,
            "student_id": student_id,
        }
    )

@ai_router.get("/courses/{course_id}/create-assignment", response_class=HTMLResponse)
async def show_create_assignment_form(request: Request, course_id: int, db: Session = Depends(get_db)):
    materials = db.query(CourseMaterial).filter(CourseMaterial.course_id == course_id).order_by(CourseMaterial.uploaded_at.desc()).all()
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)
    return templates.TemplateResponse("create_assignment.html", {"request": request, "course_id": course_id, "course":course, "materials": materials})

@ai_router.post("/courses/{course_id}/create-assignment", response_class=HTMLResponse)
async def create_assignment(
    request: Request,
    course_id: int,
    title: str = Form(...),
    description: str = Form(...),
    deadline: str = Form(...),
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta()),
    material_ids: List[int] = Form(None)
):
    deadline_dt = datetime.fromisoformat(deadline)
    assignment = Assignment(
        course_id=course_id,
        title=title,
        description=description,
        deadline=deadline_dt,
        teacher_id=user["user_id"]
    )
    if material_ids:
        assignment.materials = db.query(CourseMaterial).filter(CourseMaterial.id.in_(material_ids)).all()
    db.add(assignment)
    db.commit()
    return RedirectResponse(f"/ai/teacher/{course_id}/assignments", status_code=303)

@ai_router.get("/assignments/{assignment_id}/submit", response_class=HTMLResponse)
async def show_submit_assignment_form(request: Request, assignment_id: int, db: Session = Depends(get_db)):
    assignment = db.query(Assignment).filter_by(id=assignment_id).first()
    return templates.TemplateResponse("submit_assignment.html", {"request": request, "assignment": assignment})

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
    filename = f"{datetime.utcnow().timestamp()}_{file.filename}"
    file_location = f"uploads/assignments/{assignment_id}_{filename}"  # no leading slash
    save_path = f"backend/{file_location}"  # actual filesystem path

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the file
    with open(save_path, "wb") as f:
        f.write(await file.read())

    
    # Extract and evaluate
    try:
        raw_text = extract_text_from_pdf(f"backend/{file_location}")
        ai_score, ai_feedback = evaluate_assignment_text(
            raw_text, 
            assignment.title, 
            assignment.description
        )
    except Exception as e:
        logging.error(f"AI Evaluation failed: {e}")
        ai_score, ai_feedback = None, None
    
    # Check if a previous submission exists
    existing_submission = (
        db.query(AssignmentSubmission)
        .filter_by(assignment_id=assignment_id, student_id=student_id)
        .first()
    )
    
    if existing_submission:
        # If resubmission, update the existing record
        existing_submission.file_path = file_location
        existing_submission.ai_score = ai_score
        existing_submission.ai_feedback = ai_feedback
        existing_submission.submitted_at = datetime.utcnow()
    else:
        # Create new submission
        submission = AssignmentSubmission(
            assignment_id=assignment_id,
            student_id=student_id,
            file_path=save_path,
            ai_score=ai_score,
            ai_feedback=ai_feedback,
            submitted_at=datetime.utcnow()
        )
        db.add(submission)
    
    db.commit()
    
    return HTMLResponse(
        f"""<div class='toast success'>
                ✅ Submitted! AI Score: {ai_score or "Pending"}
                <br>
                <small>{ai_feedback or "AI feedback pending"}</small>
            </div>""",
        status_code=200
    )

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
    
    return templates.TemplateResponse(
        "student_assignments.html",
        {
            "request": request,
            "assignments": assignments,
            "submissions": submissions,
            "submission_dict": submission_dict,
            "course": course,
        },
    )
@ai_router.post("/assignments/{assignment_id}/grade/{submission_id}")
async def grade_submission(
    assignment_id: int,
    submission_id: int,
    teacher_score: Optional[int] = Form(None),
    comment: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("teacher"))
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
    
    # Add a comment if provided
    if comment and comment.strip():
        # Log the comment being added
        logging.info(f"Adding comment to submission {submission_id}: '{comment}' by user {user['user_id']}")
        
        new_comment = AssignmentComment(
            submission_id=submission_id,
            user_id=user["user_id"],
            message=comment
        )
        db.add(new_comment)
    
    db.commit()
    
    # Verify comment was added
    comments = db.query(AssignmentComment).filter_by(submission_id=submission_id).all()
    logging.info(f"After commit, submission {submission_id} has {len(comments)} comments")
    
    # Use the correct prefix for your routes
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

    return templates.TemplateResponse("teacher_assignment_submissions.html", {
        "request": request,
        "assignment": assignment,
        "submissions": submissions
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
    return templates.TemplateResponse("student_assignments.html", {
        "request": request,
        "assignments": assignments,
        "submissions": submissions
    })


@ai_router.get("/teacher/{course_id}/assignments", response_class=HTMLResponse)
async def teacher_assignments(request: Request, course_id: int,db: Session = Depends(get_db), user: dict = Depends(require_role("teacher"))):
    assignments = (
    db.query(Assignment)
    .filter_by(teacher_id=user["user_id"], course_id=course_id)
    .order_by(Assignment.deadline.desc())
    .all()
        )
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)
    return templates.TemplateResponse("teacher_assignments.html", {
        "request": request,
        "assignments": assignments,
        "course":course
    })

