from collections import defaultdict
import csv
import io
import random
import re
import string
from fastapi import APIRouter, HTTPException, Depends, Query, Request, Form
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import UploadFile, File 
import pandas as pd
from pydantic import BaseModel, EmailStr, validator
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
from typing import List, Literal, Optional
import os
from dotenv import load_dotenv
from sqlalchemy import desc, func, text
from sqlalchemy.orm import Session
import yagmail
from backend.utils.permissions import require_teacher_or_ta
from backend.utils.tokens import ALGORITHM, PASSWORD_PATTERN, create_access_token, decode_token, generate_verification_token
from backend.utils.tokens import confirm_token
from backend.utils.email_utils import send_verification_email
from backend.db.models import AttendanceCode, AttendanceRecord, TeachingAssistant, User,Course,Enrollment,CourseInvite
from backend.db.database import engine,get_db

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
auth_router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")
print(engine.url)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    confirm_password: str
    role: str
    f_name: str
    l_name: str = ""
    
    @validator('password')
    def password_strength(cls, v):
        pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*(),.?":{}|<>]).{8,}$')
        if not pattern.match(v):
            raise ValueError(
                "Password must be at least 8 characters and include at least "
                "one lowercase letter, one uppercase letter, one number, and one special character"
            )
        return v
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def is_valid_password(password: str) -> bool:
    """
    Validate password meets complexity requirements:
    - At least 8 characters
    - At least one lowercase letter
    - At least one uppercase letter
    - At least one number
    - At least one special character
    """
    return bool(PASSWORD_PATTERN.match(password))

def get_base_url(request: Request = None) -> str:
    """
    Get the base URL for the application - handles local development vs production
    """
    # First priority: Use environment variable if set
    base_url = os.getenv('FRONTEND_URL')
    
    # Second priority: Use request host if available
    if not base_url and request:
        host = request.headers.get('host', '')
        scheme = request.headers.get('x-forwarded-proto', 'http')
        if not scheme or scheme == 'null':
            scheme = 'https' if request.url.scheme == 'https' else 'http'
        base_url = f"{scheme}://{host}"
    
    # Fallback to localhost if nothing else works
    if not base_url:
        base_url = "http://127.0.0.1:8000"
        
    return base_url

def require_role(required_role: str):
    def role_checker(request: Request):
        token = request.cookies.get("access_token")
        if not token:
            raise HTTPException(status_code=401, detail="Missing token in cookies")

        payload = decode_token(token)  # Your decode_token logic
        if payload.get("role") != required_role:
            raise HTTPException(status_code=403, detail="Not authorized")
        return payload  # Contains user_id, role, etc.

    return role_checker

@auth_router.post("/signup")
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == payload.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Verify password strength
    if not is_valid_password(payload.password):
        raise HTTPException(
            status_code=400, 
            detail="Password must be at least 8 characters with at least one lowercase letter, "
                  "one uppercase letter, one number, and one special character"
        )
    
    # Verify passwords match
    if payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    
    new_user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=payload.role,
        f_name=payload.f_name,
        l_name=payload.l_name,
        is_verified=False  # Default to unverified
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Send verification email
    token = generate_verification_token(payload.email)
    base_url = get_base_url()
    verification_link = f"{base_url}/auth/verify-email?token={token}"
    send_verification_email(payload.email, verification_link)
    
    return {"msg": "User created successfully. Please check your email or spam to verify your account."}

# === HTMX SIGNUP FORM ===
@auth_router.get("/signup-page", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@auth_router.post("/signup-form", response_class=HTMLResponse)
def signup_form(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    role: str = Form(...),
    f_name: str = Form(...),
    l_name: str = Form(...),
    db: Session = Depends(get_db)
):
    # Helper for error block (reuse this for all errors)
    def error_block(message):
        return HTMLResponse(
            content=f"""
<div class="bg-red-100 border border-red-400 text-red-700 rounded px-4 py-3 mb-4 flex justify-between items-center">
  <div>
    <strong class="font-bold">Error: </strong> {message}
  </div>
  <button onclick="window.location.reload()" class="bg-red-200 text-red-800 font-semibold px-2 py-1 ml-4 rounded hover:bg-red-300">Refresh</button>
</div>
""",
            status_code=200
        )

    # Check if user already exists
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        return error_block("A user with this email already exists.")

    # Validate password strength
    if not is_valid_password(password):
        return error_block(
            "Password must be at least 8 characters with at least one lowercase letter, "
            "one uppercase letter, one number, and one special character."
        )

    # Validate password match
    if password != confirm_password:
        return error_block("Passwords do not match.")

    # ... proceed with user creation
    new_user = User(
        f_name=f_name,
        l_name=l_name,
        email=email,
        hashed_password=hash_password(password),
        role=role,
        is_verified=False
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Send verification email
    token = generate_verification_token(email)
    base_url = get_base_url(request)
    link = f"{base_url}/auth/verify-email?token={token}"
    send_verification_email(email, link)

    # Show success message as a toast/banner
    return HTMLResponse(
        content="""
<div class="bg-green-100 border border-green-400 text-green-800 rounded px-4 py-3 mb-4 flex justify-between items-center">
  <div>
    <strong class="font-bold">Success! </strong>
    User created successfully. Please check your email to verify your account. Please check spam if not found in email!
  </div>
</div>
""",
        status_code=200
    )

# === JSON API LOGIN ===
@auth_router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    
    # Check if user exists and password is correct
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check if email is verified
    if not user.is_verified:
        raise HTTPException(
            status_code=401, 
            detail="Email not verified. Please check your inbox for the verification link."
        )
    
    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


# === HTMX LOGIN FORM ===
@auth_router.get("/login-page", response_class=HTMLResponse)
def login_page(request: Request, msg: str = ""):
    logged_out = request.cookies.get("logged_out")
    response = templates.TemplateResponse("login.html", {
        "request": request,
        "logged_out": logged_out,
        "msg": msg
    })
    # Clear the cookie after reading so it doesn't show again next time
    response.delete_cookie("logged_out")
    return response


@auth_router.post("/login-form", response_class=HTMLResponse)
def login_form(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email).first()
    
    # Check if user exists and password is correct
    if not user or not verify_password(password, user.hashed_password):
        return HTMLResponse(
            content='<div class="toast error" style="text-align:center;">❌ Invalid email or password</div>',
            status_code=200
        )
    
    # Check if email is verified
    if not user.is_verified:
        # Generate new verification token
        token = generate_verification_token(email)
        base_url = get_base_url(request)
        link = f"{base_url}/auth/verify-email?token={token}"
        send_verification_email(email, link)
        
        return HTMLResponse(
            content='<div class="toast warning" style="text-align:center;">⚠️ Email not verified. A new verification link has been sent to your email.</div>',
            status_code=200
        )
    
    # Determine redirect URL based on role
    if user.role == "admin":
        redirect_url = "/auth/admin/dashboard"
    elif user.role == "student":
        redirect_url = "/auth/student/courses"
    elif user.role == "teacher":
        redirect_url = "/auth/teacher/dashboard"
    else:
        return HTMLResponse(content="Invalid role", status_code=400)
    
    # Create JWT token with user_id included
    access_token = create_access_token(
        data={
            "sub": user.email,
            "role": user.role,
            "user_id": user.id
        }, 
        expires_delta=timedelta(hours=24)
    )
    
    # Set token in cookie and redirect
    response = HTMLResponse(
        content='<div class="toast success" style="text-align:center;">✅ Login successful! Redirecting...</div>',
        status_code=200
    )
    response.headers["HX-Redirect"] = redirect_url
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=60 * 60 * 24,  # 1 day
        secure=False,  # Set to True in production with HTTPS
        samesite="Lax"
    )
    
    return response

@auth_router.get("/student/profile", response_class=HTMLResponse)
async def student_profile(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_role("student"))):
    if current_user["role"] != "student":
        return RedirectResponse(url="/auth/dashboard")
    
    student_id = current_user["user_id"]
    student = db.query(User).filter_by(id=student_id).first()
    enrollments = db.query(Enrollment).filter_by(student_id=student_id, is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]
    return templates.TemplateResponse(
        "student_profile.html",
        {
            "request": request,
            "user": current_user,
            "courses": courses,
            "student":student,
        }
    )

# Step 4: Add route to update profile
@auth_router.post("/student/update-profile", response_class=HTMLResponse)
async def update_student_profile(
    request: Request,
    f_name: str = Form(...),
    l_name: str = Form(None),
    roll_number: str = Form(None),
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    if user["role"] != "student":
        return HTMLResponse(content="❌ You don't have permission to update this profile", status_code=403)
    
    user_obj = db.query(User).filter_by(id=user["user_id"]).first()
    if not user_obj:
        return HTMLResponse(content="❌ User not found", status_code=404)

    user_obj.f_name = f_name
    user_obj.l_name = l_name
    user_obj.roll_number = roll_number

    db.commit()
    db.refresh(user_obj)
    
    # Return success message
    return HTMLResponse(
        content="""
        <div class="p-4 bg-green-50 border-l-4 border-green-500 text-green-700 rounded">
            ✅ Profile updated successfully!
        </div>
        """
    )

@auth_router.get("/logout")
def logout():
    response = RedirectResponse(url="/auth/login-page")
    response.delete_cookie("access_token")
    response.set_cookie(key="logged_out", value="1", max_age=5)  # Expires in 5 seconds
    return response


# === USER INFO + ROLE ROUTES ===
@auth_router.get("/me")
def read_users_me(token: str = Depends(oauth2_scheme)):
    user_data = decode_token(token)
    return {"email": user_data.get("sub"), "role": user_data.get("role")}

@auth_router.get("/admin/dashboard")
def admin_dashboard(user=Depends(require_role("admin"))):
    return {"message": "Welcome Admin!", "email": user.get("sub")}


@auth_router.get("/student/courses", response_class=HTMLResponse)
def student_courses(
    request: Request,
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    student_id = user["user_id"]
    student = db.query(User).filter_by(id=user["user_id"]).first()


    # ✅ Only pending invites
    invites = db.query(CourseInvite).filter_by(student_id=student_id, status="pending").all()
    ta_invites = db.query(TeachingAssistant).filter_by(student_id=student_id, status="pending").all()

    # ✅ Enrolled courses
    enrollments = db.query(Enrollment).filter_by(student_id=student_id, is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]

    # ✅ Build invite links for pending ones
    invite_links = []
    for invite in invites:
        token = generate_verification_token(invite.student.email)
        link = f"/auth/accept-invite?token={token}&course_id={invite.course_id}"
        invite_links.append({
            "course": invite.course,
            "link": link,
            "invite_id": invite.id,  # useful if you later use AJAX/HTMX to accept without reloading
            "token":token
        })

    ta_invite_links = [
    {
        "course": ta.course,
        "link": f"/auth/accept-ta-invite?course_id={ta.course_id}"
    }
    for ta in ta_invites
]
    ta_courses = db.query(TeachingAssistant).filter_by(student_id=student_id, status="accepted").all()
    ta_course_list = [ta.course for ta in ta_courses]

    return templates.TemplateResponse("student_dashboard.html", {
        "request": request,
        "user": student,
        "courses": courses,
        "pending_invites": invite_links,
        "ta_invites": ta_invite_links,
        "ta_courses": ta_course_list
    })





@auth_router.get("/users", response_class=HTMLResponse)
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    html = "<h2>Registered Users</h2><ul>"
    for user in users:
        html += f"<li><strong>{user.email}</strong> - Role: {user.role} - Password: {user.hashed_password} </li>"
    html += "</ul>"
    return html


@auth_router.get("/verify-email", response_class=HTMLResponse)
def verify_email(request: Request, token: str, db: Session = Depends(get_db)):
    email = confirm_token(token)
    if not email:
        return templates.TemplateResponse("message.html", {
            "request": request,
            "title": "Verification Failed",
            "message": "❌ Invalid or expired token."
        })
        
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return templates.TemplateResponse("message.html", {
            "request": request,
            "title": "User Not Found",
            "message": "❌ User does not exist."
        })
        
    if user.is_verified:
        return templates.TemplateResponse("message.html", {
            "request": request,
            "title": "Already Verified",
            "message": "✅ Email already verified!"
        })
        
    user.is_verified = True
    db.commit()
   
    return templates.TemplateResponse("message.html", {
        "request": request,
        "title": "Success",
        "message": "🎉 Email verified successfully! You can now <a href='/auth/login-page'>login</a>."
    })

@auth_router.post("/forgot-password", response_class=HTMLResponse)
def forgot_password(request: Request, email: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return HTMLResponse(content="❌ No account found with that email", status_code=404)
    
    token = generate_verification_token(email)
    base_url = get_base_url(request)
    reset_link = f"{base_url}/auth/reset-password?token={token}"
    
    # Send email with reset link
    user_email = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    
    yag = yagmail.SMTP(user=user_email, password=password)
    subject = "Reset your password"
    content = f"Click the link to reset your password: {reset_link}"
    yag.send(to=email, subject=subject, contents=content)
    
    return HTMLResponse(content="📧 Password reset link sent!")

@auth_router.get("/reset-password", response_class=HTMLResponse)
def reset_password_form(request: Request, token: str):
    email = confirm_token(token)
    if not email:
        return HTMLResponse(content="❌ Invalid or expired token", status_code=400)
    
    return templates.TemplateResponse(
        "reset_password.html",
        {"request": request, "token": token}
    )

@auth_router.post("/reset-password", response_class=HTMLResponse)
def reset_password(
    request: Request,
    token: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db)
):
    # Validate password complexity
    if not is_valid_password(new_password):
        return templates.TemplateResponse(
            "reset_password.html",
            {
                "request": request, 
                "token": token,
                "error": "Password must be at least 8 characters with at least one lowercase letter, one uppercase letter, one number, and one special character"
            },
            status_code=400
        )
       
    if new_password != confirm_password:
        return templates.TemplateResponse(
            "reset_password.html",
            {
                "request": request, 
                "token": token,
                "error": "Passwords do not match"
            },
            status_code=400
        )
       
    email = confirm_token(token)
    if not email:
        return templates.TemplateResponse(
            "reset_password.html",
            {
                "request": request, 
                "token": token,
                "error": "Invalid or expired token"
            },
            status_code=400
        )
       
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return templates.TemplateResponse(
            "reset_password.html",
            {
                "request": request, 
                "token": token,
                "error": "User not found"
            },
            status_code=400
        )
       
    user.hashed_password = hash_password(new_password)
    db.commit()
   
    # Use a regular HTTP redirect to login page after successful password reset
    # This ensures a complete page refresh
    response = RedirectResponse(url="/auth/login-page?msg=Password+reset+successful", status_code=303)
    return response

@auth_router.get("/forgot-password-page", response_class=HTMLResponse)
def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request})

@auth_router.get("/resend-verification", response_class=HTMLResponse)
def resend_verification(request: Request, email: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return HTMLResponse(content="❌ No account found with that email", status_code=404)
        
    if user.is_verified:
        return HTMLResponse(content="✅ Email already verified! You can login.", status_code=200)
        
    token = generate_verification_token(email)
    base_url = get_base_url(request)
    link = f"{base_url}/auth/verify-email?token={token}"
    send_verification_email(email, link)
    
    return HTMLResponse(content="📧 Verification email resent! Please check your inbox.")


@auth_router.post("/courses")
def create_course(
    title: str = Form(...),
    description: str = Form(...),
    db: Session = Depends(get_db),
    user=Depends(require_role("teacher"))
):
    # Count existing courses by this teacher
    existing_courses_count = db.query(Course).filter(Course.teacher_id == user["user_id"]).count()
    
    # Check if the teacher already has 15 courses
    if existing_courses_count >= 15:
        return HTMLResponse(
            content='<div class="toast error">❌ You can create a maximum of 15 courses per account.</div>',
            status_code=400
        )
    
    # If under the limit, create the new course
    new_course = Course(title=title, description=description, teacher_id=user["user_id"])
    db.add(new_course)
    db.commit()
    db.refresh(new_course)
    
    return HTMLResponse(
        content='<div class="toast success">✅ Course created successfully!</div>',
        status_code=200
    )

@auth_router.post("/courses/{course_id}/upload-students", response_class=HTMLResponse)
def upload_students(course_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    df = pd.read_excel(file.file)

    for email in df["email"]:
        student = db.query(User).filter(User.email == email).first()
        if not student:
            return HTMLResponse(content="❌ Student not found", status_code=404)
        invite = db.query(CourseInvite).filter_by(course_id=course_id, student_id=student.id).first()
        enrollment = db.query(Enrollment).filter_by(course_id=course_id, student_id=student.id).first()

        #if invite and invite.status == "accepted":
            #return HTMLResponse(content="✅ Student already enrolled.", status_code=200)

        if not invite:
            invite = CourseInvite(course_id=course_id, student_id=student.id)
            db.add(invite)
        else:
            invite.status = "pending"

        # Send email with confirmation link
        token = generate_verification_token(email)
        link = f"http://127.0.0.1:8000/auth/accept-invite?token={token}&course_id={course_id}"
        send_verification_email(email, f"You've been invited to join a course! Accept: {link}")

    db.commit()

    toast = f"<div class='toast success'>✅ student(s) invited successfully.</div>"
    return HTMLResponse(content=toast, status_code=200)





@auth_router.get("/courses/new", response_class=HTMLResponse)
def new_course_form(request: Request,    db: Session = Depends(get_db),user=Depends(require_role("teacher"))):
    teacher_id = user["user_id"]
    courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()
    return templates.TemplateResponse("create_course.html", {"request": request, "courses":courses})


@auth_router.get("/courses/{course_id}/enroll", response_class=HTMLResponse)
def enroll_students_page(request: Request, course_id: int, db: Session = Depends(get_db), user=Depends(require_teacher_or_ta())):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)
    
    teacher_id = user["user_id"]
    courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()
    
    return templates.TemplateResponse("enroll_students.html", {
        "request": request,
        "course": course,
        "role": user["role"],
        "courses": courses
    })

@auth_router.get("/courses/{course_id}/invite-student", response_class=HTMLResponse)
def invite_student_page(request: Request, course_id: int,user=Depends(require_teacher_or_ta())):

    return templates.TemplateResponse("invite_student.html", {
        "request": request,
        "course_id": course_id
    })

@auth_router.post("/courses/{course_id}/invite-student", response_class=HTMLResponse)
def invite_student(
    request: Request,
    course_id: int,
    email: str = Form(...),
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    student = db.query(User).filter(User.email == email).first()
    if not student:
        return HTMLResponse(content="❌ Student not found", status_code=404)

    invite = db.query(CourseInvite).filter_by(course_id=course_id, student_id=student.id).first()
    enrollment = db.query(Enrollment).filter_by(course_id=course_id, student_id=student.id).first()

    #if invite and invite.status == "accepted":
        #return HTMLResponse(content="✅ Student already enrolled.", status_code=200)

    if not invite:
        invite = CourseInvite(course_id=course_id, student_id=student.id)
        db.add(invite)
    else:
        invite.status = "pending"

    # Send email with confirmation link
    token = generate_verification_token(email)
    link = f"http://127.0.0.1:8000/auth/accept-invite?token={token}&course_id={course_id}"
    send_verification_email(email, f"You've been invited to join a course! Accept: {link}")

    db.commit()
    
    return HTMLResponse(
        content='<div class="toast success">✅ Invite sent to student via email.</div>',
        status_code=200
    )
    
@auth_router.get("/accept-invite", response_class=HTMLResponse)
def accept_invite(token: str, course_id: int, db: Session = Depends(get_db)):
    email = confirm_token(token)
    if not email:
        return HTMLResponse("❌ Invalid or expired invite", status_code=400)

    student = db.query(User).filter(User.email == email).first()
    if not student:
        return HTMLResponse("❌ Student not found", status_code=404)

    invite = db.query(CourseInvite).filter_by(course_id=course_id, student_id=student.id).first()
    if not invite:
        return HTMLResponse("❌ Invite not found", status_code=404)

    if invite.status == "accepted":
        return HTMLResponse("✅ You've already joined this course!")

    invite.status = "accepted"


    # Check if already enrolled
    enrollment = db.query(Enrollment).filter_by(course_id=course_id, student_id=student.id).first()
    if not enrollment:
        enrollment = Enrollment(course_id=course_id, student_id=student.id, is_accepted=True)
        db.add(enrollment)
    else:
        enrollment.is_accepted = True

    db.commit()

    return HTMLResponse("""
        <div class="toast success">🎉 Invite accepted and course joined!</div>
        <script>
          setTimeout(() => { window.location.href = '/auth/student/courses'; }, 1500);
        </script>
    """)



@auth_router.post("/courses/{course_id}/accept", response_class=HTMLResponse)
def accept_invite(
    course_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    enrollment = db.query(Enrollment).filter_by(course_id=course_id, student_id=user["user_id"]).first()

    if not enrollment or enrollment.is_accepted:
        return HTMLResponse(content="⚠️ Invalid or already accepted invite.", status_code=400)

    enrollment.is_accepted = True

    # Update invite if exists
    invite = db.query(CourseInvite).filter_by(course_id=course_id, student_id=user["user_id"]).first()
    if invite:
        invite.status = "accepted"

    db.commit()

    course = db.query(Course).filter(Course.id == course_id).first()
    return HTMLResponse(content=f"""
        <div class="card">
            <h4>{course.title}</h4>
            <p>{course.description}</p>
            <div class="toast success">🎉 Successfully joined the course!</div>
        </div>
    """)



@auth_router.get("/teacher/dashboard", response_class=HTMLResponse)
def teacher_dashboard(
    request: Request,
    db: Session = Depends(get_db),
    user=Depends(require_role("teacher"))
):
    teacher_id = user["user_id"]
    courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()
    teacher = db.query(User).filter(User.id == teacher_id).first()
    
    return templates.TemplateResponse("teacher_dashboard.html", {
        "request": request,
        "courses": courses,
        "teacher_name":teacher.f_name
    })

@auth_router.get("/debug/invites")
def debug_invites(db: Session = Depends(get_db)):
    invites = db.query(CourseInvite).all()
    return [{"email": db.query(User).filter(User.id == i.student_id).first().email,
             "course_id": i.course_id, "status": i.status} for i in invites]

@auth_router.post("/courses/{course_id}/generate-attendance-code", response_class=HTMLResponse)
def generate_attendance_code(
    request : Request,
    course_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    # Ensure the teacher owns the course
    course = db.query(Course).filter(
        Course.id == course_id,
        Course.teacher_id == user["user_id"]
    ).first()

    if not course:
        return HTMLResponse(content="❌ Unauthorized or course not found", status_code=403)

    now = datetime.utcnow()
    existing_code = db.query(AttendanceCode).filter(
        AttendanceCode.course_id == course_id,
        AttendanceCode.expires_at > now
    ).order_by(desc(AttendanceCode.created_at)).first()
    if existing_code:
        code = existing_code.code
        expires_at = existing_code.expires_at
    else:
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        expires_at = datetime.utcnow() + timedelta(minutes=10)

        # Create and save the new attendance code
        new_code = AttendanceCode(
            course_id=course_id,
            code=code,
            expires_at=expires_at
        )
        db.add(new_code)
        db.commit()

    return templates.TemplateResponse("attendance_code.html", {
        "request": request,
        "code": code,
        "expires_at": expires_at,
    }, status_code=200)


@auth_router.post("/courses/{course_id}/submit-attendance", response_class=HTMLResponse)
def submit_attendance(
    course_id: int,
    code: str = Form(...),
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    student_id = user["user_id"]
    now = datetime.utcnow()

    # Validate the code
    valid_code = db.query(AttendanceCode).filter(
        AttendanceCode.course_id == course_id,
        AttendanceCode.code == code,
        AttendanceCode.expires_at > now
    ).first()

    if not valid_code:
        return HTMLResponse(
            content="<div class='toast error'>❌ Invalid or expired code.</div>",
            status_code=400
        )

    # Prevent duplicate attendance entries with same code
    already_present = db.query(AttendanceRecord).filter_by(
        student_id=student_id,
        course_id=course_id,
        code_used=code
    ).first()

    if already_present:
        return HTMLResponse(
            content="<div class='toast warning'>⚠️ You’ve already marked attendance with this code.</div>",
            status_code=200
        )

    # Record attendance
    record = AttendanceRecord(
        student_id=student_id,
        course_id=course_id,
        code_used=code
    )
    db.add(record)
    db.commit()

    return HTMLResponse(
        content="<div class='toast success'>✅ Attendance marked successfully!</div>"
    )

@auth_router.get("/courses/{course_id}/attendance", response_class=HTMLResponse)
def view_attendance_page(
    course_id: int,
    request: Request,
    date: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        return HTMLResponse(content="❌ Course not found", status_code=404)

    teacher_id = user["user_id"]
    courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()

    selected_date = None
    query = db.query(AttendanceRecord).filter(AttendanceRecord.course_id == course_id)

    if date:
        try:
            selected_date = datetime.strptime(date, "%Y-%m-%d").date()
            query = query.filter(func.date(AttendanceRecord.attended_at) == selected_date)
        except ValueError:
            selected_date = None

    records = query.all()
    students = [e.student for e in course.enrollments]

    formatted_records = []
    for record in records:
        student = db.query(User).filter(User.id == record.student_id).first()
        formatted_records.append({
            "student": {
                "name": f"{student.f_name} {student.l_name}"
            },
            "date": record.attended_at,
            "present": True  # Since they submitted a code or were marked manually
        })

    return templates.TemplateResponse("teacher_attendance.html", {
        "request": request,
        "course": course,
        "courses":courses,
        "students": students,
        "attendance_records": formatted_records,
        "selected_date": selected_date.strftime("%Y-%m-%d") if selected_date else "",
        "role": user["role"]
    })


@auth_router.post("/courses/{course_id}/mark-manual-attendance", response_class=HTMLResponse)
def mark_manual_attendance(
    course_id: int,
    present_ids: List[int] = Form(default=[]),
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    today = datetime.utcnow().date()

    for student_id in present_ids:
        already_marked = db.query(AttendanceRecord).filter(
            AttendanceRecord.course_id == course_id,
            AttendanceRecord.student_id == student_id,
            func.date(AttendanceRecord.attended_at) == today
        ).first()

        if not already_marked:
            db.add(AttendanceRecord(
                student_id=student_id,
                course_id=course_id,
                attended_at=datetime.utcnow(),
                code_used="Manual"
            ))

    db.commit()
    return HTMLResponse(
        content="<div class='toast success'>✅ Attendance saved!</div>",
        status_code=200
    )

@auth_router.get("/courses/{course_id}/attendance/export-attendance")
def export_attendance_csv(
    course_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_teacher_or_ta())
):
    course = db.query(Course).filter_by(id=course_id, teacher_id=user["user_id"]).first()
    if not course:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    records = db.query(AttendanceRecord).filter_by(course_id=course_id).all()
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Added Roll Number to the header
    writer.writerow(["Student Name", "Roll Number", "Email", "Date", "Code Used"])
    
    for record in records:
        student = db.query(User).filter(User.id == record.student_id).first()
        
        # Added roll_number to the row, with a fallback if it's None
        writer.writerow([
            f"{student.f_name} {student.l_name}",
            student.roll_number or "Not provided",  # Include roll number or placeholder if not available
            student.email,
            record.attended_at.strftime("%d/%m/%Y %H:%M"),
            record.code_used
        ])
        
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_{course_id}.csv"}
    )

@auth_router.delete("/courses/{course_id}/attendance/clear")
def clear_attendance_records(course_id: int, db: Session = Depends(get_db), user=Depends(require_role("teacher"))):
    course = db.query(Course).filter_by(id=course_id, teacher_id=user["user_id"]).first()
    if not course:
        raise HTTPException(status_code=403, detail="Unauthorized")

    db.query(AttendanceRecord).filter_by(course_id=course_id).delete()
    db.commit()
    return {"msg": f"✅ Cleared all attendance for course ID {course_id}"}


@auth_router.post("/courses/{course_id}/submit-attendance", response_class=HTMLResponse)
def submit_attendance(
    course_id: int,
    code: str = Form(...),
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    student_id = user["user_id"]
    now = datetime.utcnow()

    valid_code = db.query(AttendanceCode).filter(
        AttendanceCode.course_id == course_id,
        AttendanceCode.code == code,
        AttendanceCode.expires_at > now
    ).first()

    if not valid_code:
        return HTMLResponse("<div class='toast error'>❌ Invalid or expired code.</div>", status_code=400)

    already_marked = db.query(AttendanceRecord).filter_by(
        student_id=student_id,
        course_id=course_id,
        code_used=code
    ).first()

    if already_marked:
        return HTMLResponse("<div class='toast warning'>⚠️ Already marked present with this code.</div>", status_code=200)

    db.add(AttendanceRecord(
        student_id=student_id,
        course_id=course_id,
        code_used=code
    ))
    db.commit()

    return HTMLResponse("<div class='toast success'>✅ Attendance marked successfully!</div>")

# View attendance page
@auth_router.get("/courses/{course_id}/student-attendance", response_class=HTMLResponse)
def view_student_attendance(
    course_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    student_id = user["user_id"]
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        return HTMLResponse("❌ Course not found", status_code=404)
    enrollments = db.query(Enrollment).filter_by(student_id=student_id, is_accepted=True).all()
    courses = [enroll.course for enroll in enrollments]

    records = db.query(AttendanceRecord).filter_by(course_id=course_id, student_id=student_id).all()
    return templates.TemplateResponse("student_attendance.html", {
        "request": request,
        "course": course,
        "records": records,
        "courses":courses
    })

@auth_router.get("/courses/{course_id}/invite-ta", response_class=HTMLResponse)
def invite_ta_page(
    request: Request,
    course_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_role("teacher"))
):
    teacher_id = user["user_id"]
    courses = db.query(Course).filter(Course.teacher_id == teacher_id).all()
    course = db.query(Course).filter_by(id=course_id, teacher_id=user["user_id"]).first()
    if not course:
        return HTMLResponse(content="❌ Unauthorized", status_code=403)

    return templates.TemplateResponse("invite_ta.html", {
        "request": request,
        "course": course,
        "courses":courses
    })


@auth_router.post("/courses/{course_id}/invite-ta", response_class=HTMLResponse)
def invite_ta(
    course_id: int,
    email: str = Form(...),
    db: Session = Depends(get_db),
    user=Depends(require_role("teacher"))
):
    course = db.query(Course).filter_by(id=course_id, teacher_id=user["user_id"]).first()
    if not course:
        return HTMLResponse(content="<div class='toast error'> Course doesn't Exist!</div>", status_code=200)

    student = db.query(User).filter_by(email=email).first()
    if not student:
        return HTMLResponse(content="<div class='toast error'> User Doesn't Exist!</div>", status_code=200)

    existing = db.query(TeachingAssistant).filter_by(course_id=course_id, student_id=student.id).first()
    if existing:
        return HTMLResponse(content="<div class='toast error'> TA already invited or has joined!</div>", status_code=200)

    invite = TeachingAssistant(course_id=course_id, student_id=student.id, status="pending")
    db.add(invite)
    db.commit()

    return HTMLResponse("<div class='toast success'>✅ TA invite sent successfully!</div>")



@auth_router.post("/courses/{course_id}/accept-ta", response_class=HTMLResponse)
def accept_ta_invite(
    course_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_role("student"))
):
    ta = db.query(TeachingAssistant).filter_by(course_id=course_id, student_id=user["user_id"]).first()
    if not ta or ta.status == "accepted":
        return HTMLResponse("❌ Invalid or already accepted", status_code=400)

    ta.status = "accepted"
    db.commit()
    return HTMLResponse('<div class="toast success">🎉 You are now a TA for this course!</div>')
