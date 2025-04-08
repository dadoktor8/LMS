from fastapi import APIRouter, HTTPException, Depends, Request, Form
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import UploadFile, File 
import pandas as pd
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
from typing import Literal
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from utils.tokens import generate_verification_token
from utils.tokens import confirm_token
from utils.email_utils import send_verification_email
from utils.tokens import generate_verification_token
from db.models import User
from db.database import engine,get_db

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
auth_router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")
print(engine.url)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    role: Literal["admin", "student"]

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_role(required_role: str):
    def role_checker(token: str = Depends(oauth2_scheme)):
        payload = decode_token(token)
        if payload.get("role") != required_role:
            raise HTTPException(status_code=403, detail="Not authorized")
        return payload
    return role_checker

# === JSON API SIGNUP ===
@auth_router.post("/signup")
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == payload.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    new_user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=payload.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    print(f"Creating user with email: {payload.email}")
    return {"msg": "User created successfully"}

# === HTMX SIGNUP FORM ===
@auth_router.get("/signup-page", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@auth_router.post("/signup-form", response_class=HTMLResponse)
def signup_form(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    db: Session = Depends(get_db)
):
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        return HTMLResponse(content="❌ User already exists", status_code=400)
    
    new_user = User(email=email, hashed_password=hash_password(password), role=role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    print("✅ User added:", new_user.email)
    token = generate_verification_token(email)
    link = f"{os.getenv('FRONTEND_URL', 'http://127.0.0.1:8000')}/auth/verify-email?token={token}"
    send_verification_email(email, link)
    return HTMLResponse(content="✅ User created successfully!")

# === JSON API LOGIN ===
@auth_router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }
@auth_router.get("/login")
def redirect_to_login_page(msg: str = ""):
    return RedirectResponse(url=f"/auth/login-page?msg={msg}", status_code=302)


# === HTMX LOGIN FORM ===
@auth_router.get("/login-page", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

from fastapi.responses import Response

@auth_router.post("/login-form", response_class=HTMLResponse)
def login_form(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return HTMLResponse(
            content='<div class="toast error">❌ Invalid email or password</div>',
            status_code=401
        )
    
    if user.role == "admin":
        redirect_url = "/auth/admin/dashboard"
    elif user.role == "student":
        redirect_url = "/auth/student/courses"
    elif user.role == "teacher":
        redirect_url = "/auth/teacher/dashboard"
    else:
        return HTMLResponse(
            content=f'<div class="toast success">✅ Welcome back, {user.email}!</div>'
        )

    # 🔁 Return an HTMX-specific redirect
    response = HTMLResponse(status_code=200)
    response.headers["HX-Redirect"] = redirect_url
    return response


# === USER INFO + ROLE ROUTES ===
@auth_router.get("/me")
def read_users_me(token: str = Depends(oauth2_scheme)):
    user_data = decode_token(token)
    return {"email": user_data.get("sub"), "role": user_data.get("role")}

@auth_router.get("/admin/dashboard")
def admin_dashboard(user=Depends(require_role("admin"))):
    return {"message": "Welcome Admin!", "email": user.get("sub")}

@auth_router.get("/student/courses")
def student_courses(user=Depends(require_role("student"))):
    return {"message": "Welcome Student! Here are your courses.", "email": user.get("sub")}


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
        "message": "🎉 Email verified successfully!"
    })

@auth_router.post("/forgot-password", response_class=HTMLResponse)
def forgot_password(request: Request, email: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return HTMLResponse(content="❌ No account found with that email", status_code=404)

    token = generate_verification_token(email)
    reset_link = f"{os.getenv('FRONTEND_URL', 'http://127.0.0.1:8000')}/auth/reset-password?token={token}"
    send_verification_email(email, reset_link)  # Reuse your email function

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
    if new_password != confirm_password:
        return HTMLResponse(content="❌ Passwords do not match", status_code=400)

    email = confirm_token(token)
    if not email:
        return HTMLResponse(content="❌ Invalid or expired token", status_code=400)

    user = db.query(User).filter(User.email == email).first()
    if not user:
        return HTMLResponse(content="❌ User not found", status_code=400)

    user.hashed_password = hash_password(new_password)
    db.commit()

    # Redirect to login page after successful password reset
    return RedirectResponse(url="/auth/login?msg=Password+reset+successful", status_code=303)

@auth_router.get("/forgot-password-page", response_class=HTMLResponse)
def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request})


@auth_router.post("/courses")
def create_course(
    title: str = Form(...),
    description: str = Form(...),
    db: Session = Depends(get_db),
    user=Depends(require_role("teacher"))
):
    new_course = Course(title=title, description=description, teacher_id=user["user_id"])
    db.add(new_course)
    db.commit()
    db.refresh(new_course)
    return {"msg": "Course created", "course_id": new_course.id}

@auth_router.post("/courses/{course_id}/upload-students")
def upload_students(course_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    df = pd.read_excel(file.file)

    for email in df["email"]:
        student = db.query(User).filter(User.email == email).first()
        if student:
            enrollment = Enrollment(course_id=course_id, student_id=student.id)
            db.add(enrollment)

    db.commit()
    return {"msg": "Students enrolled successfully"}


@auth_router.get("/courses/new", response_class=HTMLResponse)
def new_course_form(request: Request):
    return templates.TemplateResponse("create_course.html", {"request": request})


@auth_router.get("/courses/{course_id}/upload-students-page", response_class=HTMLResponse)
def upload_students_page(request: Request, course_id: int):
    return templates.TemplateResponse("upload_students.html", {
        "request": request,
        "course_id": course_id
    })

@auth_router.get("/teacher/dashboard", response_class=HTMLResponse)
def teacher_dashboard(request: Request, db: Session = Depends(get_db)):
    # Fetch teacher's courses (optional enhancement)
    return templates.TemplateResponse("teacher_dashboard.html", {
        "request": request
    })

