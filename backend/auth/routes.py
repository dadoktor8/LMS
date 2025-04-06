# backend/auth/routes.py
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
from typing import Literal
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from db.session import get_db
from db.models import User

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
auth_router = APIRouter()

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
    return {"msg": "User created successfully"}

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
