from datetime import datetime, timedelta
import re
from fastapi import HTTPException
from itsdangerous import URLSafeTimedSerializer
from dotenv import load_dotenv
import os

import jwt

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")

SECURITY_SALT = "email-confirm"

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

s = URLSafeTimedSerializer(SECRET_KEY)

PASSWORD_PATTERN = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*(),.?":{}|<>]).{8,}$')

def generate_verification_token(email: str) -> str:
    return s.dumps(email, salt=SECURITY_SALT)

def confirm_token(token: str, expiration=3600) -> str | None:
    try:
        return s.loads(token, salt=SECURITY_SALT, max_age=expiration)
    except Exception:
        return None

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