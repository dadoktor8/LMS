from itsdangerous import URLSafeTimedSerializer
from dotenv import load_dotenv
import os

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")

SECURITY_SALT = "email-confirm"

s = URLSafeTimedSerializer(SECRET_KEY)

def generate_verification_token(email: str) -> str:
    return s.dumps(email, salt=SECURITY_SALT)

def confirm_token(token: str, expiration=3600) -> str | None:
    try:
        return s.loads(token, salt=SECURITY_SALT, max_age=expiration)
    except Exception:
        return None
