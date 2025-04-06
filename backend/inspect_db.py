import os
from db.session import SessionLocal
from db.models import User

# Resolve correct absolute path to database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "backend", "test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"
print(DB_PATH)
print(BASE_DIR)

# Now import the session (which uses the env var)
from db.session import SessionLocal

db = SessionLocal()

users = db.query(User).all()

print("=== Users in Database ===")
for user in users:
    print(f"ID: {user.id}, Email: {user.email}, Role: {user.role}, Hashed Password: {user.hashed_password}")
