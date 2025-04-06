from db.session import SessionLocal
from db.models import User

try:
    db = SessionLocal()
    users = db.query(User).all()
    print("✅ DB is working! Users in database:")
    for user in users:
        print(f"ID: {user.id}, Email: {user.email}, Role: {user.role}")
except Exception as e:
    print("❌ DB connection failed:", e)
finally:
    db.close()
