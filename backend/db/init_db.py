# backend/db/init_db.py
from backend.db.database import Base,engine
from backend.db import models
from backend.db.models import User


def init_db():
    print("📦 Initializing DB...")
    print("📦 Creating tables in:", engine.url)
    print("📦 Tables created:", Base.metadata.tables.keys())
    Base.metadata.create_all(bind=engine)
