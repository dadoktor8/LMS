# backend/db/init_db.py
from db.session import Base, engine
from db import models
from db.models import User


def init_db():
    print("📦 Initializing DB...")
    Base.metadata.create_all(bind=engine)
