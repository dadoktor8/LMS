from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DB_PATH = os.path.join(BACKEND_DIR, "test.db")

# Get database URL from environment variables
# Render automatically provides DATABASE_URL for PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL")

# If DATABASE_URL is not provided, fall back to SQLite
if not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH}"
    connect_args = {"check_same_thread": False}
else:
    # For PostgreSQL
    connect_args = {}
    
    # Render provides PostgreSQL URLs that start with postgres://
    # SQLAlchemy expects postgresql://, so fix that if needed
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()