# backend/db/session.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
from db.database import SessionLocal,Base,engine

load_dotenv()

# Ensure the DB path always resolves to backend/test.db regardless of where script is run
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
