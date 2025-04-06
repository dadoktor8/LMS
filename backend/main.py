from dotenv import load_dotenv
import os
import sys
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parents[0] / '.env')
print("Loaded SECRET_KEY:", os.getenv("SECRET_KEY"))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
sys.path.append(str(Path(__file__).resolve().parent))
import os
from backend.auth.routes import auth_router
from db.init_db import init_db

app = FastAPI()
init_db()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router, prefix="/auth")

@app.get("/")
def root():
    return {"message": "Welcome to the backend!"}