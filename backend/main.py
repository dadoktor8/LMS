from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from starlette.middleware.sessions import SessionMiddleware
from backend.db.database import Base
load_dotenv(dotenv_path=Path(__file__).resolve().parents[0] / '.env')
print("Loaded SECRET_KEY:", os.getenv("SECRET_KEY"))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from backend.ai.ai_routes import ai_router


sys.path.append(str(Path(__file__).resolve().parent))
import os
from backend.auth.routes import auth_router
from backend.db.init_db import init_db
secret_key = os.getenv("SECRET_KEY")
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=secret_key)
templates = Jinja2Templates(directory="backend/templates")
app.mount("/static", StaticFiles(directory="backend/static"), name="static")
app.mount("/uploads", StaticFiles(directory="backend/uploads"), name="uploads")


init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router, prefix="/auth")
app.include_router(ai_router, prefix="/ai")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

#Made with love Docjenny&GPT4-o