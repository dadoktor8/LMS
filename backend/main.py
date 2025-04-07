from dotenv import load_dotenv
import os
import sys
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parents[0] / '.env')
print("Loaded SECRET_KEY:", os.getenv("SECRET_KEY"))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

sys.path.append(str(Path(__file__).resolve().parent))
import os
from backend.auth.routes import auth_router
from db.init_db import init_db
from db.session import get_db

app = FastAPI()
templates = Jinja2Templates(directory="backend/templates")
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

init_db()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router, prefix="/auth")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})
