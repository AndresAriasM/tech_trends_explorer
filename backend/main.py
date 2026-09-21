"""
Tech Trends Explorer — FastAPI Backend
Entry point: uvicorn main:app --reload
"""

from dotenv import load_dotenv
load_dotenv()  # loads backend/.env (or root .env if running from root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import (
    assistant,
    auth,
    categories,
    delphi,
    hype_cycle,
    s_curve,
    tasks,
    workshop,
)

app = FastAPI(
    title="Tech Trends Explorer API",
    description="Backend for Hype Cycle and technology trend analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hype_cycle.router, prefix="/api/hype-cycle", tags=["Hype Cycle"])
app.include_router(categories.router, prefix="/api/categories", tags=["Categories"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["Tasks"])
app.include_router(s_curve.router,   prefix="/api/s-curve",   tags=["S-Curve"])
app.include_router(workshop.router,  prefix="/api/workshop",  tags=["Workshop"])
app.include_router(auth.router,      prefix="/api/auth",      tags=["Auth"])
app.include_router(delphi.router,    prefix="/api/delphi",    tags=["Delphi"])
app.include_router(assistant.router, prefix="/api/assistant", tags=["Asistente IA"])


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
