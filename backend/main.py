"""
Tech Trends Explorer — FastAPI Backend
Entry point: uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import hype_cycle, categories, tasks

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


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
