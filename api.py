"""
Phase C — Minimal API wrapper

Endpoints:
  GET  /health
  POST /ingest?pdf_path=...
  POST /ask?question=...
"""

from fastapi import FastAPI, HTTPException

from ingest import ingest
from query import ask

app = FastAPI(title="Production RAG API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest_pdf(pdf_path: str) -> dict:
    try:
        return ingest(pdf_path)
    except Exception as e:  # keep it simple for beginner phase
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask")
def ask_question(question: str) -> dict:
    try:
        return ask(question)
    except Exception as e:  # keep it simple for beginner phase
        raise HTTPException(status_code=400, detail=str(e))

