from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from faq_engine import FAQEngine

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="E-Commerce FAQ Chatbot",
              description="Semantic FAQ search with summarization + FAISS",
              version="0.1.0")


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class AskResponse(BaseModel):
    question: str
    answer: str
    source: str
    category: str
    confidence: float
    note: Optional[str] = None


# Load dataset and initialize engine on startup
@app.on_event("startup")
async def startup_event():
    global engine
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.exception("Failed to load data.json: %s", e)
        raise

    engine = FAQEngine(data)
    logger.info("FAQEngine initialized with %d documents", len(data))


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Accept a user question, run semantic search, and return the best compressed answer.

    If top match confidence is below 0.6, suggest human support (bonus requirement).
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must be provided")

    results = engine.query(req.question, top_k=req.top_k or 3)
    if not results:
        raise HTTPException(status_code=404, detail="no results found")

    best = results[0]
    conf = best.get("score", 0.0)

    note = None
    if conf < 0.6:
        note = "Low confidence — forwarding to human support is recommended."

    meta = best["metadata"]

    return AskResponse(
        question=req.question,
        answer=best["answer"],
        source=meta.get("source") or meta.get("title") or "unknown",
        category=meta.get("category") or "general",
        confidence=round(conf, 4),
        note=note,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
