"""FastAPI HTTP layer for the GDPR Smart Agent.

Thin wrapper over ``src.agent.graph.answer_question`` — no business logic
lives here. Run with::

    uvicorn src.api:app --reload

Interactive docs are then served at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agent.graph import answer_question, get_graph


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="A GDPR question.")
    k: int = Field(4, ge=1, le=20, description="Number of passages to retrieve.")


class Source(BaseModel):
    citation: str
    article_number: int | str
    paragraph_number: int | str
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build (and cache) the agent graph once at startup, not per request.
    get_graph()
    yield


app = FastAPI(
    title="GDPR Smart Agent",
    description="Ask questions about the EU GDPR and get answers with article citations.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Cheap liveness probe — does not call the LLM."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a GDPR question with citations to the underlying articles."""
    result = answer_question(request.question, k=request.k)
    sources = [
        Source(
            citation=chunk.citation,
            article_number=chunk.article_number,
            paragraph_number=chunk.paragraph_number,
            title=chunk.title,
            url=chunk.url,
        )
        for chunk in result.get("chunks", [])
    ]
    return AskResponse(answer=result["answer"], sources=sources)
