"""FastAPI server: REST + Server-Sent-Event streams of the LangGraph run, and the static UI.

    uvicorn app.api:app --port 8000     (run from the backend/ folder)
"""
from __future__ import annotations

import json
from typing import Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, memory, service
from .config import FRONTEND_DIR, settings
from .graph import get_graph, get_store
from .ratelimit import groq_usage
from .samples import SCENARIOS
from .vectorstore import get_index

app = FastAPI(title="Autonomous Support Investigation Agent")


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    get_graph()
    try:  # Qdrant Cloud strict mode rejects filters on non-indexed fields -> make sure indexes exist
        idx = get_index()
        if not idx.local and idx.exists():
            idx.ensure_payload_indexes()
    except Exception as e:  # noqa: BLE001 - the UI shows Qdrant problems via /api/health
        print(f"WARNING: Qdrant check failed at startup: {e}")


def sse(events: Iterator[dict]) -> StreamingResponse:
    def gen():
        for ev in events:
            yield f"data: {json.dumps(ev, default=str)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class TicketIn(BaseModel):
    customer_id: str = "anonymous"
    subject: str
    body: str
    auto_accept: bool = False


class ResumeIn(BaseModel):
    answer: Optional[str] = None      # clarification answer
    action: Optional[str] = None      # review: accept | amend | reject
    feedback: Optional[str] = None
    root_cause: Optional[str] = None  # review: the lead's diagnosis (authoritative, indexed as lead-verified)
    solution: Optional[str] = None
    auto_accept: bool = False


class LeadSolutionIn(BaseModel):
    root_cause: str = ""
    solution: str = ""
    note: str = ""
    auto_accept: bool = False


class FollowupIn(BaseModel):
    message: str
    auto_accept: bool = False


@app.get("/api/health")
def health():
    try:
        idx = get_index()
        vectors = {"historical": idx.count("historical"), "live": idx.count("live")}
    except Exception as e:  # noqa: BLE001
        vectors = {"historical": 0, "live": 0, "error": str(e)[:300]}
    return {
        "groq_configured": bool(settings.groq_api_key),
        "model": settings.groq_model, "fast_model": settings.groq_fast_model,
        "qdrant": "cloud" if settings.qdrant_url else f"embedded:{settings.qdrant_path}",
        "collection": settings.collection,
        "vectors": vectors,
        "db": db.counts(),
        "groq_usage": groq_usage.snapshot(),
        "qdrant_max_rps": settings.qdrant_max_rps if settings.qdrant_url else None,
    }


@app.get("/api/graph")
def graph_diagram():
    """Topology of the compiled graph (the UI draws it, so the picture is always the real graph)."""
    dg = get_graph().get_graph()
    return {"nodes": list(dg.nodes),
            "edges": [{"source": e.source, "target": e.target, "conditional": e.conditional} for e in dg.edges],
            "mermaid": dg.draw_mermaid()}


@app.get("/api/samples")
def samples():
    return SCENARIOS


@app.post("/api/tickets")
def create_ticket(t: TicketIn):
    if not t.subject.strip() and not t.body.strip():
        raise HTTPException(400, "subject or body required")
    _, events = service.new_ticket(t.customer_id, t.subject, t.body, auto_accept=t.auto_accept)
    return sse(events)


@app.post("/api/threads/{thread_id}/resume")
def resume(thread_id: str, r: ResumeIn):
    if r.action:
        value = {"action": r.action, "feedback": r.feedback, "root_cause": r.root_cause, "solution": r.solution,
                 "by": "support_lead"}
    else:
        value = {"answer": r.answer or ""}
    return sse(service.resume(thread_id, value, auto_accept=r.auto_accept))


@app.post("/api/threads/{thread_id}/lead-solution")
def lead_solution(thread_id: str, s: LeadSolutionIn):
    """The lead's root cause + fix for a case: applied at review, or to a case that already finished
    (e.g. escalated). The reply is rewritten from it and the case is indexed as lead-verified."""
    if not (s.root_cause.strip() or s.solution.strip()):
        raise HTTPException(400, "root_cause or solution required")
    try:
        events = service.lead_solution(thread_id, s.root_cause, s.solution, s.note, auto_accept=s.auto_accept)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))
    return sse(events)


@app.post("/api/threads/{thread_id}/followup")
def followup(thread_id: str, f: FollowupIn):
    return sse(service.followup(thread_id, f.message, auto_accept=f.auto_accept))


@app.post("/api/threads/{thread_id}/retry")
def retry(thread_id: str):
    return sse(service.retry(thread_id))


@app.get("/api/threads/{thread_id}")
def thread(thread_id: str):
    return service.thread_snapshot(thread_id)


@app.get("/api/cases")
def cases(limit: int = 50):
    return db.list_cases(limit)


@app.get("/api/customers/{customer_id}/memory")
def customer_memory(customer_id: str):
    return {"customer_id": customer_id, "facts": memory.get_facts(get_store(), customer_id),
            "cases": db.customer_cases(customer_id, limit=10)}


@app.delete("/api/customers/{customer_id}/memory")
def clear_memory(customer_id: str):
    return {"removed": memory.clear(get_store(), customer_id)}


@app.get("/api/tickets/{ticket_id}")
def ticket(ticket_id: str):
    t = db.get_ticket(ticket_id)
    if not t:
        raise HTTPException(404, "not found")
    return t


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
