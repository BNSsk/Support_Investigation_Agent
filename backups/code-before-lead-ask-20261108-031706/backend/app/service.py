"""Run / resume the graph and turn LangGraph stream updates into UI-friendly events.
Shared by the FastAPI server, the CLI, the demo script and the evaluators."""
from __future__ import annotations

import time
import uuid
from typing import Any, Iterator

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel

from . import db
from .graph import get_graph
from .nodes import review_update


def jsonable(x: Any, depth: int = 0) -> Any:
    if depth > 8:
        return str(x)
    if isinstance(x, BaseMessage):
        d = {"type": x.type, "content": x.content if isinstance(x.content, str) else str(x.content)}
        if isinstance(x, AIMessage) and x.tool_calls:
            d["tool_calls"] = [{"name": t["name"], "args": t["args"]} for t in x.tool_calls]
        if isinstance(x, ToolMessage):
            d["tool"] = x.name
        return d
    if isinstance(x, BaseModel):
        return x.model_dump()
    if isinstance(x, Command):
        return jsonable(x.update, depth + 1)
    if isinstance(x, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v, depth + 1) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _merge_updates(upd: Any) -> dict:
    """ToolNode may emit a list of updates (one per Command); merge them for display."""
    if isinstance(upd, dict):
        return upd
    out: dict = {}
    for u in upd or []:
        u = u.update if isinstance(u, Command) else u
        if not isinstance(u, dict):
            continue
        for k, v in u.items():
            if isinstance(v, list):
                out.setdefault(k, []).extend(v)
            elif isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k].update(v)
            else:
                out[k] = v
    return out


def summarize(node: str, u: dict) -> str:
    try:
        if node == "intake":
            head = (f"Case {u['case_id']} persisted as OPEN" if u.get("case_id") else
                    "Follow-up turn: case re-opened, previous evidence + classification reused (no re-retrieval)")
            return (f"{head}; loaded {len(u.get('customer_memory') or [])} long-term fact(s), "
                    f"{len(u.get('customer_history') or [])} previous case(s)")
        if node == "retrieve":
            return " | ".join(f"{l['by']}: {l['n']} (top sim {l.get('top_similarity')})"
                              for l in u.get("retrieval_log") or [])
        if node == "triage":
            c = u.get("classification") or {}
            return (f"{c.get('queue')} / {c.get('type')} / {c.get('priority')}  tags={c.get('tags')}  "
                    f"skills={u.get('skills')}")
        if node == "investigate":
            m = (u.get("messages") or [None])[-1]
            if isinstance(m, AIMessage) and m.tool_calls:
                return "Agent chose: " + ", ".join(t["name"] for t in m.tool_calls)
            return "Investigation notes written -> respond"
        if node == "tools":
            if u.get("clarifications"):
                c = u["clarifications"][0]
                return f"Customer answered '{c['question'][:80]}': {c['answer'][:120]}"
            msgs = [m for m in u.get("messages") or [] if isinstance(m, ToolMessage)]
            skipped = [str(m.content) for m in msgs if str(m.content).startswith("Not asked:")]
            if skipped:
                return "ask_customer not sent - " + skipped[0][len("Not asked: "):].split(". Do not ask")[0][:160]
            extra = f" (+{len(u.get('evidence') or {})} evidence)" if u.get("evidence") else ""
            esc = " - ESCALATION FLAGGED" if u.get("escalation_requested") else ""
            return "Executed: " + ", ".join(m.name or "tool" for m in msgs) + extra + esc
        if node == "respond":
            d = u.get("draft") or {}
            if d.get("escalation_required"):
                return f"Escalation prepared -> {d.get('target_team')} ({d.get('urgency')}), citing {len(d.get('evidence_used') or [])} case(s)"
            return f"Resolution drafted citing {len(d.get('evidence_used') or [])} case(s)"
        if node == "evaluate":
            e = u.get("evaluation") or {}
            verdict = "PASSED" if e.get("passed") else "FAILED"
            nxt = " -> escalating (retries exhausted)" if u.get("escalation_requested") else \
                (" -> re-investigating" if not e.get("passed") and u.get("messages") else "")
            return f"Evidence check {verdict}: groundedness={e.get('groundedness')} confidence={e.get('confidence')}{nxt}"
        if node == "review":
            r = u.get("review") or {}
            if r.get("action") == "amend":
                return f"Review: support-lead diagnosis applied by {r.get('by')} -> rewriting the reply from it"
            if r.get("action") == "reject" and u.get("lead_guidance"):
                return f"Review: reject by {r.get('by')} (diagnosis recorded) -> re-investigating"
            return f"Review: {r.get('action')} by {r.get('by')}" + (" (note saved)" if r.get("note") else "")
        if node == "finalize":
            r = (u.get("final") or {}).get("index") or {}
            idx = (("indexed as LEAD-VERIFIED case" if r.get("lead_verified") else "indexed as trusted case")
                   + " for future retrieval") if r.get("indexed") else f"not indexed ({r.get('reason')})"
            return f"Case {(u.get('final') or {}).get('status', '').upper()} and persisted; {idx}"
        if node == "update_memory":
            m = u.get("memory_update") or {}
            if m.get("error"):
                return f"Memory update skipped: {m['error'][:80]}"
            return f"Long-term memory: saved {len(m.get('saved') or [])}, removed {len(m.get('removed') or [])}"
    except Exception:  # noqa: BLE001
        pass
    return node


def enrich_final(final: dict | None) -> dict | None:
    """Make sure a final result carries the customer's ticket and each evidence case's subject + description.
    New runs store them; cases finished before that are filled in from the database for the UI."""
    if not isinstance(final, dict):
        return final
    out = dict(final)
    if not out.get("ticket") and out.get("case_id"):
        c = db.get_case(out["case_id"]) or {}
        out["ticket"] = {"subject": c.get("subject"), "body": c.get("body")}
    evidence = []
    for e in out.get("evidence") or []:
        e = dict(e)
        if not e.get("issue") and e.get("case_id"):
            t = db.get_ticket(e["case_id"]) or {}
            e["subject"] = e.get("subject") or t.get("subject")
            e["issue"] = (t.get("body") or "")[:900]
            e["resolution"] = e.get("resolution") or (t.get("answer") or t.get("resolution") or "")[:1100]
        evidence.append(e)
    out["evidence"] = evidence
    return out


def config_for(thread_id: str, auto_accept: bool = False) -> dict:
    return {"configurable": {"thread_id": thread_id, "auto_accept": auto_accept}, "recursion_limit": 80}


def _pending_interrupt(snap) -> dict | None:
    intr = getattr(snap, "interrupts", None) or ()
    if not intr:
        for task in snap.tasks or ():
            if task.interrupts:
                intr = task.interrupts
                break
    return intr[0].value if intr else None


def run_stream(thread_id: str, payload: Any, auto_accept: bool = False) -> Iterator[dict]:
    """Stream events and persist them so a thread's workflow can be replayed later."""
    for ev in _run_stream(thread_id, payload, auto_accept):
        try:
            db.log_event(thread_id, ev)
        except Exception:  # noqa: BLE001 - logging must never break a run
            pass
        yield ev


def _run_stream(thread_id: str, payload: Any, auto_accept: bool = False) -> Iterator[dict]:
    graph = get_graph()
    cfg = config_for(thread_id, auto_accept)
    yield {"type": "run_started", "thread_id": thread_id, "ts": time.time()}
    try:
        for mode, chunk in graph.stream(payload, cfg, stream_mode=["updates", "custom"]):
            if mode == "custom":  # LLM call accounting + rate-limit waits emitted from app.llm
                if isinstance(chunk, dict):
                    yield {"type": chunk.get("kind", "custom"), **chunk, "ts": time.time()}
                continue
            for node, upd in chunk.items():
                if node == "__interrupt__":
                    continue
                u = _merge_updates(upd)
                yield {"type": "node", "node": node, "summary": summarize(node, u), "data": jsonable(u),
                       "ts": time.time()}
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"[:1500], "thread_id": thread_id}
        return
    snap = graph.get_state(cfg)
    pending = _pending_interrupt(snap)
    if pending is not None:
        yield {"type": "interrupt", "thread_id": thread_id, "interrupt": jsonable(pending), "next": list(snap.next)}
    else:
        yield {"type": "completed", "thread_id": thread_id, "final": enrich_final(jsonable(snap.values.get("final")))}


def new_ticket(customer_id: str, subject: str, body: str, auto_accept: bool = False,
               thread_id: str | None = None) -> tuple[str, Iterator[dict]]:
    thread_id = thread_id or f"thread-{uuid.uuid4().hex[:10]}"
    payload = {"ticket": {"subject": subject.strip(), "body": body.strip()},
               "customer_id": (customer_id or "anonymous").strip()}
    return thread_id, run_stream(thread_id, payload, auto_accept)


def resume(thread_id: str, value: Any, auto_accept: bool = False) -> Iterator[dict]:
    return run_stream(thread_id, Command(resume=value), auto_accept)


def followup(thread_id: str, message: str, auto_accept: bool = False) -> Iterator[dict]:
    pending = _pending_interrupt(get_graph().get_state(config_for(thread_id)))
    if isinstance(pending, dict) and pending.get("type") == "clarification":
        # a question is still open: this message answers it (starting a new turn would orphan the tool call)
        return resume(thread_id, {"answer": message.strip()}, auto_accept)
    return run_stream(thread_id, {"followup": message.strip()}, auto_accept)


def retry(thread_id: str) -> Iterator[dict]:
    """Re-run from the last checkpoint (e.g. after a transient LLM/rate-limit error)."""
    return run_stream(thread_id, None)


def lead_solution(thread_id: str, root_cause: str = "", solution: str = "", note: str = "",
                  by: str = "support_lead", auto_accept: bool = False) -> Iterator[dict]:
    """A support lead's root cause + solution for a case.
    - paused at review: same as resuming with action 'amend';
    - already finished (e.g. escalated, or resolved with a weaker answer): the decision is applied as if the
      review node had made it, and the run continues at `respond` -> evidence check -> review -> finalize, so
      the customer reply is rewritten from the lead's fix and the case is indexed as lead-verified.
    Raises LookupError (unknown thread) / ValueError (case is busy) before anything streams."""
    decision = {"action": "amend", "root_cause": root_cause, "solution": solution, "feedback": note, "by": by}
    graph = get_graph()
    cfg = config_for(thread_id, auto_accept)
    snap = graph.get_state(cfg)
    pending = _pending_interrupt(snap)
    if isinstance(pending, dict) and pending.get("type") == "review":
        return resume(thread_id, decision, auto_accept)
    v = snap.values or {}
    if not v.get("case_id"):
        raise LookupError(f"no case on thread {thread_id}")
    if pending is not None or snap.next:
        raise ValueError("the case is waiting for a customer answer or still running; try again when it stops")
    # explicit request on a finished case: apply even if the same diagnosis was recorded earlier
    graph.update_state(cfg, review_update(v, decision, force_amend=True), as_node="review")
    db.update_case(v["case_id"], status="open")
    return run_stream(thread_id, None, auto_accept)


def thread_snapshot(thread_id: str) -> dict:
    graph = get_graph()
    snap = graph.get_state(config_for(thread_id))
    v = snap.values or {}
    return {
        "thread_id": thread_id,
        "next": list(snap.next),
        "interrupt": jsonable(_pending_interrupt(snap)),
        "values": {**jsonable({k: v.get(k) for k in (
            "case_id", "customer_id", "ticket", "classification", "skills", "evidence", "retrieval_log",
            "customer_memory", "customer_history", "clarifications", "draft", "evaluation",
            "review", "lead_guidance", "memory_update", "investigation_notes", "messages")}),
            "final": enrich_final(jsonable(v.get("final")))},
        "case": db.get_case(v["case_id"]) if v.get("case_id") else None,
        # replayed "completed" events of older cases get the ticket + evidence descriptions filled in
        "events": [{**ev, "final": enrich_final(ev.get("final"))} if ev.get("type") == "completed" else ev
                   for ev in db.thread_events(thread_id)],
    }
