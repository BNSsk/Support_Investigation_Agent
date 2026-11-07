"""Tools the investigation agent may call. The LLM decides if/when/which to call.

State-changing tools return a LangGraph `Command` so their effect (new evidence, a changed
classification, an escalation request) lands directly in the graph state.
"""
from __future__ import annotations

import json
import re
from typing import Annotated, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt

from . import db, labels, memory
from .config import settings
from .vectorstore import get_index


def _clean(v):
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in ("", "null", "none", "any", "n/a", "all"):
        return None
    return v.strip() if isinstance(v, str) else v


def lead_knowledge(guidance: dict | None) -> str:
    """A support lead's diagnosis as one evidence text: root cause + verified solution."""
    g = guidance or {}
    parts = []
    if g.get("root_cause"):
        parts.append(f"Root cause (support lead): {g['root_cause']}")
    if g.get("solution"):
        parts.append(f"Verified solution (support lead): {g['solution']}")
    return "\n".join(parts)


LEAD_MARK = "[SUPPORT-LEAD VERIFIED]\n"
REPLY_MARK = "\nReply sent: "


def knowledge_answer(resolution: str | None, guidance: dict | None) -> str:
    """What a finished case teaches future tickets. A lead's diagnosis comes first (it is often knowledge no
    historical case has), then the reply sent. clip_answer() never cuts the lead's part."""
    lead = lead_knowledge(guidance)
    if not lead:
        return resolution or ""
    return f"{LEAD_MARK}{lead}" + (f"{REPLY_MARK}{resolution}" if resolution else "")


def clip_answer(answer: str | None, n: int) -> str:
    """Shorten a past resolution to n characters for a prompt. The support lead's verified block of a
    lead-verified case (root cause + solution) is NEVER cut: it is kept whole and only the reply text after it
    is shortened to n characters - otherwise a long diagnosis pushes the lead's later points out of view."""
    a = answer or ""
    if not a.startswith(LEAD_MARK):
        return a[:n]
    lead, sep, reply = a.partition(REPLY_MARK)
    return lead + (sep + reply[:n] if sep else "")


def evidence_item(h: dict, retrieved_by: str) -> dict:
    return {
        "case_id": h.get("case_id") or h.get("id"),
        "subject": h.get("subject"),
        "body": (h.get("body") or "")[:900],
        "answer": clip_answer(h.get("answer") or h.get("resolution"), 1100),
        "queue": h.get("queue"), "type": h.get("type"), "priority": h.get("priority"),
        "tags": h.get("tags") or [],
        "source": h.get("source", "historical"),
        "lead_verified": bool(h.get("lead_verified")),
        "similarity": h.get("similarity"),
        "retrieved_by": [retrieved_by],
    }


def compact(h: dict) -> dict:
    sim = h.get("similarity")
    out = {
        "id": h.get("case_id"), "sim": round(sim, 2) if sim is not None else None,
        "labels": f"{h.get('queue')}/{h.get('type')}/{h.get('priority')}",
        "tags": (h.get("tags") or [])[:4], "subject": h.get("subject"),
        "issue": (h.get("body") or "")[:180],
        "resolution": clip_answer(h.get("answer"), 600 if h.get("lead_verified") else 280),
    }
    if h.get("source") == "live":
        out["new_resolved_case"] = True
    if h.get("lead_verified"):
        out["lead_verified"] = True
    return out


@tool
def search_similar_tickets(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    queue: Optional[str] = None,
    ticket_type: Optional[str] = None,
    priority: Optional[str] = None,
    tag: Optional[str] = None,
    k: int = 4,
) -> Command:
    """Hybrid search over historical and resolved support cases. Use a TARGETED query (symptom, error,
    product or request type), optionally filtered by queue, ticket_type, priority or one tag. Returns
    only cases not already in your evidence pool."""
    queue = labels.normalize("queue", _clean(queue))
    ticket_type = labels.normalize("type", _clean(ticket_type))
    priority = labels.normalize("priority", _clean(priority))
    tag = _clean(tag)
    k = max(1, min(int(k or 4), 6))
    known = list((state.get("evidence") or {}).keys())  # already in the prompt -> don't send them again
    hits = get_index().search(query, k=k, queue=queue, ticket_type=ticket_type, priority=priority, tag=tag,
                              exclude_ids=known or None)
    if not hits and (queue or ticket_type or priority or tag):  # filters too strict -> relax once
        hits = get_index().search(query, k=k, exclude_ids=known or None)
    label = f"tool:search '{query[:60]}'"
    payload = {"query": query, "filters": {"queue": queue, "type": ticket_type, "priority": priority, "tag": tag},
               "new_cases": [compact(h) for h in hits],
               "note": f"{len(known)} cases already in the evidence pool were excluded" if known else ""}
    return Command(update={
        "evidence": {h["case_id"]: evidence_item(h, label) for h in hits},
        "retrieval_log": [{"by": "agent_tool", "query": query, "filters": payload["filters"], "n": len(hits),
                           "top_similarity": hits[0]["similarity"] if hits else None}],
        "messages": [ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id=tool_call_id)],
    })


@tool
def get_ticket_by_id(ticket_id: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Full record (complete issue and resolution) of a case id (HT-..., CASE-... or LEAD-CASE-...). Use it
    before relying on a case whose resolution is cut short in the evidence pool."""
    ticket_id = (ticket_id or "").strip()
    if ticket_id.upper().startswith("LEAD-"):  # a support lead's diagnosis for a case
        case = db.get_case(ticket_id[5:]) or {}
        g = case.get("lead_guidance")
        content = (json.dumps({"id": ticket_id, "source": "lead", "case_id": case.get("case_id"),
                               "diagnosis": lead_knowledge(g)}, ensure_ascii=False)
                   if g else f"No support-lead diagnosis found for {ticket_id}")
        return Command(update={"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]})
    t = db.get_ticket(ticket_id)
    if not t:
        return Command(update={"messages": [ToolMessage(content=f"No ticket found with id {ticket_id}",
                                                        tool_call_id=tool_call_id)]})
    if t["source"] == "live":
        g = t.get("lead_guidance")
        t = {**t, "id": t["case_id"], "answer": knowledge_answer(t.get("resolution"), g), "lead_verified": bool(g),
             **{k: (t.get("classification") or {}).get(k) for k in ("queue", "type", "priority", "tags")}}
    full = {"id": t.get("id"), "source": t["source"], "status": t.get("status", "resolved"),
            "queue": t.get("queue"), "type": t.get("type"), "priority": t.get("priority"), "tags": t.get("tags"),
            "subject": t.get("subject"), "issue": (t.get("body") or "")[:1000],
            "resolution": clip_answer(t.get("answer"), 1400)}
    if t.get("lead_verified"):
        full["lead_verified"] = True
    item = evidence_item({**t, "case_id": t.get("id")}, "tool:get_ticket_by_id")
    item["similarity"] = None
    return Command(update={
        "evidence": {full["id"]: item},
        "messages": [ToolMessage(content=json.dumps(full, ensure_ascii=False), tool_call_id=tool_call_id)],
    })


@tool
def get_customer_history(
    state: Annotated[dict, InjectedState],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """This customer's previous/open cases and remembered facts. Use it to detect repeat issues."""
    cid = state.get("customer_id") or "anonymous"
    cases = db.customer_cases(cid, exclude_case=state.get("case_id"), limit=8)
    return json.dumps({
        "customer_id": cid,
        "remembered_facts": [{"key": f["key"], "value": f.get("value")} for f in memory.get_facts(store, cid)],
        "cases": [{"case_id": c["case_id"], "status": c["status"], "subject": c["subject"],
                   "classification": c.get("classification"), "resolution": (c.get("resolution") or "")[:300],
                   "created_at": c["created_at"]} for c in cases],
    }, ensure_ascii=False)


@tool
def get_category_statistics(queue: Optional[str] = None, tag: Optional[str] = None,
                            ticket_type: Optional[str] = None) -> str:
    """Volume, type/priority distribution and top tags for a queue, tag and/or type. Use it to
    sanity-check a classification."""
    stats = db.category_stats(queue=labels.normalize("queue", _clean(queue)), tag=_clean(tag),
                              ticket_type=labels.normalize("type", _clean(ticket_type)))
    return json.dumps(stats, ensure_ascii=False)


@tool
def update_classification(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    queue: Optional[str] = None,
    ticket_type: Optional[str] = None,
    priority: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> Command:
    """Correct the classification when evidence contradicts triage. Pass only changed fields and a
    short reason."""
    current = dict(state.get("classification") or {})
    changes = {}
    for kind, val in (("queue", queue), ("type", ticket_type), ("priority", priority)):
        norm = labels.normalize(kind, _clean(val))
        if norm and norm != current.get(kind):
            changes[kind] = {"from": current.get(kind), "to": norm}
            current[kind] = norm
    if tags:
        changes["tags"] = {"from": current.get("tags"), "to": tags[:5]}
        current["tags"] = tags[:5]
    current["updated_by"] = "agent:update_classification"
    current["update_reason"] = reason
    msg = {"updated": bool(changes), "changes": changes, "classification": current}
    return Command(update={
        "classification": current,
        "messages": [ToolMessage(content=json.dumps(msg), tool_call_id=tool_call_id)],
    })


@tool
def escalate_ticket(reason: str, target_team: str, tool_call_id: Annotated[str, InjectedToolCallId],
                    urgency: str = "high") -> Command:
    """Flag the case for a specialist team (target_team = a queue name): security/data incidents,
    multi-user outages, legal/HR, repeated unresolved issues, or no safe resolution in the evidence."""
    req = {"reason": reason, "target_team": labels.normalize("queue", target_team) or target_team,
           "urgency": urgency}
    return Command(update={
        "escalation_requested": req,
        "messages": [ToolMessage(content=json.dumps({"escalation_flagged": True, **req}),
                                 tool_call_id=tool_call_id)],
    })


# ---- clarification guards: no fixed number of questions, but irrelevant ones are not sent -----------------
_ASK_STOP = {
    "could", "would", "please", "kindly", "provide", "share", "tell", "let", "know", "send", "specify", "confirm",
    "give", "you", "your", "yours", "the", "and", "any", "are", "can", "for", "with", "that", "this", "these",
    "those", "from", "have", "has", "had", "what", "which", "when", "where", "how", "who", "see", "also", "well",
    "about", "more", "details", "detail", "information", "info", "help", "our", "ours", "able", "there", "they",
    "them", "was", "were", "been", "being", "did", "does", "into", "just", "some", "may", "might", "need",
    "exact", "exactly", "currently", "seeing", "experiencing", "issue", "problem", "case", "ticket", "investigate",
}
_NO_MORE_INFO = re.compile(
    r"\b(?:don'?t|do not|doesn'?t|does not|can'?t|cannot|unable to)\s+(?:have|know|provide|share|find|remember)\b"
    r"|\bno (?:more|further|other|additional) (?:details|information|info)\b|\bnot sure\b|\bno idea\b"
    r"|\bplease proceed\b|^\s*(?:n/?a|none|no|nope|idk|unknown)\s*[.!]?\s*$", re.I)
_SECRET_REQUEST = re.compile(
    r"\b(?:provide|share|send|give|enter|type|confirm|tell)\s+(?:us\s+|me\s+)?(?:with\s+)?(?:your\s+|the\s+)?"
    r"(?:full\s+|complete\s+)?(?:password|passcode|pin|cvv|cvc|security code|card number|credit card number|"
    r"social security number|ssn|one[- ]time (?:code|password)|otp)\b"
    r"|\bwhat(?:'s| is) your (?:password|passcode|pin|card number|cvv)\b", re.I)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower()) if w not in _ASK_STOP}


def question_problem(question: str, clarifications: list[dict]) -> str | None:
    """Why a clarification question should NOT be sent to the customer (None = fine to ask).
    There is no limit on the number of questions; only these irrelevant ones are stopped."""
    if _SECRET_REQUEST.search(question or ""):
        return "it asks for a password, card number or other secret, which must never be requested"
    if clarifications:
        last = clarifications[-1].get("answer") or ""
        if len(last.split()) <= 18 and _NO_MORE_INFO.search(last):
            return f"the customer already said they have no more details ('{last[:90]}')"
    q = _content_words(question)
    for c in clarifications or []:
        p = _content_words(c.get("question", ""))
        if q and p and len(q & p) / min(len(q), len(p)) >= 0.6:
            return (f"it repeats an earlier question ('{c['question'][:90]}'), which the customer answered: "
                    f"'{(c.get('answer') or '')[:90]}'")
    return None


@tool
def ask_customer(question: str, tool_call_id: Annotated[str, InjectedToolCallId],
                 state: Annotated[dict, InjectedState]) -> Command:
    """Ask the customer ONE focused question; the case pauses until they reply. Call it alone, only when
    information only the customer has is missing and would change the resolution. Repeated questions,
    questions after the customer said they have no more details, and requests for secrets are rejected."""
    problem = question_problem(question, state.get("clarifications") or [])
    if problem:  # not sent; the round is consumed so the agent moves on
        return Command(update={"messages": [ToolMessage(
            content=f"Not asked: {problem}. Do not ask again; continue with the evidence you have "
                    "(missing details can still be requested in the reply).", tool_call_id=tool_call_id)]})
    case_id = state.get("case_id")
    db.update_case(case_id, status="awaiting_customer")
    # Human-in-the-loop: the graph is checkpointed here; resume with Command(resume={"answer": ...})
    answer = interrupt({"type": "clarification", "question": question, "case_id": case_id})
    text = (answer.get("answer") if isinstance(answer, dict) else str(answer or "")).strip() \
        or "(no additional information provided)"
    db.update_case(case_id, status="open")
    return Command(update={
        "clarifications": [{"question": question, "answer": text}],
        # asking does not consume investigation budget: the agent still needs a round to use the answer
        "tool_rounds": max((state.get("tool_rounds") or 1) - 1, 0),
        "messages": [ToolMessage(content=f"Customer answered: {text}", tool_call_id=tool_call_id)],
    })


TOOLS = [search_similar_tickets, get_ticket_by_id, get_customer_history, get_category_statistics,
         update_classification, escalate_ticket, ask_customer]
