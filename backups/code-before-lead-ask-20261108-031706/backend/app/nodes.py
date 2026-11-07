"""LangGraph node implementations (10-node graph, see graph.py)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from . import db, labels, memory, skills
from .config import settings
from .llm import invoke_chat, is_api_error, structured
from .state import AgentState
from .tools import TOOLS, clip_answer, evidence_item, knowledge_answer, lead_knowledge
from .vectorstore import get_index

# =============================================================== helpers

MEMORY_QUERY_KEYS = ("product", "environment", "integration", "platform", "system", "recurring_issue", "plan")


def ticket_text(state: AgentState) -> str:
    t = state.get("ticket") or {}
    parts = [f"Subject: {t.get('subject') or '(none)'}", "", t.get("body") or ""]
    for c in state.get("clarifications") or []:
        parts.append(f"\n[Clarification] Q: {c['question']}\nA: {c['answer']}")
    for m in state.get("messages") or []:
        if isinstance(m, HumanMessage) and str(m.content).startswith("Customer follow-up:"):
            parts.append(f"\n[{m.content}]")
    return "\n".join(parts).strip()


def search_query(state: AgentState) -> str:
    """Ticket + clarifications + follow-up, enriched with remembered product/environment facts
    (so a vague 'it is broken again' still retrieves cases about the customer's actual system)."""
    t = state.get("ticket") or {}
    extra = " ".join(c["answer"] for c in state.get("clarifications") or [])
    if state.get("followup"):
        extra += " " + state["followup"]
    facts = [f.get("value", "") for f in state.get("customer_memory") or []
             if any(k in f.get("key", "") for k in MEMORY_QUERY_KEYS)]
    return f"{t.get('subject') or ''}\n{t.get('body') or ''}\n{extra}\n{' '.join(facts[:3])}".strip()


CASE_ID_RE = re.compile(r"\b(?:HT-\d{3,6}|(?:LEAD-)?CASE-\d{6}-[0-9A-Fa-f]{6})\b")
_HYPHENS = re.compile(r"[\u2010-\u2015\u2212\ufe58\ufe63\uff0d]")  # gpt-oss often writes U+2011 in ids


def norm_id(text: str) -> str:
    """Normalise look-alike hyphens so 'CASE\u2011261104\u2011ABC123' matches the pool key 'CASE-261104-ABC123'."""
    return _HYPHENS.sub("-", text or "").strip()


# ---- support-lead knowledge ------------------------------------------------------------------------------
# A lead's root cause + solution is authoritative evidence for its case (pool item LEAD-<case_id>), and an
# accepted case that carries one is indexed as "lead-verified" so similar future tickets prefer it.
LEAD_SOURCE = "lead"
LEAD_VERIFIED_BOOST = 0.10   # ranking bonus for lead-verified cases over equally similar historical ones
# "(not the customer)": with the old wording the model read the lead's point as customer feedback and argued it away
REJECT_PREFIX = "The support lead (not the customer) rejected the draft. Lead's feedback: "
LEGACY_REJECT_PREFIXES = ("Support lead rejected the draft: ",)  # threads saved before the wording changed
REJECT_SUFFIX = "\n\nRe-investigate and address this feedback."


def lead_id(case_id: str | None) -> str:
    return f"LEAD-{case_id}"


def lead_feedback(state: AgentState) -> list[str]:
    """Every rejection comment the lead wrote on this case (they build on each other, e.g. 'also ask X')."""
    out = []
    for m in state.get("messages") or []:
        c = str(m.content) if isinstance(m, HumanMessage) else ""
        prefix = next((p for p in (REJECT_PREFIX, *LEGACY_REJECT_PREFIXES) if c.startswith(p)), None)
        if prefix:
            out.append(c[len(prefix):].split(REJECT_SUFFIX)[0].removesuffix(". Re-investigate.").strip())
    return [x for x in out if x]


def lead_digest(state: AgentState) -> str:
    """Everything the support lead said about this case, handed to respond and the judge verbatim."""
    parts = []
    g = state.get("lead_guidance") or {}
    if lead_knowledge(g):
        parts.append(f"[{lead_id(state.get('case_id'))}] diagnosis:\n{lead_knowledge(g)}"
                     + (f"\nLead's note: {g['note']}" if g.get("note") else ""))
    parts += [f"Review feedback on an earlier draft: {fb}" for fb in lead_feedback(state)]
    return "\n".join(parts)


def _rank_score(e: dict) -> float:
    if e.get("source") == LEAD_SOURCE:  # the lead's diagnosis of THIS case always comes first
        return 9.0
    # explicitly opened records (similarity None) are relevant by construction
    sim = e.get("similarity") if e.get("similarity") is not None else 0.75
    return sim + (LEAD_VERIFIED_BOOST if e.get("lead_verified") else 0.0)


def ranked_evidence(state: AgentState, limit: int = 10, exclude_tool: bool = False,
                    prefer: list[str] | tuple = ()) -> list[dict]:
    ev = list((state.get("evidence") or {}).values())
    if exclude_tool:  # evidence gathered by tools is already visible in the ToolMessages
        ev = [e for e in ev if not all(str(b).startswith("tool:") for b in e.get("retrieved_by") or [])]
    ranked = sorted(ev, key=_rank_score, reverse=True)
    if prefer:  # e.g. the cases the investigation notes rely on come first (the lead's diagnosis stays on top)
        first = [e for e in ranked if e.get("source") == LEAD_SOURCE or e["case_id"] in prefer]
        ranked = first + [e for e in ranked if e not in first]
    return ranked[:limit]


def evidence_digest(state: AgentState, limit: int = 5, answer_chars: int = 280, exclude_tool: bool = False,
                    prefer: list[str] | tuple = ()) -> str:
    rows = []
    for e in ranked_evidence(state, limit, exclude_tool, prefer):
        if e.get("source") == LEAD_SOURCE:
            rows.append(f"[{e['case_id']}] (SUPPORT-LEAD DIAGNOSIS of this case - authoritative)\n  "
                        + (e.get("answer") or "").replace("\n", "\n  "))
            continue
        sim = f"{e['similarity']:.2f}" if e.get("similarity") is not None else "opened"
        src = ("LEAD-VERIFIED CASE" if e.get("lead_verified") else
               "NEW-RESOLVED-CASE" if e.get("source") == "live" else "historical")
        # a lead-verified case keeps its whole lead block (clip_answer); its reply text gets a larger budget
        chars = max(answer_chars, 600) if e.get("lead_verified") else answer_chars
        rows.append(
            f"[{e['case_id']}] ({src}, sim={sim}) {e.get('queue')}/{e.get('type')}/{e.get('priority')} "
            f"tags={','.join((e.get('tags') or [])[:4])}\n  Issue: {(e.get('subject') or '')[:90]} - "
            f"{(e.get('body') or '')[:160]}\n  Past resolution: {clip_answer(e.get('answer'), chars)}")
    return "\n".join(rows) or "(no evidence yet)"


def pool_stats(state: AgentState) -> str:
    """One line that tells the agent whether the existing evidence is strong enough to skip a search."""
    ev = state.get("evidence") or {}
    lead = ("Support-lead diagnosis present: build on it; search only for details it leaves open. "
            if any(e.get("source") == LEAD_SOURCE for e in ev.values()) else "")
    sims = sorted((e["similarity"] for e in ev.values() if e.get("similarity") is not None), reverse=True)
    if not sims:
        return lead + "Pool: empty - search first."
    q = (state.get("classification") or {}).get("queue")
    top = ranked_evidence(state, 5)
    agree = sum(1 for e in top if e.get("queue") == q)
    verified = sum(1 for e in ev.values() if e.get("lead_verified"))
    return (f"{lead}Pool: {len(ev)} cases, top similarity {sims[0]:.2f}, {agree}/{len(top)} of the best share "
            f"queue '{q}'" + (f", {verified} lead-verified." if verified else "."))


def label_digest(state: AgentState, limit: int = 6) -> str:
    """Labels + subject only: all triage needs from similar cases (resolutions are not used there)."""
    rows = [f"[{e['case_id']}] {e.get('queue')}/{e.get('type')}/{e.get('priority')} "
            f"tags={','.join((e.get('tags') or [])[:4])} | {(e.get('subject') or '')[:90]}"
            for e in ranked_evidence(state, limit)]
    return "\n".join(rows) or "(no similar cases)"


def votes_str(votes: dict | None) -> str:
    """Similarity-weighted label shares as compact text (JSON with decimals costs ~2x the tokens)."""
    return "\n".join(f"{kind}: " + ", ".join(f"{k} {round(v * 100)}%" for k, v in dist.items())
                     for kind, dist in (votes or {}).items()) or "(none)"


def memory_digest(state: AgentState) -> str:
    facts = state.get("customer_memory") or []
    return "\n".join(f"- {f['key']}: {f.get('value')}" for f in facts) or "(nothing remembered)"


def history_digest(state: AgentState) -> str:
    rows = []
    for h in (state.get("customer_history") or [])[:5]:
        cl = h.get("classification") or {}
        rows.append(f"- {h['case_id']} [{h['status']}] {h.get('subject')} ({cl.get('queue', '?')}) "
                    f"-> {(h.get('resolution') or 'no resolution yet')[:140]}")
    return "\n".join(rows) or "(no previous cases)"


def classification_str(c: dict | None) -> str:
    if not c:
        return "(not classified yet)"
    return (f"queue={c.get('queue')}, type={c.get('type')}, priority={c.get('priority')}, "
            f"tags={c.get('tags')}")


def recent_messages(messages: list, keep: int = 40, max_chars: int = 60000) -> list:
    """Opening ticket + most recent messages (a generous bound for very long threads),
    never starting the tail with an orphaned ToolMessage."""
    if not messages:
        return []
    head, rest = messages[0], list(messages[1:])[-keep:]
    budget = max_chars - len(str(head.content))
    tail: list = []
    for m in reversed(rest):
        size = len(str(m.content)) + (len(str(getattr(m, "tool_calls", ""))) if isinstance(m, AIMessage) else 0)
        if size > budget and tail:
            break
        tail.insert(0, m)
        budget -= size
    while tail and isinstance(tail[0], ToolMessage):
        tail.pop(0)
    return [head] + tail


def clamp(x, lo=0.0, hi=1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return lo


def history_as_evidence(state: AgentState) -> dict:
    """Customer-history cases in evidence-pool shape, so they can be cited and checked."""
    out = {}
    for h in state.get("customer_history") or []:
        cl = h.get("classification") or {}
        out[h["case_id"]] = {"case_id": h["case_id"], "subject": h.get("subject"), "body": "",
                             "answer": h.get("resolution") or f"(previous case, status: {h.get('status')})",
                             "queue": cl.get("queue"), "type": cl.get("type"), "priority": cl.get("priority"),
                             "source": "customer_history", "similarity": None}
    return out


def needs_escalation_skill(state: AgentState) -> bool:
    c = state.get("classification") or {}
    return bool(state.get("escalation_requested") or c.get("priority") == "high"
                or (state.get("customer_history") and c.get("type") in ("Incident", "Problem")))


# =============================================================== schemas


class TriageOut(BaseModel):
    queue: str = Field(description="One of the valid queues")
    type: str = Field(description="Incident, Request, Problem or Change")
    priority: str = Field(description="high, medium or low")
    tags: list[str] = Field(description="2-5 short tags")
    rationale: str = Field(description="1-2 sentences: why, incl. how similar cases influenced it")
    skills: list[str] = Field(description="1-2 investigation skill names from the catalog")


class EvidenceUse(BaseModel):
    case_id: str
    how_used: str = Field(description="What this case contributes to the answer")


class ResponseOut(BaseModel):
    analysis: str = Field(description="Concise reasoning: what several cases agree on, conflicts, why this approach")
    conflicts: list[str] = Field(default_factory=list, description="Unresolved contradictions between cases")
    resolution: str = Field(description="Customer-facing reply (a holding reply if escalating)")
    internal_actions: list[str] = Field(default_factory=list, description="Actions for the support team")
    evidence_used: list[EvidenceUse]
    escalation_required: bool = False
    escalation_reason: str = ""
    target_team: str = Field(default="", description="Queue name of the team taking over, if escalating")
    urgency: str = Field(default="", description="low/medium/high/critical, if escalating")
    handoff_summary: str = Field(default="", description="Internal hand-off note, if escalating")
    self_confidence: float = Field(description="0-1: how well the evidence supports this answer")


class JudgeOut(BaseModel):
    grounded: bool
    groundedness: float = Field(description="0-1 share of substantive claims supported by evidence/customer info")
    addresses_issue: float = Field(description="0-1 how well the reply addresses the customer's actual problem")
    unsupported_claims: list[str] = Field(default_factory=list)
    feedback: str = Field(description="Concrete fix instructions if not grounded, else short praise")
    suggested_search: str = Field(default="", description="Query that could find missing evidence")


class MemoryFact(BaseModel):
    key: str = Field(description="snake_case category, e.g. product, environment, plan, preference, recurring_issue")
    value: str


class MemoryOut(BaseModel):
    facts: list[MemoryFact] = Field(default_factory=list, description="New or updated durable facts (max 4)")
    forget: list[str] = Field(default_factory=list, description="Keys of facts that are now wrong/obsolete")
    reasoning: str = ""


# =============================================================== 1. intake


def intake(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Persist the ticket as an OPEN case (or re-open it for a follow-up turn) and load the customer's
    long-term memory + case history. Per-turn fields are reset; short-term memory is kept."""
    thread_id = config["configurable"]["thread_id"]
    cid = state.get("customer_id") or "anonymous"
    out = {"draft": None, "evaluation": None, "tool_rounds": 0, "review": None, "final": None,
           "escalation_requested": None, "investigation_notes": "", "memory_update": None}
    case_id = state.get("case_id")
    if not case_id:
        t = state["ticket"]
        case_id = db.create_case(thread_id, cid, t.get("subject", ""), t.get("body", ""))
        out.update(case_id=case_id, customer_id=cid, followup=None, messages=[HumanMessage(
            content=f"NEW SUPPORT TICKET (case {case_id})\nSubject: {t.get('subject')}\n\n{t.get('body')}")])
    else:  # follow-up turn on the same thread
        db.update_case(case_id, status="open")
        out["messages"] = [HumanMessage(content=f"Customer follow-up: {state.get('followup')}")]
    out["customer_memory"] = memory.get_facts(store, cid)
    out["customer_history"] = [
        {k: h.get(k) for k in ("case_id", "subject", "status", "classification", "resolution", "created_at")}
        for h in db.customer_cases(cid, exclude_case=case_id, limit=5)]
    return out


def route_after_intake(state: AgentState) -> Literal["retrieve", "investigate"]:
    """New ticket -> retrieve + triage. Follow-up turn on a case whose evidence pool and classification are
    already in state -> straight to investigate; the agent searches only if the follow-up needs new evidence."""
    return "investigate" if state.get("evidence") and state.get("classification") else "retrieve"


# =============================================================== 2. retrieve


def retrieve(state: AgentState) -> dict:
    """Broad hybrid retrieval + a dedicated look-up of trusted newly resolved cases (they are few and
    would otherwise be out-ranked by 23K historical near-duplicates)."""
    query = search_query(state)
    idx = get_index()
    hits = idx.search(query, k=settings.retrieval_k + 2)
    log = [{"by": "initial_retrieval", "query": query[:200], "filters": {}, "n": len(hits),
            "top_similarity": hits[0]["similarity"] if hits else None}]
    ev = {h["case_id"]: evidence_item(h, "initial_retrieval") for h in hits}
    if idx.count("live") > 0:  # cached count: no Qdrant request when there are no live cases
        live = [h for h in idx.search(query, k=3, source="live", exclude_ids=list(ev))
                if (h["similarity"] or 0) >= 0.6]
        ev.update({h["case_id"]: evidence_item(h, "trusted_new_cases") for h in live})
        log.append({"by": "trusted_new_cases", "query": query[:200], "filters": {"source": "live"}, "n": len(live),
                    "top_similarity": live[0]["similarity"] if live else None})
    return {"evidence": ev, "retrieval_log": log, "label_votes": labels.votes(hits)}


# =============================================================== 3. triage

TRIAGE_SYSTEM = """You are the triage step of a support investigation agent. Classify the ticket and pick the
investigation skill(s) that fit it.
- Type: Incident = something that worked is broken/degraded; Problem = recurring/root-cause issue or many
  users; Request = information, access, purchase, refund or new capability; Change = modify an existing
  configuration, plan, system or integration.
- Priority: high = outage, security/data risk, financial loss, blocked operations, many users;
  medium = degraded with workaround or one user blocked; low = questions, cosmetic, information.
- Use the label distribution of similar historical cases as a PRIOR, not as the answer; if the ticket
  clearly contradicts it, follow the ticket and say why.
- Tags: 2-5 short tags in the dataset style (e.g. Network, Security, Billing, Bug, Performance, Integration).
Valid queues: {queues}
Valid types: {types}
Valid priorities: {priorities}
Investigation skill catalog (pick 1-2 by name):
{catalog}"""


def triage(state: AgentState) -> dict:
    vocab = labels.vocabulary()
    system = TRIAGE_SYSTEM.format(queues=vocab["queue"], types=vocab["type"], priorities=vocab["priority"],
                                  catalog=skills.catalog("investigate"))
    human = (f"TICKET:\n{ticket_text(state)}\n\nREMEMBERED CUSTOMER FACTS:\n{memory_digest(state)}\n\n"
             f"LABELS OF SIMILAR CASES (similarity-weighted share):\n{votes_str(state.get('label_votes'))}\n\n"
             f"MOST SIMILAR CASES:\n{label_digest(state, 6)}")
    out = structured(TriageOut, [SystemMessage(content=system), HumanMessage(content=human)], fast=True)
    votes = state.get("label_votes") or {}
    top = lambda kind: next(iter(votes.get(kind) or {}), None)  # noqa: E731
    cls = {
        "queue": labels.normalize("queue", out.queue) or top("queue"),
        "type": labels.normalize("type", out.type) or top("type"),
        "priority": labels.normalize("priority", out.priority) or top("priority") or "medium",
        "tags": [t for t in out.tags if t][:5],
        "rationale": out.rationale,
        "updated_by": "triage",
    }
    # dynamic skill loading: only the chosen skills' full instructions enter later prompts
    valid = set(skills.names("investigate"))
    chosen = [s for s in out.skills if s in valid][:2] or skills.default_for(cls["type"], cls["queue"])
    if needs_escalation_skill({**state, "classification": cls}):
        chosen.append("escalation")
    return {"classification": cls, "skills": list(dict.fromkeys(chosen))}


# =============================================================== 4. investigate (+ ToolNode)

# Static instructions + skills first, the case block stays identical across tool rounds.
INVESTIGATE_SYSTEM = """You are an autonomous support investigation agent for a help desk.
Your ONLY domain knowledge is the historical support-case base (the evidence below + your tools).
Never rely on outside product knowledge.

## How to work
- Start from the evidence pool below: it was already retrieved for this ticket and is kept across turns.
  If its best cases match the problem and agree, do NOT search again - open the full record
  (get_ticket_by_id) of the 1-2 most useful cases if their resolution is cut short, or write your notes.
- search_similar_tickets is for when the pool is not enough: top similarity below ~0.75, cases about a
  different problem, conflicting resolutions, or a new detail (clarification / follow-up) the pool does
  not cover. Use a different, narrower query than the ticket text; it returns only new cases.
- Other tools when needed: customer history for repeats, category statistics if the classification is
  doubtful, update_classification if the evidence contradicts it. Prefer several calls in ONE round.
- Do NOT simply copy the nearest case. Compare resolutions across several cases, find the consensus
  and note conflicts or gaps.
- Support-lead knowledge outranks the historical base: a SUPPORT-LEAD DIAGNOSIS (LEAD-...) is the lead's
  verdict on THIS case, and LEAD-VERIFIED cases carry a lead's confirmed root cause and fix for a similar
  one. Build on them instead of replacing them with what older cases say. If the support lead rejected a
  draft, their feedback is a requirement: do what it asks, and treat any cause it names as the hypothesis.
- The lead's word needs no confirmation from other cases. Never dismiss, weaken or mark as optional a cause,
  check or step that the support lead gave (diagnosis, note or review feedback on this case, or the lead's
  root cause and solution in a LEAD-VERIFIED case) because no case supports it - list each one in your
  notes as a required part of the answer. Only the lead can make one of their steps optional.
  A lead point like "X might be the issue, check it too" adds a cause to check NEXT TO the diagnosis; it
  does not contradict it. Never claim the lead ruled something out unless the lead's own words say so.
- ask_customer: use it (ALONE in its round) only if information that ONLY the customer has is missing
  and it would change the resolution - e.g. which product/system is affected, the exact error, or
  which of several conflicting case resolutions applies. If the ticket does not say WHAT is affected
  (e.g. just "it stopped working"), ask FIRST, before searching. Don't ask for details the reply itself
  can request (account number, dates, logs), nor for anything already known from the ticket,
  clarifications or remembered facts. Never repeat a question, and if the customer says they don't
  have the information, proceed without it. Never ask for passwords or payment data.
- escalate_ticket: use it if escalation criteria apply (security/data incident, multi-user outage,
  legal/HR matter, repeated unresolved issue, no reliable resolution for a high-priority case).
- When you have enough, STOP calling tools and reply with short INVESTIGATION NOTES (max ~150 words):
  1) hypothesis / root cause or request type  2) supporting case ids and what they show
  3) conflicts or gaps in the evidence

## Active skills
{skills}

## Case
Case: {case_id} | Customer: {customer_id}
Classification: {classification}

## Support-lead input on this case (the lead's own words - every point is required)
{lead}

## Long-term customer memory
{memory}

## Customer history (previous / open cases)
{history}

## Evidence pool (already retrieved for this ticket; best first; tool results appear in the conversation)
{pool}
{evidence}
"""



def investigate(state: AgentState) -> dict:
    """ReAct step: the model decides whether to call tools (incl. asking the customer) or to conclude."""
    rounds = state.get("tool_rounds") or 0
    remaining = max(settings.max_tool_rounds - rounds, 0)
    history = recent_messages(state.get("messages") or [])
    system = INVESTIGATE_SYSTEM.format(
        skills=skills.render(state.get("skills") or [], brief=True) or "(none)",
        case_id=state.get("case_id"), customer_id=state.get("customer_id"),
        classification=classification_str(state.get("classification")),
        lead=lead_digest(state) or "(none)",
        memory=memory_digest(state), history=history_digest(state),
        pool=pool_stats(state), evidence=evidence_digest(state, 5, 220, exclude_tool=True),
    )
    msgs = [SystemMessage(content=system)] + history
    msgs.append(HumanMessage(content=f"[Tool budget: {remaining} round(s) left.]" if remaining
                             else "Tool budget exhausted. Write your INVESTIGATION NOTES now."))
    ai = None
    if remaining > 0:
        try:
            ai = invoke_chat(msgs, tools=TOOLS, label="investigate")
        except Exception as e:  # noqa: BLE001 - malformed tool call etc. -> conclude without tools
            if is_api_error(e):  # rate limit / network / auth: surface it (the run can be retried)
                raise
    if ai is None:
        ai = invoke_chat(msgs + [HumanMessage(content="Write your INVESTIGATION NOTES now (no tools).")],
                         label="investigate")
    if getattr(ai, "tool_calls", None):
        return {"messages": [ai], "tool_rounds": rounds + 1}
    return {"messages": [ai], "investigation_notes": str(ai.content)}


def route_after_investigate(state: AgentState) -> Literal["tools", "respond"]:
    last = (state.get("messages") or [None])[-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "respond"


def route_after_tools(state: AgentState) -> Literal["investigate", "respond"]:
    """When the tool budget is spent, go straight to `respond` (it sees the full evidence pool) instead of
    paying for an extra 'write your notes' call - saves one main-model call (~3.5K tokens) per ticket.
    ask_customer refunds its round, so the agent always gets to process a customer's answer."""
    return "respond" if (state.get("tool_rounds") or 0) >= settings.max_tool_rounds else "investigate"


# =============================================================== 5. respond

RESPOND_SYSTEM = """You write the final answer for a support case using ONLY the evidence pool, the ticket,
clarifications and remembered customer facts.
GROUNDING RULE: every troubleshooting step or promise must appear in at least one cited case's past
resolution. Do NOT add generic steps from your own knowledge (e.g. re-authorize, clear cache, reinstall,
hold a reset button, specific settings) unless a cited case contains them. If the cases mainly ask the
customer for details or describe a standard process, your reply does the same - that is a valid answer.
Customer reply: acknowledge the specific issue in one sentence; give numbered, concrete steps that are
supported by the cited cases (prefer steps several cases agree on; mark single-source steps from historical
cases as optional - never a support lead's steps, see below); say what to send back if it doesn't work
(never passwords/card numbers); personalise with known customer facts; never mention case ids, similarity
scores or "dataset"; keep it under ~180 words.
In `analysis`, compare the cases: consensus, conflicts, and why the chosen approach fits this ticket.
Cite ONLY case ids from the evidence pool or the customer history (at least 2 when available).
LEAD-VERIFIED cases hold a support lead's confirmed root cause and fix: when one matches this ticket,
prefer its solution over conflicting historical cases and give EVERY step of the lead's verified solution
as a regular step - a lead-verified step is confirmed, so it is never "optional" or hedged for being
single-source (keep a condition the lead wrote, e.g. "if you need live data, ...").
Escalate (escalation_required=true, with reason, target_team = a valid queue, urgency, handoff_summary)
only if escalation criteria apply; then the reply is a holding message that sets expectations.
Valid queues: {queues}"""

RESPOND_LEAD_RULE = """

SUPPORT-LEAD INPUT (authoritative): the support lead reviewed this case. Their diagnosis and feedback
override the historical cases and count as supported evidence for the grounding rule.
- Base the customer reply on the lead's root cause and solution: explain the cause in plain words and turn
  the solution into the numbered steps. Keep steps from cases only where they agree with the lead.
- Cite the lead's diagnosis id ({lead_id}) in evidence_used when it is in the pool; where cases disagree
  with the lead, follow the lead and note the conflict in `analysis`.
- Do everything the lead's review feedback asks for (e.g. request a specific detail, mention an amount).
- Every point the lead makes - root cause, solution, note or feedback (e.g. "X might be the issue, check it
  too") - goes into the reply as a regular step or explanation. Never mark it optional, hedge it or drop it
  because no case confirms it: the lead's word is the confirmation. Only call it optional if the lead did.
  A lead's "X might be the issue, check it too" is an extra check next to the diagnosis: give it as a
  regular numbered step ("Check X ..."), without "(Optional)" or "not the main cause".
- The investigation notes never override the lead: ignore any part of them that doubts, dismisses or
  downgrades a point the lead made.
- Do not escalate just because no historical case contains the lead's fix; escalate only if the lead's
  solution itself needs another team."""


def respond(state: AgentState) -> dict:
    """One node for both outcomes: grounded resolution or escalation hand-off."""
    req = state.get("escalation_requested")
    extra = ""
    if req:  # escalating: criteria + hand-off format
        extra += "\n\n" + skills.render(["escalation"])
    elif needs_escalation_skill(state):  # may escalate: criteria only
        extra += "\n\n" + skills.render(["escalation"], brief=True)
    if req:
        extra += f"\n\nTHIS CASE MUST BE ESCALATED. Reason: {req.get('reason')} (suggested team: {req.get('target_team')})."
    ev = state.get("evaluation") or {}
    feedback = (f"\n\nPREVIOUS DRAFT FAILED THE EVIDENCE CHECK - fix this: {ev.get('feedback')}"
                if ev and not ev.get("passed") and not req else "")
    lead = lead_digest(state)
    if lead:
        extra += RESPOND_LEAD_RULE.format(lead_id=lead_id(state.get("case_id")))
    system = RESPOND_SYSTEM.format(queues=labels.vocabulary()["queue"]) + extra
    notes = state.get("investigation_notes") or ""
    human = (f"TICKET:\n{ticket_text(state)}\n\nCLASSIFICATION: {classification_str(state.get('classification'))}\n\n"
             f"REMEMBERED CUSTOMER FACTS:\n{memory_digest(state)}\n\nCUSTOMER HISTORY:\n{history_digest(state)}\n\n"
             + (f"SUPPORT-LEAD INPUT (authoritative):\n{lead}\n\n" if lead else "")
             + f"EVIDENCE POOL:\n{evidence_digest(state, 5, 380, prefer=CASE_ID_RE.findall(norm_id(notes)))}\n\n"
             f"INVESTIGATION NOTES:\n{(notes or '(none)')[:2500]}{feedback}")
    out = structured(ResponseOut, [SystemMessage(content=system), HumanMessage(content=human)])
    d = out.model_dump()
    d["escalation_required"] = bool(d["escalation_required"] or req)
    d["self_confidence"] = clamp(d["self_confidence"])
    d["kind"] = "escalation" if d["escalation_required"] else "resolution"
    if d["escalation_required"]:
        d["target_team"] = labels.normalize("queue", d.get("target_team") or (req or {}).get("target_team")) \
            or (state.get("classification") or {}).get("queue")
        d["escalation_reason"] = d.get("escalation_reason") or (req or {}).get("reason", "")
        if d.get("handoff_summary"):
            d["internal_actions"] = [f"Escalate to {d['target_team']} ({d.get('urgency') or 'high'})",
                                     d["handoff_summary"]] + d["internal_actions"]
    return {"draft": d}


# =============================================================== 6. evaluate

JUDGE_SYSTEM = """You are a strict QA reviewer for support replies. Check whether every substantive claim or
step in the draft is supported by the CITED EVIDENCE, by information the customer provided (ticket,
clarifications, remembered facts) or by SUPPORT-LEAD INPUT (a support lead's diagnosis or review feedback is
authoritative: claims and steps that follow from it are supported). Courtesy phrases, requests for information from the customer and
statements of process ("we will investigate", "a specialist will follow up") need no support. A step that
merely generalises a cited resolution is supported; specific settings, numbers, URLs, troubleshooting
steps or promises that appear in no cited case are not.
groundedness = share of substantive claims that are supported. addresses_issue = how well it solves
the customer's actual problem. If not grounded, give concrete fix instructions and a search query that
could find the missing evidence."""


def evaluate(state: AgentState) -> dict:
    """Evidence check (deterministic checks + LLM judge) and confidence scoring."""
    d = state.get("draft") or {}
    # the customer's own previous cases are legitimate evidence too (e.g. "recurring issue -> escalate")
    evidence = {**history_as_evidence(state), **(state.get("evidence") or {})}
    cited = list(dict.fromkeys(norm_id(e["case_id"]) for e in d.get("evidence_used") or [] if e.get("case_id")))
    valid = [c for c in cited if c in evidence]
    invalid = [c for c in cited if c not in evidence]
    is_esc = d.get("kind") == "escalation"
    # a lead's diagnosis is the answer for a case the historical base does not cover: no 2nd case required
    lead_cited = any(evidence[c].get("source") == LEAD_SOURCE for c in valid)
    lead = lead_digest(state)
    checks = {
        "cites_existing_evidence": bool(valid) and not invalid,
        "compares_multiple_cases": len(valid) >= 2 or is_esc or lead_cited,
        "has_customer_reply": len(d.get("resolution") or "") >= 60,
        "classification_valid": all((state.get("classification") or {}).get(k) for k in ("queue", "type", "priority")),
    }
    if is_esc:
        # A holding reply makes no factual claims -> rule checks only (saves a judge call per escalation).
        checks["has_escalation_reason"] = bool(d.get("escalation_reason"))
        checks["valid_target_team"] = bool(labels.normalize("queue", d.get("target_team")))
        j = JudgeOut(grounded=all(checks.values()), groundedness=1.0 if all(checks.values()) else 0.5,
                     addresses_issue=0.4, feedback="escalation: rule-based checks only")
    else:
        cited_txt = "\n".join(
            f"[{c}] Issue: {(evidence[c].get('subject') or '')} - {(evidence[c].get('body') or '')[:300]}\n"
            f"  Resolution: {clip_answer(evidence[c].get('answer'), 1000)}" for c in valid[:5]) or "(no valid citations)"
        human = (f"TICKET:\n{ticket_text(state)}\n\nREMEMBERED FACTS:\n{memory_digest(state)}\n\n"
                 + (f"SUPPORT-LEAD INPUT:\n{lead}\n\n" if lead else "")
                 + f"CITED EVIDENCE:\n{cited_txt}\n\n"
                 f"DRAFT:\nAnalysis: {d.get('analysis')}\nReply: {d.get('resolution')}\n"
                 f"Internal actions: {d.get('internal_actions')}")
        j = structured(JudgeOut, [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=human)], fast=True)
    grounded, addresses = clamp(j.groundedness), clamp(j.addresses_issue)

    sims = sorted([evidence[c]["similarity"] for c in valid if evidence[c].get("similarity") is not None], reverse=True)
    retrieval_strength = (1.0 if lead_cited else  # the support lead verified the fix for this very case
                          clamp(((sum(sims[:3]) / len(sims[:3])) - 0.55) / 0.35) if sims else 0.4)
    q = (state.get("classification") or {}).get("queue")
    agreement = (sum(1 for c in valid if evidence[c].get("queue") == q) / len(valid)) if valid else 0.0
    confidence = (0.40 * grounded + 0.20 * addresses + 0.20 * retrieval_strength + 0.10 * agreement
                  + 0.10 * clamp(d.get("self_confidence")) - min(0.05 * len(d.get("conflicts") or []), 0.15)
                  - (0.15 if invalid else 0.0))
    passed = bool(j.grounded) and grounded >= 0.7 and all(checks.values())
    feedback = j.feedback + (f" Cited ids not in the evidence pool: {invalid}." if invalid else "") + \
        ("" if checks["compares_multiple_cases"] else " Cite and compare at least two relevant cases.")
    prev_failed = (state.get("evaluation") or {}).get("failed_attempts", 0)
    attempts = prev_failed + (0 if passed else 1)
    result = {"passed": passed, "groundedness": round(grounded, 3), "addresses_issue": round(addresses, 3),
              "retrieval_strength": round(retrieval_strength, 3), "label_agreement": round(agreement, 3),
              "confidence": round(clamp(confidence), 3), "checks": checks, "unsupported_claims": j.unsupported_claims,
              "feedback": feedback, "suggested_search": j.suggested_search,
              "attempt": prev_failed + 1, "failed_attempts": attempts}
    update: dict = {"evaluation": result}
    if not passed and not is_esc:
        if attempts <= settings.max_eval_retries:  # retry loop: re-investigate with the judge's feedback
            update["messages"] = [HumanMessage(content=(
                f"EVIDENCE CHECK FAILED (attempt {attempts}). Issues: {feedback} Unsupported: {j.unsupported_claims}. "
                f"Run ONE targeted search for the missing evidence{(' - e.g. ' + repr(j.suggested_search)) if j.suggested_search else ''}; "
                "if nothing supports those claims, they must be dropped from the answer."))]
            update["tool_rounds"] = max(settings.max_tool_rounds - 1, 0)  # one round, then straight to respond
        else:  # retries exhausted -> escalate instead of sending an unsupported answer
            update["escalation_requested"] = {"reason": f"Evidence check failed {attempts}x: {feedback[:300]}",
                                              "target_team": q, "urgency": "medium"}
    return update


def route_after_evaluate(state: AgentState) -> Literal["review", "investigate", "respond"]:
    if (state.get("evaluation") or {}).get("passed") or (state.get("draft") or {}).get("kind") == "escalation":
        return "review"
    return "respond" if state.get("escalation_requested") else "investigate"


# =============================================================== 7. review


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _txt(x) -> str:
    return str(x or "").strip()


def lead_update(state: AgentState, decision: dict) -> dict:
    """Record the lead's root cause + solution: merged into `lead_guidance`, added to the evidence pool as the
    authoritative item LEAD-<case_id>, and persisted at once so the lead's input is never lost."""
    prev = state.get("lead_guidance") or {}
    g = {**prev, **{k: _txt(decision.get(k)) for k in ("root_cause", "solution") if _txt(decision.get(k))}}
    if _txt(decision.get("note")):
        g["note"] = _txt(decision.get("note"))
    g.update(by=decision.get("by") or "support_lead", at=_now())
    c = state.get("classification") or {}
    lid = lead_id(state.get("case_id"))
    item = {"case_id": lid, "subject": f"Support-lead diagnosis for {state.get('case_id')}", "body": "",
            "answer": lead_knowledge(g), "queue": c.get("queue"), "type": c.get("type"),
            "priority": c.get("priority"), "tags": c.get("tags") or [], "source": LEAD_SOURCE,
            "lead_verified": True, "similarity": None, "retrieved_by": ["support_lead"]}
    if state.get("case_id"):
        db.update_case(state["case_id"], lead_guidance=g)
    return {"lead_guidance": g, "evidence": {lid: item}}


def review_update(state: AgentState, decision, force_amend: bool = False) -> dict:
    """Apply a support-lead decision. Shared by the review node (interrupt) and service.lead_solution
    (a lead's fix for a case that already finished, e.g. an escalation).

    accept  -> finalize. A diagnosis typed with accept is never dropped: if it is new, it is applied first.
    amend   -> the lead's root cause / solution becomes authoritative evidence; the reply is rewritten from
               it (respond), evidence-checked and shown for review again.
    reject  -> re-investigate with the feedback (plus any diagnosis given with it)."""
    if not isinstance(decision, dict):
        decision = {"action": str(decision)}
    action = _txt(decision.get("action") or "accept").lower()
    by = decision.get("by") or "support_lead"
    fb = _txt(decision.get("feedback"))
    prev = state.get("lead_guidance") or {}
    given = any(_txt(decision.get(k)) for k in ("root_cause", "solution"))
    new_guidance = (given and force_amend) or any(_txt(decision.get(k)) and _txt(decision.get(k)) != _txt(prev.get(k))
                                                  for k in ("root_cause", "solution"))
    if action == "accept" and new_guidance:
        action = "amend"
    if action == "amend" and not new_guidance:  # nothing new to apply
        action = "reject" if fb else "accept"
    reset = {"tool_rounds": 0, "evaluation": None, "escalation_requested": None, "investigation_notes": ""}
    if action == "reject":
        fb = fb or "Rejected without comment."
        upd = lead_update(state, decision) if new_guidance else {}
        return {**upd, **reset, "review": {"action": "reject", "feedback": fb, "by": by, "at": _now()},
                "messages": [HumanMessage(content=f"{REJECT_PREFIX}{fb}{REJECT_SUFFIX}", name="support_lead")]}
    if action == "amend":
        upd = lead_update(state, {**decision, "note": fb})
        lid = lead_id(state.get("case_id"))
        return {**upd, **reset, "review": {"action": "amend", "by": by, "feedback": fb or None, "at": _now()},
                "messages": [HumanMessage(name="support_lead", content=(
                    f"The support lead diagnosed this case ({lid}):\n{lead_knowledge(upd['lead_guidance'])}\n"
                    + (f"Lead's note: {fb}\n" if fb else "")  # kept in the thread, so a later re-investigation sees it
                    + "The reply must be based on this diagnosis; every point the lead made is required, not optional."))]}
    return {"review": {"action": "accept", "by": by, "note": fb or None, "at": _now()}}


def review(state: AgentState, config: RunnableConfig) -> dict:
    """Support-lead gate: only accepted, evidence-checked resolutions become trusted knowledge, and a lead's own
    diagnosis is applied to the reply and kept as lead-verified knowledge."""
    if config["configurable"].get("auto_accept"):
        decision = {"action": "accept", "by": "auto"}
    else:
        decision = interrupt({"type": "review", "case_id": state.get("case_id"), "draft": state.get("draft"),
                              "evaluation": state.get("evaluation"), "classification": state.get("classification"),
                              "lead_guidance": state.get("lead_guidance")})
    return review_update(state, decision)


def route_after_review(state: AgentState) -> list[str] | str:
    action = (state.get("review") or {}).get("action")
    if action == "reject":
        return "investigate"
    if action == "amend":  # the lead supplied the answer: rewrite the reply from it, no new investigation
        return "respond"
    return ["finalize", "update_memory"]  # independent side effects run in parallel


# =============================================================== 8. finalize (persist + grow KB)


def finalize(state: AgentState) -> dict:
    d = state.get("draft") or {}
    ev = state.get("evaluation") or {}
    evidence = {**history_as_evidence(state), **(state.get("evidence") or {})}
    c = state.get("classification") or {}
    used = []
    for e in d.get("evidence_used") or []:
        item = evidence.get(norm_id(e["case_id"]))
        if item:
            used.append({"case_id": item["case_id"], "source": item.get("source"), "similarity": item.get("similarity"),
                         "lead_verified": bool(item.get("lead_verified")),
                         "subject": item.get("subject"), "issue": (item.get("body") or "")[:900],
                         "resolution": clip_answer(item.get("answer"), 1100), "summary": (item.get("answer") or "")[:220],
                         "queue": item.get("queue"), "how_used": e.get("how_used")})
    escalated = bool(d.get("escalation_required"))
    guidance = state.get("lead_guidance") or None
    review_ = state.get("review") or {}
    final = {
        "case_id": state.get("case_id"), "customer_id": state.get("customer_id"),
        "ticket": {"subject": (state.get("ticket") or {}).get("subject"), "body": (state.get("ticket") or {}).get("body")},
        "status": "escalated" if escalated else "resolved",
        "classification": {k: c.get(k) for k in ("queue", "type", "priority", "tags")},
        "evidence": used, "analysis": d.get("analysis"), "resolution": d.get("resolution"),
        "internal_actions": d.get("internal_actions") or [],
        "escalation": {"required": escalated, "reason": d.get("escalation_reason") or None,
                       "target_team": d.get("target_team"), "urgency": d.get("urgency")},
        "confidence": ev.get("confidence"),
        "evaluation": {k: ev.get(k) for k in ("passed", "groundedness", "addresses_issue", "retrieval_strength",
                                              "label_agreement", "checks", "unsupported_claims")},
        "clarifications": state.get("clarifications") or [],
        "skills_used": state.get("skills") or [],
        "reviewed_by": review_.get("by"),
        "review": review_ or None,
        "lead_guidance": guidance,
    }
    db.update_case(state["case_id"], status=final["status"], classification=final["classification"],
                   analysis=final["analysis"], resolution=final["resolution"], evidence=used,
                   escalation=final["escalation"], confidence=final["confidence"], evaluation=final["evaluation"],
                   clarifications=final["clarifications"], lead_guidance=guidance, review=review_ or None)

    # Continuously growing knowledge: accepted + evidence-checked resolutions are indexed, and so is any accepted
    # case with a support-lead diagnosis - even an escalation: the lead's fix is knowledge the base did not have.
    accepted = review_.get("action") == "accept"
    lead_ok = accepted and bool(lead_knowledge(guidance))
    if not accepted:
        index_result = {"indexed": False, "reason": "not accepted by the reviewer"}
    elif escalated and not lead_ok:
        index_result = {"indexed": False, "reason": "escalated cases are stored but not indexed as trusted resolutions"}
    elif not escalated and not ev.get("passed") and not lead_ok:
        index_result = {"indexed": False, "reason": "evidence check not passed"}
    else:
        t = state.get("ticket") or {}
        body = (t.get("body") or "") + "".join(f"\n(Clarified: {x['answer']})" for x in final["clarifications"])
        get_index().upsert_cases([{
            "case_id": final["case_id"], "subject": t.get("subject"), "body": body,
            # an escalation's holding reply is no resolution: then only the lead's diagnosis is the answer
            "answer": knowledge_answer(None if escalated else final["resolution"], guidance),
            "queue": c.get("queue"), "type": c.get("type"), "priority": c.get("priority"), "tags": c.get("tags") or [],
            "source": "live", "resolved_at": _now(), "customer_id": final["customer_id"],
            "lead_verified": lead_ok, "root_cause": (guidance or {}).get("root_cause") if lead_ok else None,
            "lead_solution": (guidance or {}).get("solution") if lead_ok else None}])
        db.update_case(final["case_id"], indexed=1)
        index_result = {"indexed": True, "collection": settings.collection, "case_id": final["case_id"],
                        "lead_verified": lead_ok}
    final["index"] = index_result
    return {"final": final}


# =============================================================== 9. update_memory (parallel to finalize)

MEMORY_SYSTEM = """You maintain LONG-TERM memory about a CUSTOMER for future, separate support conversations.
Keep only durable, customer-level facts that will help next time: products/services they use,
environment/tech stack, plan/tier, integrations, preferences (language, contact channel), recurring
issues, important constraints (e.g. regulated industry). Do NOT store: one-off details of this ticket,
resolution steps, speculation, secrets, passwords, payment/card data or sensitive personal data.
If an existing fact changed, return it with the SAME key and the new value; list obsolete keys in
`forget`. Return at most 4 facts, or none if nothing durable was learned."""


def update_memory(state: AgentState, *, store: BaseStore) -> dict:
    cid = state.get("customer_id") or "anonymous"
    existing = memory.get_facts(store, cid)
    d = state.get("draft") or {}
    human = (f"EXISTING FACTS:\n{json.dumps([{'key': x['key'], 'value': x.get('value')} for x in existing])}\n\n"
             f"TICKET + CLARIFICATIONS:\n{ticket_text(state)}\n\nOUTCOME: "
             f"{'escalated' if d.get('escalation_required') else 'resolved'} | "
             f"classification={classification_str(state.get('classification'))}")
    try:
        out = structured(MemoryOut, [SystemMessage(content=MEMORY_SYSTEM), HumanMessage(content=human)], fast=True)
    except Exception as e:  # noqa: BLE001 - memory must never break case completion
        return {"memory_update": {"error": str(e)[:200]}}
    res = memory.apply_update(store, cid, [x.model_dump() for x in out.facts[:4]], out.forget, state.get("case_id"))
    res["reasoning"] = out.reasoning
    return {"memory_update": res}
