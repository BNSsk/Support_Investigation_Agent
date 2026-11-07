"""LangSmith-compatible evaluators: f(inputs, outputs, reference_outputs) -> result dict(s).

Covered dimensions: classification, retrieval relevance, response quality, groundedness,
tool usage and overall agent behaviour. LLM-as-judge evaluators use the Groq fast model.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import structured
from app.tools import TOOLS

TOOL_NAMES = {t.name for t in TOOLS}


def _final(outputs: dict) -> dict:
    return (outputs or {}).get("final") or {}


# ------------------------------------------------------------------ heuristic evaluators
def classification_accuracy(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    c = _final(outputs).get("classification") or {}
    res = []
    for k in ("queue", "type", "priority"):
        ok = (c.get(k) or "").lower() == (reference_outputs.get(k) or "").lower()
        res.append({"key": f"{k}_match", "score": int(ok), "comment": f"pred={c.get(k)} ref={reference_outputs.get(k)}"})
    ref_tags = {t.lower() for t in reference_outputs.get("tags") or []}
    pred_tags = {t.lower() for t in c.get("tags") or []}
    overlap = len(ref_tags & pred_tags) / len(ref_tags | pred_tags) if (ref_tags | pred_tags) else 0
    res.append({"key": "tag_jaccard", "score": round(overlap, 3)})
    return {"results": res}


def retrieval_queue_precision(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    retrieved = (outputs or {}).get("retrieved") or []
    top = retrieved[:8]
    if not top:
        return {"key": "retrieval_queue_precision", "score": 0, "comment": "nothing retrieved"}
    p = sum(1 for r in top if r.get("queue") == reference_outputs.get("queue")) / len(top)
    return {"key": "retrieval_queue_precision", "score": round(p, 3),
            "comment": f"{len(top)} retrieved; share in reference queue"}


def tool_usage(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    calls = (outputs or {}).get("tool_calls") or []
    names = [c["name"] for c in calls]
    invalid = [n for n in names if n not in TOOL_NAMES]
    # scenario cases name the tools the situation requires (e.g. ask_customer for a vague ticket)
    expected = (((reference_outputs or {}).get("expect") or {}).get("tools")) or []
    missing = [t for t in expected if t not in names]
    budget = settings.max_tool_rounds * 3
    if invalid:
        score = 0.0
    elif missing:
        score = 0.5  # the required tool was not used
    elif not names:
        score = 0.5  # allowed (model decided no tools needed) but usually suboptimal
    elif len(names) > budget:
        score = 0.5
    else:
        score = 1.0
    return {"key": "tool_usage", "score": score,
            "comment": f"tools={names} invalid={invalid} expected={expected} missing={missing} "
                       f"distinct={len(set(names))}"}


def agent_behavior(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    f = _final(outputs)
    nodes = (outputs or {}).get("nodes") or []
    checks = {
        "has_all_output_fields": all(f.get(k) not in (None, "", []) for k in
                                     ("classification", "evidence", "analysis", "resolution", "escalation"))
                                  and f.get("confidence") is not None,
        "confidence_in_range": isinstance(f.get("confidence"), (int, float)) and 0 <= f["confidence"] <= 1,
        "escalation_has_reason": (not (f.get("escalation") or {}).get("required")) or bool(f["escalation"].get("reason")),
        "evidence_check_ran": "evaluate" in nodes,
        "case_persisted": "finalize" in nodes,
        "multi_case_evidence": len(f.get("evidence") or []) >= 2 or (f.get("escalation") or {}).get("required", False),
        "no_error": not (outputs or {}).get("error"),
    }
    # scenario cases: the behaviour the scenario must show (evals/scenarios.py EXPECT)
    exp = (reference_outputs or {}).get("expect") or {}
    facts = (outputs or {}).get("scenario_facts") or {}
    if "escalation" in exp:
        checks["escalation_as_expected"] = bool((f.get("escalation") or {}).get("required")) == exp["escalation"]
    if "clarification" in exp:
        checks["clarification_as_expected"] = bool((outputs or {}).get("clarification_asked")) == exp["clarification"]
    if "memory_reused" in exp:
        checks["memory_reused"] = bool(facts.get("memory_reused")) == exp["memory_reused"]
    if "first_case_indexed" in exp:
        checks["first_case_indexed"] = bool(facts.get("first_case_indexed")) == exp["first_case_indexed"]
    if "new_case_retrieved" in exp:
        checks["new_case_retrieved"] = bool(facts.get("new_case_retrieved")) == exp["new_case_retrieved"]
    if "lead_case_retrieved" in exp:
        checks["first_case_lead_verified"] = bool(facts.get("first_case_lead_verified"))
        checks["lead_case_retrieved"] = bool(facts.get("lead_case_retrieved")) == exp["lead_case_retrieved"]
    score = sum(checks.values()) / len(checks)
    return {"key": "agent_behavior", "score": round(score, 3),
            "comment": ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items())}


# ------------------------------------------------------------------ LLM-as-judge evaluators
class Score(BaseModel):
    score: int = Field(description="1 (very poor) to 5 (excellent)")
    reasoning: str


def _judge(system: str, human: str) -> Score:
    return structured(Score, [SystemMessage(content=system), HumanMessage(content=human)], fast=True)


def _conversation(inputs: dict, outputs: dict) -> str:
    """What the agent legitimately knew: the ticket, any clarification exchange, and remembered customer facts."""
    o = outputs or {}
    parts = [f"Subject: {inputs.get('subject')}\n{inputs.get('body')}"]
    for c in o.get("clarifications") or []:
        parts.append(f"Agent asked the customer: {c.get('question')}\nCustomer answered: {c.get('answer')}")
    facts = o.get("memory_loaded") or []
    if facts:
        parts.append("Known customer facts (long-term memory from earlier conversations): "
                     + "; ".join(f"{m.get('key')}: {m.get('value')}" for m in facts))
    return "\n\n".join(parts)


def response_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    f = _final(outputs)
    if not f.get("resolution"):
        return {"key": "response_quality", "score": 0, "comment": "no resolution"}
    s = _judge(
        "You grade a support agent's reply against a reference for the same ticket: either the reply a human "
        "agent wrote, or the key points an expert expects. Judge correctness of the approach, helpfulness, "
        "actionability and tone. The agent may be more detailed than the reference; penalise contradictions, "
        "invented facts, or missing key steps. A clarifying question the agent asked during the conversation "
        "counts towards any 'first ask' step in the reference.",
        f"CONVERSATION:\n{_conversation(inputs, outputs)}\n\nREFERENCE:\n{reference_outputs.get('answer')}\n\n"
        f"AGENT REPLY:\n{f.get('resolution')}\n\nAGENT ESCALATED: {(f.get('escalation') or {}).get('required')}")
    return {"key": "response_quality", "score": round((s.score - 1) / 4, 3), "comment": s.reasoning}


def groundedness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    f = _final(outputs)
    # full cited resolution (a support lead's verified fix is never cut), not just the 220-char summary
    ev = "\n".join(f"[{e['case_id']}] {e.get('subject')}: {e.get('resolution') or e.get('summary')}"
                   for e in f.get("evidence") or [])
    if not f.get("resolution"):
        return {"key": "groundedness", "score": 0, "comment": "no resolution"}
    s = _judge(
        "Rate how well the reply is grounded in the cited historical cases and the conversation (the ticket, the "
        "customer's answers and known customer facts). 5 = every substantive step is supported; 1 = mostly invented.",
        f"CONVERSATION:\n{_conversation(inputs, outputs)}\n\nCITED CASES:\n{ev}\n\nREPLY:\n{f.get('resolution')}")
    return {"key": "groundedness", "score": round((s.score - 1) / 4, 3), "comment": s.reasoning}


def retrieval_relevance(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    retrieved = ((outputs or {}).get("retrieved") or [])[:5]
    if not retrieved:
        return {"key": "retrieval_relevance", "score": 0, "comment": "nothing retrieved"}
    listing = "\n".join(f"- {r['case_id']}: {r.get('subject')} | {r.get('queue')}" for r in retrieved)
    s = _judge(
        "Rate how relevant the retrieved historical cases are for investigating the new ticket "
        "(5 = all highly relevant same-issue cases, 1 = unrelated).",
        f"NEW TICKET:\n{inputs.get('subject')}\n{inputs.get('body')}\n\nRETRIEVED CASES:\n{listing}")
    return {"key": "retrieval_relevance", "score": round((s.score - 1) / 4, 3), "comment": s.reasoning}


ALL_EVALUATORS = [classification_accuracy, retrieval_queue_precision, retrieval_relevance, response_quality,
                  groundedness, tool_usage, agent_behavior]
# Scenario cases have no human labels, so the two label-based evaluators apply to the held-out suite only.
SCENARIO_EVALUATORS = [retrieval_relevance, response_quality, groundedness, tool_usage, agent_behavior]
