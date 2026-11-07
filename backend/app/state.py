"""Graph state. Checkpointed per thread (= short-term memory) by the SQLite checkpointer."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_evidence(left: dict | None, right: dict | None) -> dict:
    """Evidence pool keyed by case id. Parallel/looping retrievals merge instead of overwrite."""
    out = dict(left or {})
    for cid, item in (right or {}).items():
        if cid in out:
            prev = out[cid]
            by = list(dict.fromkeys((prev.get("retrieved_by") or []) + (item.get("retrieved_by") or [])))
            sims = [s for s in (prev.get("similarity"), item.get("similarity")) if s is not None]
            out[cid] = {**prev, **{k: v for k, v in item.items() if v is not None},
                        "retrieved_by": by, "similarity": max(sims) if sims else None}
        else:
            out[cid] = item
    return out


class AgentState(TypedDict, total=False):
    # identity
    case_id: str
    customer_id: str
    ticket: dict                      # {"subject", "body"}
    followup: str | None              # set when a new turn arrives on an existing thread

    # short-term conversation / scratchpad (ticket, tool calls, clarifications, feedback)
    messages: Annotated[list[AnyMessage], add_messages]

    # long-term memory snapshot + customer history (loaded per run)
    customer_memory: list[dict]
    customer_history: list[dict]

    # retrieval
    evidence: Annotated[dict[str, dict], merge_evidence]
    retrieval_log: Annotated[list[dict], operator.add]
    label_votes: dict

    # triage / skills
    classification: dict
    skills: list[str]

    # investigation loop
    tool_rounds: int
    investigation_notes: str
    escalation_requested: dict | None

    # clarifications gathered through the ask_customer tool (interrupt/resume)
    clarifications: Annotated[list[dict], operator.add]

    # support-lead knowledge: root cause + solution given at review. It enters the evidence pool as the
    # authoritative item LEAD-<case_id>, survives follow-up turns and is indexed with the case (lead-verified).
    lead_guidance: dict | None
    # what the governing support-lead knowledge (this case's diagnosis, or a matching lead-verified case) needs
    # from the customer before its outcome can be given (e.g. "gift card issue year"), and whether it was asked:
    # {sig, source, own, applies, needs, missing, question, asked}. Checked once per lead input (sig).
    lead_check: dict | None

    # output + quality loop (none of these are sent to the LLM; they carry the loop and the outcome)
    draft: dict                       # respond -> evaluate -> review -> finalize
    evaluation: dict                  # evidence check result; also counts failed attempts (retry loop)
    review: dict                      # support-lead decision: accept / reject + feedback
    final: dict                       # persisted outcome shown in the UI (incl. index result)
    memory_update: dict               # what update_memory saved/forgot (runs in parallel to finalize)
