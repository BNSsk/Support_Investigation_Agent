"""LangGraph topology (10 nodes).

START -> intake -> retrieve -> triage -> investigate <-> tools      (ReAct loop; tools include ask_customer,
             |                            ^  |                          which interrupts for clarification)
             +---- follow-up turn --------+  |   (evidence + classification already in state: no re-retrieval)
                                          v
                                       respond -> evaluate -> investigate   (evidence check failed: retry)
                                                           -> respond       (retries exhausted: escalate)
                                                           -> review [interrupt] -> investigate (rejected)
                                                                                 -> respond (lead's diagnosis applied)
                                                                                 -> finalize || update_memory -> END
"""
from __future__ import annotations

import sqlite3
import threading

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.sqlite import SqliteStore

from . import nodes as n
from .config import settings
from .state import AgentState
from .tools import TOOLS


def build_graph(checkpointer=None, store=None):
    g = StateGraph(AgentState)
    g.add_node("intake", n.intake)              # persist open case + load long-term memory/history
    g.add_node("retrieve", n.retrieve)          # hybrid RAG (+ trusted new cases)
    g.add_node("triage", n.triage)              # classification + dynamic skill selection
    g.add_node("investigate", n.investigate)    # model decides: tools / ask customer / escalate / conclude
    g.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    g.add_node("respond", n.respond)            # grounded resolution or escalation (one schema)
    g.add_node("evaluate", n.evaluate)          # evidence check + confidence
    g.add_node("review", n.review)              # acceptance gate before knowledge grows; takes the lead's diagnosis
    g.add_node("finalize", n.finalize)          # persist final case + index trusted resolution
    g.add_node("update_memory", n.update_memory)  # long-term customer memory

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", n.route_after_intake, ["retrieve", "investigate"])
    g.add_edge("retrieve", "triage")
    g.add_edge("triage", "investigate")
    g.add_conditional_edges("investigate", n.route_after_investigate, ["tools", "respond"])
    g.add_conditional_edges("tools", n.route_after_tools, ["investigate", "respond"])
    g.add_edge("respond", "evaluate")
    g.add_conditional_edges("evaluate", n.route_after_evaluate, ["review", "investigate", "respond"])
    g.add_conditional_edges("review", n.route_after_review, ["investigate", "respond", "finalize", "update_memory"])
    g.add_edge("finalize", END)
    g.add_edge("update_memory", END)
    return g.compile(checkpointer=checkpointer, store=store)


_graph = None
_store = None
_lock = threading.Lock()


def get_store() -> SqliteStore:
    global _store
    with _lock:
        if _store is None:
            conn = sqlite3.connect(settings.memory_db, check_same_thread=False, isolation_level=None)
            _store = SqliteStore(conn)
            _store.setup()
        return _store


def get_graph():
    global _graph
    store = get_store()
    with _lock:
        if _graph is None:
            saver = SqliteSaver(sqlite3.connect(settings.checkpoint_db, check_same_thread=False))
            saver.setup()
            _graph = build_graph(checkpointer=saver, store=store)
        return _graph
