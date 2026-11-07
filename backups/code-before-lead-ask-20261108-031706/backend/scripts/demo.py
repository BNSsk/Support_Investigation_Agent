"""End-to-end demonstration of the four required paths (runs the graph in-process).

    python scripts/demo.py                 # all scenarios
    python scripts/demo.py --only clarify  # one of: normal, clarify, memory, growing_kb

NOTE: with embedded Qdrant (no QDRANT_URL) stop the API server first - the local store is single-process.
The full demo uses roughly 6 ticket runs (about 9-15K Groq tokens each); token usage is printed per ticket.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, memory, service  # noqa: E402
from app.graph import get_store  # noqa: E402
from app.samples import SCENARIOS  # noqa: E402

GENERIC_ANSWER = "I don't have more details right now - please proceed with your best recommendation."


def consume(events, step: dict | None = None, verbose: bool = True) -> tuple[str | None, dict | None, list[dict]]:
    """Print node summaries; auto-answer interrupts; return (thread_id, final, all node events)."""
    thread_id, final, nodes = None, None, []
    tokens: dict[str, int] = {}
    waited = 0.0
    while True:
        pending = None
        for ev in events:
            if ev["type"] == "run_started":
                thread_id = ev["thread_id"]
            elif ev["type"] == "llm_call":
                m = ev["model"].split("/")[-1]
                tokens[m] = tokens.get(m, 0) + ev["counted_tokens"]
            elif ev["type"] == "rate_wait":
                waited += ev["wait_s"]
                print(f"   (rate limit: {ev['model']} waiting {ev['wait_s']}s)")
            elif ev["type"] == "node":
                nodes.append(ev)
                if verbose:
                    print(f"   [{ev['node']:>17}] {ev['summary']}")
            elif ev["type"] == "interrupt":
                pending = ev["interrupt"]
            elif ev["type"] == "completed":
                final = ev["final"]
            elif ev["type"] == "error":
                print(f"   !! ERROR: {ev['message']}")
                return thread_id, None, nodes
        if pending is None:
            print(f"   GROQ TOKENS: {tokens}" + (f"  | waits: {waited:.0f}s" if waited else ""))
            return thread_id, final, nodes
        lead = has_lead(step)
        if pending.get("type") == "clarification":
            answer = (step or {}).get("clarification_answer") or GENERIC_ANSWER
            print(f"\n   >>> INTERRUPT - agent asks: {pending['question']}")
            print(f"   <<< customer answers: {answer}\n")
            events = service.resume(thread_id, {"answer": answer}, auto_accept=not lead)
        elif lead and not pending.get("lead_guidance"):
            # the support lead enters their own diagnosis at review: the reply is rewritten from it
            print(f"\n   >>> REVIEW - support lead applies a diagnosis:\n      root cause: {step['lead_root_cause']}"
                  f"\n      solution  : {step['lead_solution']}\n")
            events = service.resume(thread_id, {"action": "amend", "root_cause": step["lead_root_cause"],
                                                "solution": step["lead_solution"], "by": "demo-lead"})
        else:
            if lead:
                print("\n   >>> REVIEW - support lead accepts the reply rewritten from the diagnosis\n")
            events = service.resume(thread_id, {"action": "accept", "by": "demo"}, auto_accept=True)


def show_final(final: dict | None) -> None:
    if not final:
        print("   (no final result)")
        return
    print("\n   RESULT")
    print(f"   status      : {final['status']}   confidence: {final.get('confidence')}")
    print(f"   class       : {json.dumps(final['classification'])}")
    print(f"   evidence    : {[(e['case_id'], e.get('source'), e.get('similarity')) for e in final['evidence']]}")
    print(f"   escalation  : {final['escalation']}")
    print("   analysis    :\n" + textwrap.indent(textwrap.fill(final.get("analysis") or "", 100), "      "))
    print("   resolution  :\n" + textwrap.indent(final.get("resolution") or "", "      "))


def has_lead(step: dict | None) -> bool:
    return bool((step or {}).get("lead_root_cause") or (step or {}).get("lead_solution"))


def run_step(step: dict) -> tuple[str | None, dict | None, list[dict]]:
    print(f"\n-- ticket from {step['customer_id']}: {step['subject']!r}")
    # steps with a lead diagnosis stop at review, so the lead can apply it
    _, events = service.new_ticket(step["customer_id"], step["subject"], step["body"], auto_accept=not has_lead(step))
    tid, final, nodes = consume(events, step)
    show_final(final)
    return tid, final, nodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=[s["id"] for s in SCENARIOS])
    ap.add_argument("--reuse-customers", action="store_true",
                    help="keep the plain demo customer ids (memory/history from earlier runs then influences results)")
    args = ap.parse_args()
    db.init_db()
    store = get_store()
    # Fresh customers per run: earlier runs' memory/history would otherwise (correctly) change the agent's
    # behaviour, e.g. no clarification needed or 'recurring issue -> escalate'.
    suffix = "" if args.reuse_customers else "-" + time.strftime("%H%M%S")

    for sc in SCENARIOS:
        if args.only and sc["id"] != args.only:
            continue
        print("\n" + "=" * 100 + f"\n{sc['title']}  -  {sc['description']}\n" + "=" * 100)
        results = []
        for step in sc["steps"]:
            step = {**step, "customer_id": step["customer_id"] + suffix}
            results.append(run_step(step))
            if sc["id"] == "memory":
                facts = memory.get_facts(store, step["customer_id"])
                print(f"\n   LONG-TERM MEMORY for {step['customer_id']}: "
                      f"{[(f['key'], f.get('value')) for f in facts]}")
        if sc["id"] == "memory" and len(results) == 2:
            loaded = [n for n in results[1][2] if n["node"] == "intake"]
            asked = any(n["node"] == "tools" and n["data"].get("clarifications") for n in results[1][2])
            if loaded:
                print(f"\n   CHECK: second conversation loaded memory -> {loaded[0]['summary']}; "
                      f"clarification needed: {asked}")
        if sc["id"] == "growing_kb" and len(results) == 2 and results[0][1] and results[1][1]:
            first_case = results[0][1]["case_id"]
            st = service.thread_snapshot(results[1][0])
            found = first_case in (st["values"].get("evidence") or {})
            cited = any(e["case_id"] == first_case for e in results[1][1]["evidence"])
            print(f"\n   CHECK: newly resolved case {first_case} retrieved for the 2nd ticket: {found}; cited: {cited}")
        if sc["id"] == "lead_knowledge" and len(results) == 2 and results[0][1] and results[1][1]:
            first = results[0][1]
            st = service.thread_snapshot(results[1][0])
            item = (st["values"].get("evidence") or {}).get(first["case_id"]) or {}
            cited = any(e["case_id"] == first["case_id"] for e in results[1][1]["evidence"])
            print(f"\n   CHECK: case {first['case_id']} indexed as lead-verified: "
                  f"{bool((first.get('index') or {}).get('lead_verified'))}; retrieved for the 2nd ticket: "
                  f"{bool(item)} (lead_verified={item.get('lead_verified')}); cited: {cited}")


if __name__ == "__main__":
    main()
