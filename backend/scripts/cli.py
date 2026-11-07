"""Interactive CLI client (alternative to the web UI).

    python scripts/cli.py --customer cust-acme
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, service  # noqa: E402


def drive(events) -> None:
    while True:
        thread_id, pending = None, None
        for ev in events:
            if ev["type"] == "run_started":
                thread_id = ev["thread_id"]
            elif ev["type"] == "node":
                print(f"  [{ev['node']:>17}] {ev['summary']}")
            elif ev["type"] == "interrupt":
                pending = ev["interrupt"]
            elif ev["type"] == "completed":
                print("\n" + json.dumps(ev["final"], indent=2))
            elif ev["type"] == "error":
                print(f"  ERROR: {ev['message']}")
        if not pending:
            return
        if pending["type"] == "clarification":
            ans = input(f"\nAgent asks: {pending['question']}\nYour answer> ")
            events = service.resume(thread_id, {"answer": ans})
        else:
            d = pending.get("draft") or {}
            print(f"\nPROPOSED REPLY:\n{d.get('resolution')}\n\nconfidence={pending.get('evaluation', {}).get('confidence')}")
            act = input("accept / reject? > ").strip().lower() or "accept"
            fb = input("feedback> ") if act.startswith("r") else ""
            events = service.resume(thread_id, {"action": "reject" if act.startswith("r") else "accept", "feedback": fb})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", default="cli-user")
    args = ap.parse_args()
    db.init_db()
    subject = input("Subject> ")
    print("Body (finish with an empty line):")
    lines = []
    while (line := input()) != "":
        lines.append(line)
    _, events = service.new_ticket(args.customer, subject, "\n".join(lines))
    drive(events)


if __name__ == "__main__":
    main()
