"""Flatten evals/results/{heldout,scenarios}.json into the row files the results dashboard reads.

    python -m evals.export_dashboard        (from backend/) -> evals/results/dashboard/*.json
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "dashboard"

HELDOUT_METRICS = ["queue_match", "type_match", "priority_match", "tag_jaccard", "retrieval_queue_precision",
                   "retrieval_relevance", "response_quality", "groundedness", "tool_usage", "agent_behavior"]
SCENARIO_METRICS = ["retrieval_relevance", "response_quality", "groundedness", "tool_usage", "agent_behavior"]
SCENARIO_ORDER = ["normal", "clarify", "memory", "growing_kb", "lead_knowledge"]


def _score(ex: dict, key: str):
    v = (ex.get("scores") or {}).get(key) or {}
    return v.get("score")


def _checks(ex: dict) -> tuple[int, int, list[str]]:
    comment = ((ex.get("scores") or {}).get("agent_behavior") or {}).get("comment") or ""
    items = [c.strip() for c in comment.split(",") if "=" in c]
    failed = [c.split("=")[0] for c in items if c.endswith("FAIL")]
    return len(items) - len(failed), len(items), failed


def heldout_rows(doc: dict) -> list[dict]:
    rows = []
    for ex in doc["examples"]:
        o, ref = ex["outputs"], ex["reference"] or {}
        c = o.get("classification") or {}
        rows.append({
            "ticket": (ex.get("metadata") or {}).get("holdout_id") or ex["inputs"].get("customer_id"),
            "subject": ex["inputs"]["subject"],
            "ref_queue": ref.get("queue"), "agent_queue": c.get("queue"),
            "ref_type": ref.get("type"), "agent_type": c.get("type"),
            "ref_priority": ref.get("priority"), "agent_priority": c.get("priority"),
            "status": o.get("status") or "error", "confidence": o.get("confidence"),
            "clarified": bool(o.get("clarifications")),
            "tools": ", ".join(t["name"] for t in o.get("tool_calls") or []) or "none",
            "tokens": o.get("tokens"), "latency_s": o.get("latency_s"),
            **{m: _score(ex, m) for m in HELDOUT_METRICS},
            "quality_note": ((ex.get("scores") or {}).get("response_quality") or {}).get("comment"),
            "trace_url": ex.get("trace_url"),
        })
    return rows


def scenario_rows(doc: dict) -> list[dict]:
    rows = []
    for ex in doc["examples"]:
        o, inp = ex["outputs"], ex["inputs"]
        facts = o.get("scenario_facts") or {}
        steps = o.get("steps") or []
        clar = (o.get("clarifications") or [{}])[0] if o.get("clarifications") else {}
        passed, total, failed = _checks(ex)
        mem = o.get("memory_loaded") or []
        rows.append({
            "case": inp["case"], "scenario": inp["scenario"], "scenario_title": inp.get("scenario_title"),
            "order": SCENARIO_ORDER.index(inp["scenario"]) if inp["scenario"] in SCENARIO_ORDER else 9,
            "first_subject": steps[0]["subject"] if len(steps) > 1 else None,
            "first_status": steps[0]["status"] if len(steps) > 1 else None,
            "subject": inp["subject"], "body": inp["body"],
            "question": clar.get("question"), "answer": clar.get("answer"),
            "memory_facts": "; ".join(f"{m.get('key')}: {m.get('value')}" for m in mem) or None,
            "status": o.get("status") or "error", "confidence": o.get("confidence"),
            "first_case_indexed": facts.get("first_case_indexed"),
            "new_case_retrieved": facts.get("new_case_retrieved"),
            "lead_case_retrieved": facts.get("lead_case_retrieved"),
            "new_case_cited": facts.get("new_case_cited"),
            "checks_passed": passed, "checks_total": total, "failed_checks": ", ".join(failed) or None,
            "all_checks_passed": passed == total and total > 0,
            "tools": ", ".join(t["name"] for t in o.get("tool_calls") or []) or "none",
            "reply": o.get("resolution"),
            "quality_note": ((ex.get("scores") or {}).get("response_quality") or {}).get("comment"),
            "tokens": o.get("tokens"), "latency_s": o.get("latency_s"),
            **{m: _score(ex, m) for m in SCENARIO_METRICS},
            "trace_url": ex.get("trace_url"),
        })
    return sorted(rows, key=lambda r: (r["order"], r["case"]))


def run_rows(docs: dict[str, dict]) -> list[dict]:
    return [{"suite": s, "dataset": d.get("dataset"), "dataset_url": d.get("dataset_url"),
             "experiment": d.get("experiment"), "experiment_url": d.get("experiment_url"),
             "finished_at": d.get("finished_at"), "model": d.get("model"), "fast_model": d.get("fast_model"),
             "judge_model": d.get("judge_model") or d.get("fast_model"), "evaluators": ", ".join(d.get("evaluators") or [])}
            for s, d in docs.items()]


def main() -> None:
    docs = {s: json.loads((RESULTS / f"{s}.json").read_text(encoding="utf-8")) for s in ("heldout", "scenarios")}
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"heldout_tickets.json": heldout_rows(docs["heldout"]),
             "scenario_cases.json": scenario_rows(docs["scenarios"]),
             "eval_runs.json": run_rows(docs)}
    for name, rows in files.items():
        (OUT / name).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{OUT / name}: {len(rows)} rows, {(OUT / name).stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
