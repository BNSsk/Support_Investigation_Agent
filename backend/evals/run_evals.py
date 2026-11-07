"""Run the evaluation suites with LangSmith datasets + evaluators (from backend/):

    python -m evals.run_evals                          # both suites
    python -m evals.run_evals --suite heldout --n 5    # first 5 held-out tickets only
    python -m evals.run_evals --suite scenarios --only clarify-cant-log-in
    python -m evals.run_evals --local                  # same evaluators, no LangSmith (JSON report only)

Suites
  heldout    40 English tickets held out at ingest (never indexed, so no answer leakage); human labels + human
             reply as reference. All 7 evaluators. Runs in parallel (--concurrency).
  scenarios  3 new cases for each of the 5 preloaded demo scenarios (evals/scenarios.py), multi-step where the
             scenario needs it. Reference key points + expected behaviour. The 5 evaluators that need no human
             labels. Runs one case at a time (cases write trusted cases to the shared index).

Isolation: every run works on a throw-away copy of app.db (historical tickets only) with fresh checkpoint and
memory stores, so the app's case list and customer memory are never touched. The Qdrant index is only read
by the held-out suite; the scenario suite needs to index cases (growing knowledge base, support-lead fix) and
deletes every point it wrote after each case. Results go to evals/results/<suite>.json (used by the report page).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.config import settings  # noqa: E402  (loads .env; nothing else may be imported before the sandbox)

RESULTS_DIR = BACKEND / "evals" / "results"
DATASETS = {"heldout": "support-agent-heldout", "scenarios": "support-agent-scenarios"}
FALLBACK_ANSWER = "I don't have more information. Please proceed with your best recommendation."
EVAL_PREFIX = "eval-"


# ======================================================================= sandbox + index guard
def make_sandbox() -> Path:
    """Point the app at a temp copy of app.db (historical tickets, no cases) and empty checkpoint/memory stores.
    Must run before app.db/app.graph are used."""
    tmp = Path(tempfile.mkdtemp(prefix="eval-sandbox-"))
    with sqlite3.connect(settings.app_db.as_uri() + "?mode=ro", uri=True) as src, \
            sqlite3.connect(tmp / "app.db") as dst:
        src.backup(dst)
    with sqlite3.connect(tmp / "app.db") as c:
        c.execute("DELETE FROM cases")
        c.execute("DELETE FROM events")
    settings.app_db = tmp / "app.db"
    settings.checkpoint_db = tmp / "checkpoints.db"
    settings.memory_db = tmp / "memory.db"
    return tmp


class IndexGuard:
    """Blocks or records the agent's writes of trusted cases to the shared Qdrant collection."""

    def __init__(self) -> None:
        self.allow = False
        self.written: set[str] = set()
        self._lock = threading.Lock()

    def install(self) -> None:
        from app.vectorstore import CaseIndex
        original, guard = CaseIndex.upsert_cases, self

        def upsert_cases(index, records, *args, **kwargs):
            if not guard.allow:
                return None  # held-out suite: the knowledge base must not change during evaluation
            with guard._lock:
                guard.written.update(r["case_id"] for r in records)
            return original(index, records, *args, **kwargs)

        CaseIndex.upsert_cases = upsert_cases

    def purge(self) -> int:
        """Delete every live point written by this run or created by an eval customer (also leftovers)."""
        from qdrant_client import models
        from app.vectorstore import get_index
        idx = get_index()
        live = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value="live"))])
        ids, offset = [], None
        while True:
            pts, offset = idx._q(idx.client.scroll, settings.collection, scroll_filter=live, limit=200, offset=offset,
                                 with_payload=["case_id", "customer_id"], with_vectors=False)
            ids += [p.id for p in pts if (p.payload.get("case_id") in self.written
                                          or str(p.payload.get("customer_id") or "").startswith(EVAL_PREFIX))]
            if offset is None:
                break
        if ids:
            idx._q(idx.client.delete, settings.collection,
                   points_selector=models.PointIdsList(points=ids), wait=True)
        idx._count_cache.clear()
        with self._lock:
            self.written.clear()
        return len(ids)


GUARD = IndexGuard()


# ======================================================================= running the agent
def run_ticket(step: dict, customer_id: str) -> dict:
    """One ticket end-to-end: auto-answers clarifications with the step's answer, applies a support lead's
    diagnosis at review when the step has one, otherwise auto-accepts. Returns the trajectory."""
    from app import service
    from app.graph import get_graph
    lead = bool(step.get("lead_root_cause") or step.get("lead_solution"))
    t0 = time.time()
    thread_id, events = service.new_ticket(customer_id, step["subject"], step["body"], auto_accept=not lead)
    nodes, tool_calls, clarifications = [], [], []
    final, error, memory_loaded, tokens, lead_applied = None, None, None, 0, False
    for _ in range(10):  # interrupt/resume rounds
        pending = None
        for ev in events:
            kind = ev["type"]
            if kind == "node":
                nodes.append(ev["node"])
                data = ev.get("data") or {}
                if ev["node"] == "investigate":
                    for m in data.get("messages") or []:
                        tool_calls += [{"name": c.get("name"), "args": c.get("args")} for c in m.get("tool_calls") or []]
                elif ev["node"] == "intake" and memory_loaded is None:
                    memory_loaded = data.get("customer_memory") or []
            elif kind == "llm_call":
                tokens += ev.get("counted_tokens") or 0
            elif kind == "interrupt":
                pending = ev["interrupt"]
            elif kind == "completed":
                final = ev["final"]
            elif kind == "error":
                error = ev["message"]
        if error or not pending:
            break
        if pending.get("type") == "clarification":
            answer = step.get("clarification_answer") or FALLBACK_ANSWER
            clarifications.append({"question": pending.get("question"), "answer": answer})
            events = service.resume(thread_id, {"answer": answer}, auto_accept=not lead)
        elif lead and not pending.get("lead_guidance"):
            lead_applied = True
            events = service.resume(thread_id, {"action": "amend", "root_cause": step.get("lead_root_cause", ""),
                                                "solution": step.get("lead_solution", ""), "by": "eval-support-lead"})
        else:
            events = service.resume(thread_id, {"action": "accept", "by": "eval"}, auto_accept=True)
    values = get_graph().get_state(service.config_for(thread_id)).values or {}
    pool = sorted((values.get("evidence") or {}).values(), key=lambda e: e.get("similarity") or 0, reverse=True)
    return {
        "final": final, "nodes": nodes, "tool_calls": tool_calls,
        "clarification_asked": bool(clarifications), "clarifications": clarifications,
        "retrieved": [{"case_id": e["case_id"], "subject": e.get("subject"), "queue": e.get("queue"),
                       "similarity": e.get("similarity"), "source": e.get("source"),
                       "lead_verified": bool(e.get("lead_verified")), "retrieved_by": e.get("retrieved_by")}
                      for e in pool],
        "memory_loaded": memory_loaded or [], "lead_applied": lead_applied,
        "tokens": tokens, "latency_s": round(time.time() - t0, 1), "error": error,
    }


def heldout_target(inputs: dict) -> dict:
    return run_ticket({"subject": inputs["subject"], "body": inputs["body"]}, inputs.get("customer_id", "eval"))


def scenario_target(inputs: dict) -> dict:
    """Run every step of a scenario (fresh eval customers); the last step is the one that is scored."""
    suffix = uuid.uuid4().hex[:4]
    steps: list[dict] = []
    try:
        for step in inputs["steps"]:
            steps.append(run_ticket(step, f"{EVAL_PREFIX}{step['customer_id']}-{suffix}"))
            if steps[-1]["error"] or not steps[-1]["final"]:
                break
    finally:
        if GUARD.allow:
            GUARD.purge()
    last, first = steps[-1], steps[0]
    facts = {"steps_completed": f"{len(steps)}/{len(inputs['steps'])}", "memory_reused": bool(last["memory_loaded"]),
             "clarification_asked": last["clarification_asked"]}
    if len(inputs["steps"]) > 1:
        first_case = (first["final"] or {}).get("case_id")
        index = (first["final"] or {}).get("index") or {}
        hit = next((r for r in last["retrieved"] if len(steps) > 1 and r["case_id"] in (first_case, f"LEAD-{first_case}")),
                   None)
        facts.update({
            "first_case_id": first_case, "first_case_indexed": bool(index.get("indexed")),
            "first_case_lead_verified": bool(index.get("lead_verified")),
            "new_case_retrieved": bool(hit), "lead_case_retrieved": bool(hit and hit.get("lead_verified")),
            "new_case_cited": len(steps) > 1 and any(e.get("case_id") == first_case
                                                     for e in (last["final"] or {}).get("evidence") or []),
        })
    summary = [{"subject": s_in["subject"], "customer_id": f"{EVAL_PREFIX}{s_in['customer_id']}",
                "case_id": (s["final"] or {}).get("case_id"), "status": (s["final"] or {}).get("status"),
                "clarifications": s["clarifications"], "memory_loaded": s["memory_loaded"],
                "index": (s["final"] or {}).get("index"), "lead_applied": s["lead_applied"],
                "resolution": (s["final"] or {}).get("resolution"), "tokens": s["tokens"],
                "latency_s": s["latency_s"], "error": s["error"]}
               for s_in, s in zip(inputs["steps"], steps)]
    return {**last, "tokens": sum(s["tokens"] for s in steps), "latency_s": round(sum(s["latency_s"] for s in steps), 1),
            "steps": summary, "scenario_facts": facts}


# ======================================================================= datasets
def heldout_examples(n: int | None) -> list[dict]:
    rows = [json.loads(line) for line in open(settings.eval_holdout, encoding="utf-8")]
    return [{"inputs": {"subject": r["subject"], "body": r["body"], "customer_id": f"{EVAL_PREFIX}{r['id']}"},
             "outputs": {k: r[k] for k in ("queue", "type", "priority", "tags", "answer")},
             "metadata": {"holdout_id": r["id"]}} for r in rows[:n]]


def scenario_examples(only: list[str] | None) -> list[dict]:
    from evals.scenarios import examples
    return [e for e in examples() if not only or e["inputs"]["case"] in only or e["inputs"]["scenario"] in only]


def example_key(suite: str, inputs: dict) -> str:
    return inputs["case"] if suite == "scenarios" else inputs["customer_id"]


def sync_dataset(client, suite: str, examples: list[dict]):
    """Create/update the LangSmith dataset (one example per stable key) and return the examples to run."""
    name = DATASETS[suite]
    desc = {"heldout": "40 English support tickets held out at ingest (never indexed): human labels + human reply.",
            "scenarios": "3 new cases for each of the 5 demo scenarios (normal, clarification, long-term memory, "
                         "growing knowledge base, support-lead fix) with reference key points and expected behaviour."}
    ds = (client.read_dataset(dataset_name=name) if client.has_dataset(dataset_name=name)
          else client.create_dataset(name, description=desc[suite]))
    existing = {example_key(suite, e.inputs): e for e in client.list_examples(dataset_id=ds.id)}
    for ex in examples:
        cur = existing.get(example_key(suite, ex["inputs"]))
        if cur is None:
            client.create_example(inputs=ex["inputs"], outputs=ex["outputs"], metadata=ex["metadata"], dataset_id=ds.id)
        elif cur.inputs != ex["inputs"] or (cur.outputs or {}) != ex["outputs"]:
            client.update_example(cur.id, inputs=ex["inputs"], outputs=ex["outputs"], metadata=ex["metadata"])
    wanted = [example_key(suite, e["inputs"]) for e in examples]
    by_key = {example_key(suite, e.inputs): e for e in client.list_examples(dataset_id=ds.id)}
    return ds, [by_key[k] for k in wanted if k in by_key]


# ======================================================================= reporting
def compact_outputs(o: dict | None) -> dict:
    o = o or {}
    f = o.get("final") or {}
    return {
        "status": f.get("status"), "confidence": f.get("confidence"), "classification": f.get("classification"),
        "analysis": f.get("analysis"), "resolution": f.get("resolution"), "escalation": f.get("escalation"),
        "evidence": [{"case_id": e.get("case_id"), "subject": e.get("subject"), "source": e.get("source"),
                      "lead_verified": e.get("lead_verified"), "similarity": e.get("similarity"),
                      "how_used": e.get("how_used")} for e in f.get("evidence") or []],
        "skills_used": f.get("skills_used"), "evidence_check": f.get("evaluation"),
        "tool_calls": o.get("tool_calls"), "clarifications": o.get("clarifications"),
        "retrieved": (o.get("retrieved") or [])[:6], "memory_loaded": o.get("memory_loaded"),
        "nodes": o.get("nodes"), "tokens": o.get("tokens"), "latency_s": o.get("latency_s"), "error": o.get("error"),
        "steps": o.get("steps"), "scenario_facts": o.get("scenario_facts"),
    }


def summarize(rows: list[dict]) -> dict:
    scores: dict[str, list[float]] = {}
    for r in rows:
        for k, v in r["scores"].items():
            if v.get("score") is not None:
                scores.setdefault(k, []).append(v["score"])
    out = {k: round(statistics.mean(v), 3) for k, v in scores.items()}
    outs = [r["outputs"] for r in rows]
    tok = [o["tokens"] for o in outs if o.get("tokens")]
    lat = sorted(o["latency_s"] for o in outs if o.get("latency_s"))
    out["_run"] = {
        "examples": len(rows), "errors": sum(1 for o in outs if o.get("error") or not o.get("status")),
        "escalated": sum(1 for o in outs if o.get("status") == "escalated"),
        "clarifications": sum(1 for o in outs if o.get("clarifications")),
        "tokens_mean": round(statistics.mean(tok)) if tok else None, "tokens_total": sum(tok),
        "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "latency_p95_s": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None,
    }
    return out


def write_results(suite: str, rows: list[dict], meta: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{suite}.json"
    doc = {"suite": suite, **meta, "summary": summarize(rows), "examples": rows}
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def print_summary(suite: str, summary: dict) -> None:
    print(f"\n{suite.upper()} - mean over examples:")
    for k, v in summary.items():
        if not k.startswith("_"):
            print(f"  {k:<28} {v}")
    print(f"  run: {summary['_run']}")


# ======================================================================= suites
def suite_config(suite: str):
    from evals.evaluators import ALL_EVALUATORS, SCENARIO_EVALUATORS
    if suite == "heldout":
        return heldout_target, ALL_EVALUATORS
    return scenario_target, SCENARIO_EVALUATORS


def run_langsmith(suite: str, examples: list[dict], prefix: str, concurrency: int) -> Path:
    from langsmith import Client, evaluate
    client = Client()
    target, evaluators = suite_config(suite)
    ds, data = sync_dataset(client, suite, examples)
    print(f"LangSmith dataset '{ds.name}': running {len(data)} example(s), concurrency {concurrency}")
    meta = {"model": settings.groq_model, "fast_model": settings.groq_fast_model, "judge_model": settings.groq_fast_model,
            "retrieval_k": settings.retrieval_k, "max_tool_rounds": settings.max_tool_rounds, "suite": suite}
    res = evaluate(target, data=data, evaluators=evaluators, experiment_prefix=f"{prefix}-{suite}",
                   max_concurrency=concurrency, metadata=meta,
                   description=f"{suite} suite: {len(data)} examples, models {settings.groq_model} / "
                               f"{settings.groq_fast_model}")
    project = client.read_project(project_name=res.experiment_name)
    rows = []
    for r in res:
        run, ex = r["run"], r["example"]
        evs = (r.get("evaluation_results") or {}).get("results") or []
        rows.append({
            "example_id": str(ex.id), "inputs": ex.inputs, "reference": ex.outputs, "metadata": ex.metadata,
            "outputs": compact_outputs(run.outputs), "run_error": run.error,
            "trace_url": client.get_run_url(run=run, project_name=res.experiment_name),
            "scores": {e.key: {"score": e.score, "comment": e.comment} for e in evs},
        })
    path = write_results(suite, rows, {
        "dataset": ds.name, "dataset_url": ds.url, "experiment": res.experiment_name,
        "experiment_url": f"{ds.url}/compare?selectedSessions={project.id}", "finished_at": time.strftime("%Y-%m-%d %H:%M"),
        "evaluators": [e.__name__ for e in evaluators], **meta})
    print(f"\nLangSmith experiment: {res.experiment_name}")
    return path


def run_local(suite: str, examples: list[dict], concurrency: int) -> Path:
    from concurrent.futures import ThreadPoolExecutor
    target, evaluators = suite_config(suite)

    def one(ex: dict) -> dict:
        try:
            out = target(ex["inputs"])
        except Exception as e:  # noqa: BLE001
            out = {"error": f"{type(e).__name__}: {e}"}
        scores = {}
        for ev in evaluators:
            try:
                r = ev(ex["inputs"], out, ex["outputs"])
            except Exception as e:  # noqa: BLE001
                r = {"key": ev.__name__, "score": None, "comment": f"evaluator error: {e}"}
            for item in r.get("results", [r]):
                scores[item["key"]] = {"score": item["score"], "comment": item.get("comment")}
        print(f"  done: {ex['inputs']['subject'][:60]!r}")
        return {"inputs": ex["inputs"], "reference": ex["outputs"], "metadata": ex["metadata"],
                "outputs": compact_outputs(out), "scores": scores}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(one, examples))
    return write_results(suite, rows, {"dataset": DATASETS[suite] + " (local)", "experiment": None,
                                       "finished_at": time.strftime("%Y-%m-%d %H:%M"),
                                       "evaluators": [e.__name__ for e in evaluators],
                                       "model": settings.groq_model, "fast_model": settings.groq_fast_model})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["heldout", "scenarios", "all"], default="all")
    ap.add_argument("--n", type=int, default=None, help="limit the held-out suite to the first N tickets")
    ap.add_argument("--only", action="append", help="scenario case id or scenario name (repeatable)")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel held-out tickets (scenarios always run 1)")
    ap.add_argument("--local", action="store_true", help="run evaluators locally even if LANGSMITH_API_KEY is set")
    ap.add_argument("--prefix", default="support-agent")
    args = ap.parse_args()

    sandbox = make_sandbox()
    from app import db  # noqa: E402  (imports after the sandbox is in place)
    db.init_db()
    GUARD.install()
    print(f"Sandbox: {sandbox} (app.db copy + fresh checkpoints/memory)")
    use_ls = bool(os.getenv("LANGSMITH_API_KEY")) and not args.local
    if not use_ls:
        print("LANGSMITH_API_KEY not set (or --local) -> running evaluators locally")
    suites = ["heldout", "scenarios"] if args.suite == "all" else [args.suite]
    try:
        for suite in suites:
            GUARD.allow = suite == "scenarios"
            if GUARD.allow:
                left = GUARD.purge()
                if left:
                    print(f"Removed {left} leftover eval point(s) from the index")
            examples = heldout_examples(args.n) if suite == "heldout" else scenario_examples(args.only)
            conc = args.concurrency if suite == "heldout" else 1
            t0 = time.time()
            path = run_langsmith(suite, examples, args.prefix, conc) if use_ls else run_local(suite, examples, conc)
            print_summary(suite, json.loads(path.read_text(encoding="utf-8"))["summary"])
            print(f"Results: {path}  ({time.time() - t0:.0f}s)")
    finally:
        GUARD.allow = True
        GUARD.purge()
        shutil.rmtree(sandbox, ignore_errors=True)


if __name__ == "__main__":
    main()
