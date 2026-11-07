"""Local structured database (SQLite).

Tables
------
historical_tickets : the ~28K English dataset tickets (source of truth for get_ticket_by_id / stats)
cases              : every NEW customer query handled by the agent (open -> resolved/escalated)
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS historical_tickets (
    id TEXT PRIMARY KEY,
    subject TEXT,
    body TEXT,
    answer TEXT,
    type TEXT,
    queue TEXT,
    priority TEXT,
    tags TEXT
);
CREATE INDEX IF NOT EXISTS idx_hist_queue ON historical_tickets(queue);

CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    thread_id TEXT,
    customer_id TEXT,
    subject TEXT,
    body TEXT,
    status TEXT,
    classification TEXT,
    analysis TEXT,
    resolution TEXT,
    evidence TEXT,
    escalation TEXT,
    confidence REAL,
    evaluation TEXT,
    clarifications TEXT,
    indexed INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT,
    lead_guidance TEXT,
    review TEXT
);
CREATE INDEX IF NOT EXISTS idx_cases_customer ON cases(customer_id);
CREATE INDEX IF NOT EXISTS idx_cases_thread ON cases(thread_id);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT,
    payload TEXT,
    ts REAL
);
CREATE INDEX IF NOT EXISTS idx_events_thread ON events(thread_id);
"""

JSON_FIELDS = {"classification", "evidence", "escalation", "evaluation", "clarifications", "lead_guidance", "review"}

# Columns added after the first release: CREATE TABLE IF NOT EXISTS does not add them to an existing database.
ADDED_CASE_COLUMNS = {"lead_guidance": "TEXT", "review": "TEXT"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    conn = sqlite3.connect(settings.app_db, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)
        have = {r[1] for r in c.execute("PRAGMA table_info(cases)")}
        for col, typ in ADDED_CASE_COLUMNS.items():
            if col not in have:
                c.execute(f"ALTER TABLE cases ADD COLUMN {col} {typ}")


def _row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    for k in list(d):
        if k in JSON_FIELDS or k == "tags":
            if d[k]:
                try:
                    d[k] = json.loads(d[k])
                except (TypeError, json.JSONDecodeError):
                    pass
    return d


# ---------------------------------------------------------------- historical
def reset_historical() -> None:
    with connect() as c:
        c.execute("DELETE FROM historical_tickets")


def insert_historical(rows: Iterable[dict]) -> None:
    with connect() as c:
        c.executemany(
            "INSERT OR REPLACE INTO historical_tickets VALUES (?,?,?,?,?,?,?,?)",
            [
                (r["id"], r["subject"], r["body"], r["answer"], r["type"], r["queue"],
                 r["priority"], json.dumps(r["tags"]))
                for r in rows
            ],
        )


def get_ticket(ticket_id: str) -> dict | None:
    """Look up a historical ticket or a live case by id."""
    ticket_id = ticket_id.strip()
    with connect() as c:
        row = c.execute("SELECT * FROM historical_tickets WHERE id=?", (ticket_id,)).fetchone()
        if row:
            d = _row(row)
            d["source"] = "historical"
            return d
        row = c.execute("SELECT * FROM cases WHERE case_id=?", (ticket_id,)).fetchone()
        if row:
            d = _row(row)
            d["source"] = "live"
            return d
    return None


def label_values() -> dict[str, list[str]]:
    """Valid queue/type/priority labels observed in the dataset."""
    out: dict[str, list[str]] = {}
    with connect() as c:
        for col in ("queue", "type", "priority"):
            out[col] = [r[0] for r in c.execute(
                f"SELECT {col}, COUNT(*) n FROM historical_tickets WHERE {col} IS NOT NULL "
                f"GROUP BY {col} ORDER BY n DESC")]
    return out


def category_stats(queue: str | None = None, tag: str | None = None, ticket_type: str | None = None) -> dict:
    """Distribution statistics over historical tickets (optionally filtered)."""
    where, params = [], []
    if queue:
        where.append("queue = ?"); params.append(queue)
    if ticket_type:
        where.append("type = ?"); params.append(ticket_type)
    if tag:
        where.append("tags LIKE ?"); params.append(f'%"{tag}"%')
    sql = "SELECT queue, type, priority, tags FROM historical_tickets"
    if where:
        sql += " WHERE " + " AND ".join(where)
    with connect() as c:
        rows = c.execute(sql, params).fetchall()
    n = len(rows)
    q, t, p, tg = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        q[r["queue"]] += 1; t[r["type"]] += 1; p[r["priority"]] += 1
        for x in json.loads(r["tags"] or "[]"):
            tg[x] += 1
    pct = lambda cnt: {k: round(v / n, 3) for k, v in cnt.most_common(6)} if n else {}
    return {"filter": {"queue": queue, "tag": tag, "type": ticket_type}, "total_tickets": n,
            "queue_share": pct(q), "type_share": pct(t), "priority_share": pct(p),
            "top_tags": dict(tg.most_common(10))}


def counts() -> dict:
    with connect() as c:
        hist = c.execute("SELECT COUNT(*) FROM historical_tickets").fetchone()[0]
        cases = c.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        resolved = c.execute("SELECT COUNT(*) FROM cases WHERE status='resolved'").fetchone()[0]
        indexed = c.execute("SELECT COUNT(*) FROM cases WHERE indexed=1").fetchone()[0]
        lead = c.execute("SELECT COUNT(*) FROM cases WHERE indexed=1 AND lead_guidance IS NOT NULL "
                         "AND lead_guidance != 'null'").fetchone()[0]
    return {"historical": hist, "cases": cases, "resolved": resolved, "indexed_live": indexed,
            "lead_verified": lead}


# ---------------------------------------------------------------- live cases
def create_case(thread_id: str, customer_id: str, subject: str, body: str) -> str:
    case_id = f"CASE-{datetime.now():%y%m%d}-{uuid.uuid4().hex[:6].upper()}"
    ts = now()
    with connect() as c:
        c.execute(
            "INSERT INTO cases (case_id, thread_id, customer_id, subject, body, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (case_id, thread_id, customer_id, subject, body, "open", ts, ts),
        )
    return case_id


def update_case(case_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = now()
    cols, vals = [], []
    for k, v in fields.items():
        cols.append(f"{k}=?")
        vals.append(json.dumps(v) if k in JSON_FIELDS else v)
    vals.append(case_id)
    with connect() as c:
        c.execute(f"UPDATE cases SET {', '.join(cols)} WHERE case_id=?", vals)


def get_case(case_id: str) -> dict | None:
    with connect() as c:
        return _row(c.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone())


def list_cases(limit: int = 50) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT case_id, thread_id, customer_id, subject, status, confidence, classification, updated_at"
            " FROM cases ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def log_event(thread_id: str, event: dict) -> None:
    with connect() as c:
        c.execute("INSERT INTO events (thread_id, payload, ts) VALUES (?,?,?)",
                  (thread_id, json.dumps(event, default=str), event.get("ts")))


def thread_events(thread_id: str) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT payload FROM events WHERE thread_id=? ORDER BY id", (thread_id,)).fetchall()
    return [json.loads(r[0]) for r in rows]


def customer_cases(customer_id: str, exclude_case: str | None = None, limit: int = 10) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT case_id, subject, body, status, classification, resolution, created_at FROM cases"
            " WHERE customer_id=? AND case_id != ? ORDER BY created_at DESC LIMIT ?",
            (customer_id, exclude_case or "", limit)).fetchall()
    return [_row(r) for r in rows]
