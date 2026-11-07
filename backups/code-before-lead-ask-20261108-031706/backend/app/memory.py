"""Long-term, customer-level memory on top of the LangGraph Store (SQLite-backed).

Namespace: ("customers", <customer_id>, "facts"); one item per fact key.
Only durable, support-relevant facts are kept (product/environment/plan/preferences/recurring
issues). Secrets and payment data are filtered out deterministically, and the number of facts per
customer is capped so memory stays small and relevant.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from langgraph.store.base import BaseStore

from .config import settings

SENSITIVE = [
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),                        # card-like numbers
    re.compile(r"(?i)\b(password|passwd|pwd|passcode|pin|cvv|secret|api[_ -]?key|token)\b"),
    re.compile(r"(?i)\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),          # IBAN-like
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                         # SSN-like
]


def ns(customer_id: str) -> tuple[str, ...]:
    return ("customers", customer_id or "anonymous", "facts")


def slug(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")[:48] or "fact"


def is_sensitive(text: str) -> bool:
    return any(p.search(text or "") for p in SENSITIVE)


def get_facts(store: BaseStore, customer_id: str) -> list[dict]:
    items = store.search(ns(customer_id), limit=100)
    facts = [{"key": it.key, **(it.value or {}),
              "updated_at": it.updated_at.isoformat() if it.updated_at else None} for it in items]
    return sorted(facts, key=lambda f: f.get("updated_at") or "", reverse=True)


def apply_update(store: BaseStore, customer_id: str, facts: list[dict], forget: list[str],
                 source_case: str) -> dict:
    added, skipped, removed = [], [], []
    for key in forget or []:
        store.delete(ns(customer_id), slug(key))
        removed.append(slug(key))
    for f in facts or []:
        key, value = slug(f.get("key", "")), (f.get("value") or "").strip()
        if not value or is_sensitive(value) or len(value) > 300:
            skipped.append(key)
            continue
        store.put(ns(customer_id), key, {"value": value, "source_case": source_case,
                                          "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        added.append({"key": key, "value": value})
    # cap memory size: drop the oldest facts beyond the limit
    current = get_facts(store, customer_id)
    for old in current[settings.max_memory_facts:]:
        store.delete(ns(customer_id), old["key"])
        removed.append(old["key"])
    return {"saved": added, "skipped_sensitive_or_empty": skipped, "removed": removed}


def clear(store: BaseStore, customer_id: str) -> int:
    facts = get_facts(store, customer_id)
    for f in facts:
        store.delete(ns(customer_id), f["key"])
    return len(facts)
