"""Label vocabulary (from the dataset) + normalisation of model output to valid labels."""
from __future__ import annotations

import difflib
from collections import Counter, defaultdict
from functools import lru_cache

from . import db

FALLBACK = {
    "queue": ["Technical Support", "Product Support", "Customer Service", "IT Support", "Billing and Payments",
              "Returns and Exchanges", "Service Outages and Maintenance", "Sales and Pre-Sales",
              "Human Resources", "General Inquiry"],
    "type": ["Incident", "Request", "Problem", "Change"],
    "priority": ["high", "medium", "low"],
}


@lru_cache
def vocabulary() -> dict[str, list[str]]:
    try:
        v = db.label_values()
        if all(v.get(k) for k in FALLBACK):
            return v
    except Exception:  # noqa: BLE001
        pass
    return FALLBACK


def normalize(kind: str, value: str | None) -> str | None:
    if not value:
        return None
    options = vocabulary()[kind]
    lowered = {o.lower(): o for o in options}
    if value.lower() in lowered:
        return lowered[value.lower()]
    match = difflib.get_close_matches(value, options, n=1, cutoff=0.5)
    return match[0] if match else None


def votes(hits: list[dict]) -> dict:
    """Similarity-weighted label distribution of retrieved cases (a prior for triage)."""
    out: dict[str, Counter] = defaultdict(Counter)
    for h in hits:
        w = max(h.get("similarity") or 0.0, 0.01)
        for kind in ("queue", "type", "priority"):
            if h.get(kind):
                out[kind][h[kind]] += w
        for t in h.get("tags") or []:
            out["tags"][t] += w
    res = {}
    for kind, cnt in out.items():
        total = sum(cnt.values()) or 1
        res[kind] = {k: round(v / total, 2) for k, v in cnt.most_common(5 if kind == "tags" else 4)}
    return res
