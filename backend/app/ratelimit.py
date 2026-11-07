"""Groq usage accounting + Qdrant request throttling.

Groq: NO client-side limits. The account is on a paid plan, so the app never waits for, trims or refuses a
call because of token or request counts. What remains is accounting for the UI and /api/health:
  * every call's input / output / cached tokens are stored per day, per API key (sha256 fingerprint, never
    the key itself) and per model in SQLite (table llm_usage_by_key);
  * the plan's real limits are read from Groq's x-ratelimit-* response headers and shown (never enforced);
  * a genuine 429 from Groq is retried by the Groq SDK (it honours retry-after) and counted here.

Qdrant: simple min-interval spacing (QDRANT_MAX_RPS) + retry with exponential backoff on transient errors.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone

from .config import settings


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def key_fingerprint(api_key: str) -> str:
    """Stable, non-reversible id of an API key: usage is tracked per key (never the key itself)."""
    return hashlib.sha256((api_key or "").encode()).hexdigest()[:12] if api_key else "no-key"


class GroqUsage:
    """Records token usage and the limits Groq reports. It never blocks or delays a call."""

    def __init__(self, api_key: str | None = None) -> None:
        self.key_id = key_fingerprint(settings.groq_api_key if api_key is None else api_key)
        self._lock = threading.Lock()
        self._models: dict[str, dict] = {}   # model -> limits/remaining from the latest response headers
        self._db_init()

    # ------------------------------------------------------------- persistence (per API key)
    def _db(self):
        return sqlite3.connect(settings.app_db, timeout=30)

    def _db_init(self) -> None:
        with self._db() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS llm_usage_by_key (
                day TEXT, key_id TEXT, model TEXT, requests INTEGER DEFAULT 0, input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0, cached_tokens INTEGER DEFAULT 0, PRIMARY KEY (day, key_id, model))""")

    def usage_today(self, model: str | None = None) -> dict:
        with self._db() as c:
            rows = c.execute("SELECT model, requests, input_tokens, output_tokens, cached_tokens FROM llm_usage_by_key "
                             "WHERE day=? AND key_id=?", (today(), self.key_id)).fetchall()
        out = {r[0]: {"requests": r[1], "input_tokens": r[2], "output_tokens": r[3], "cached_tokens": r[4],
                      "counted_tokens": r[2] + r[3] - r[4]} for r in rows}
        return out.get(model, {"requests": 0, "counted_tokens": 0}) if model else out

    def record(self, model: str, inp: int, out: int, cached: int) -> int:
        """Store one call's usage; returns the tokens Groq counts (cached prompt tokens excluded)."""
        try:
            with self._db() as c:
                c.execute("""INSERT INTO llm_usage_by_key (day, key_id, model, requests, input_tokens, output_tokens,
                             cached_tokens) VALUES (?,?,?,1,?,?,?) ON CONFLICT(day, key_id, model) DO UPDATE SET
                             requests=requests+1, input_tokens=input_tokens+excluded.input_tokens,
                             output_tokens=output_tokens+excluded.output_tokens,
                             cached_tokens=cached_tokens+excluded.cached_tokens""",
                          (today(), self.key_id, model, inp, out, cached))
        except sqlite3.Error:
            pass
        return max(inp + out - cached, 0)

    # ------------------------------------------------------------- response headers (informational)
    def on_response(self, model: str, status: int, headers) -> None:
        def num(name):
            v = headers.get(name)
            try:
                return int(float(v)) if v is not None else None
            except ValueError:
                return None
        with self._lock:
            m = self._models.setdefault(model, {"groq_429s_since_start": 0})
            for key, header in (("tokens_per_minute", "x-ratelimit-limit-tokens"),
                                ("requests_limit", "x-ratelimit-limit-requests"),
                                ("tokens_remaining_this_minute", "x-ratelimit-remaining-tokens"),
                                ("requests_remaining", "x-ratelimit-remaining-requests")):
                if num(header) is not None:
                    m[key] = num(header)
            if status == 429:
                m["groq_429s_since_start"] += 1
            m["seen_at"] = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            models = {k: dict(v) for k, v in self._models.items()}
        return {"key_id": self.key_id, "client_limits": "off (paid plan: limits are only reported)",
                "models": models, "today": self.usage_today()}


groq_usage = GroqUsage()


def httpx_response_hook(response) -> None:
    """httpx event hook: read Groq's rate-limit headers for display."""
    try:
        model = json.loads(response.request.content or b"{}").get("model", "?")
        groq_usage.on_response(model, response.status_code, response.headers)
    except Exception:  # noqa: BLE001
        pass


# =================================================================== Qdrant
class QdrantThrottle:
    """Spaces requests to at most `rps` per second and retries transient failures with backoff."""

    def __init__(self, rps: float) -> None:
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()
        self.retries = 0

    def call(self, fn, *args, attempts: int = 6, **kwargs):
        delay = 1.0
        for i in range(attempts):
            if self.min_interval:
                with self._lock:
                    now = time.time()
                    wait = self._next - now
                    self._next = max(now, self._next) + self.min_interval
                if wait > 0:
                    time.sleep(wait)
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                transient = any(s in msg for s in ("429", "too many", "503", "502", "504", "timed out", "timeout",
                                                   "connection", "temporarily", "rate", "getaddrinfo",
                                                   "name resolution", "nodename", "server disconnected",
                                                   "remoteprotocol", "ssl"))
                if not transient or i == attempts - 1:
                    raise
                self.retries += 1
                time.sleep(delay)
                delay = min(delay * 2, 20)
