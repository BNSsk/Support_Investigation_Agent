"""Groq chat models with usage accounting (no client-side rate limiting) + robust structured output."""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Sequence, TypeVar, get_origin

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel

from .config import settings
from .ratelimit import groq_usage, httpx_response_hook

T = TypeVar("T", bound=BaseModel)

# Generous output caps (gpt-oss reasoning tokens count as output); they only guard against runaway replies.
MAX_OUT = {False: int(os.getenv("GROQ_MAX_OUT_MAIN", 4096)), True: int(os.getenv("GROQ_MAX_OUT_FAST", 2048))}

# Errors from the API itself (not from the model's output): re-raise instead of retrying in another format.
API_ERRORS = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError",
              "AuthenticationError", "PermissionDeniedError", "NotFoundError"}


def is_api_error(e: BaseException) -> bool:
    return type(e).__name__ in API_ERRORS


class LLMNotConfigured(RuntimeError):
    pass


def model_name(fast: bool) -> str:
    return settings.groq_fast_model if fast else settings.groq_model


@lru_cache(maxsize=4)
def chat(fast: bool = False) -> ChatGroq:
    if not settings.groq_api_key:
        raise LLMNotConfigured("GROQ_API_KEY is not set. Add it to .env and restart.")
    model = model_name(fast)
    kwargs = {}
    if "gpt-oss" in model:
        # low effort keeps latency down; set GROQ_REASONING_EFFORT=medium/high for harder reasoning
        kwargs["reasoning_effort"] = os.getenv("GROQ_REASONING_EFFORT", "low")
    # No client-side pacing: a genuine 429/5xx is retried by the Groq SDK (it honours retry-after).
    return ChatGroq(model=model, temperature=0, max_retries=3, api_key=settings.groq_api_key,
                    max_tokens=MAX_OUT[fast], timeout=120,
                    http_client=httpx.Client(timeout=120, event_hooks={"response": [httpx_response_hook]}),
                    **kwargs)


def _emit(payload: dict) -> None:
    """Send a custom event to the UI stream when running inside the graph."""
    try:
        from langgraph.config import get_stream_writer
        get_stream_writer()(payload)
    except Exception:  # noqa: BLE001 - not inside a graph run
        pass


def _usage(msg: AIMessage | None) -> tuple[int, int, int]:
    if msg is None:
        return 0, 0, 0
    um = getattr(msg, "usage_metadata", None) or {}
    inp, out = um.get("input_tokens", 0), um.get("output_tokens", 0)
    cached = (um.get("input_token_details") or {}).get("cache_read", 0) or 0
    if not cached:
        tu = (getattr(msg, "response_metadata", None) or {}).get("token_usage") or {}
        cached = (tu.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    return inp, out, cached


def _tracked(fast: bool, call, label: str):
    """Run one LLM call and record its token usage (shown per node in the UI)."""
    model = model_name(fast)
    result, raw = call()
    inp, out, cached = _usage(raw)
    counted = groq_usage.record(model, inp, out, cached)
    _emit({"kind": "llm_call", "model": model, "label": label, "input_tokens": inp, "output_tokens": out,
           "cached_tokens": cached, "counted_tokens": counted})
    return result


def invoke_chat(messages: Sequence[BaseMessage], fast: bool = False, tools: list | None = None,
                label: str = "chat") -> AIMessage:
    llm = chat(fast).bind_tools(tools) if tools else chat(fast)

    def call():
        ai = llm.invoke(list(messages))
        return ai, ai

    return _tracked(fast, call, label)


def _salvage(schema: type[T], candidates: list) -> T | None:
    """Recover structured output from a reply that failed strict parsing (tool-call args, JSON in the text
    or in Groq's `failed_generation`) - saves a second LLM call for small format slips."""
    list_fields = {n for n, f in schema.model_fields.items() if get_origin(f.annotation) is list}
    for c in candidates:
        try:
            if isinstance(c, str):
                m = re.search(r"\{.*\}", c, re.S)
                c = json.loads(m.group(0)) if m else None
            if isinstance(c, dict):
                if len(c) == 1 and isinstance(next(iter(c.values())), dict) and next(iter(c)) == schema.__name__:
                    c = next(iter(c.values()))  # {"MemoryOut": {...}}
                c = {k: ([] if v is None and k in list_fields else v) for k, v in c.items()}
                return schema.model_validate(c)
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
    return None


def structured(schema: type[T], messages: Sequence[BaseMessage], fast: bool = False, label: str | None = None) -> T:
    """Pydantic output via tool calling. A reply that fails strict parsing is first repaired locally;
    only if that fails is the call repeated in JSON mode."""
    label = label or schema.__name__
    last_err: Exception | None = None
    for method in ("function_calling", "json_mode"):
        msgs = list(messages)
        if method == "json_mode":
            msgs.append(HumanMessage(content="Return ONLY a JSON object (no prose) that validates against this JSON "
                                             "schema:\n" + json.dumps(schema.model_json_schema())))
        runnable = chat(fast).with_structured_output(schema, method=method, include_raw=True)

        def call():
            out = runnable.invoke(msgs)
            return out, out.get("raw")

        try:
            out = _tracked(fast, call, label)
            parsed = out.get("parsed")
            if parsed is None:
                raw = out.get("raw")
                fixed = _salvage(schema, [tc.get("args") for tc in getattr(raw, "tool_calls", None) or []]
                                 + [tc.get("args") for tc in getattr(raw, "invalid_tool_calls", None) or []]
                                 + [getattr(raw, "content", None)])
                if fixed is not None:
                    _emit({"kind": "llm_note", "label": label, "note": "format slip repaired locally (no retry)"})
                    return fixed
                raise ValueError(f"unparseable output: {out.get('parsing_error')}")
            return schema.model_validate(parsed) if isinstance(parsed, dict) else parsed
        except LLMNotConfigured:
            raise
        except Exception as e:  # noqa: BLE001
            if is_api_error(e):  # rate limit / network / auth: a JSON-mode retry would not help
                raise
            # Groq rejects malformed tool calls with HTTP 400 `tool_use_failed` and returns the text it generated
            body = getattr(e, "body", None)
            failed = ((body or {}).get("error") or {}).get("failed_generation") if isinstance(body, dict) else None
            fixed = _salvage(schema, [failed]) if failed else None
            if fixed is not None:
                _emit({"kind": "llm_note", "label": label, "note": "tool_use_failed repaired locally (no retry)"})
                return fixed
            last_err = e
    raise RuntimeError(f"Structured output failed for {schema.__name__}: {last_err}") from last_err
