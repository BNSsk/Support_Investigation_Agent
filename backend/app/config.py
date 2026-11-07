"""Central configuration. Secrets come from <project root>/.env (or backend/.env); never committed."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FRONTEND_DIR = ROOT_DIR / "frontend"
SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# Accepted names for the same setting (first one is canonical).
ALIASES = {"GROQ_MODEL": ["LLM_MAIN"], "GROQ_FAST_MODEL": ["LLM_FAST"]}


def _warn_duplicates(path: Path) -> None:
    """python-dotenv silently keeps the LAST of duplicated keys - make that visible (values masked)."""
    seen: dict[str, list[str]] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            seen.setdefault(k.strip(), []).append(v.strip().strip('"').strip("'"))
    for k, vals in seen.items():
        if len(vals) > 1 and len(set(vals)) > 1:
            masked = ", ".join(f"...{v[-4:]}" if v else "(empty)" for v in vals)
            print(f"WARNING: {k} is defined {len(vals)} times in {path} ({masked}); "
                  f"the LAST one (...{vals[-1][-4:]}) is used. Remove the others.")


def _load_env() -> None:
    """Merge <root>/.env and backend/.env into os.environ.
    Precedence: real environment > root .env > backend/.env. Empty values never override."""
    for path in (ROOT_DIR / ".env", BACKEND_DIR / ".env"):
        if not path.exists():
            continue
        _warn_duplicates(path)
        for key, value in dotenv_values(path).items():
            if value and value.strip() and not os.environ.get(key, "").strip():
                os.environ[key] = value.strip()
    for canonical, alts in ALIASES.items():
        if not os.environ.get(canonical):
            for alt in alts:
                if os.environ.get(alt):
                    os.environ[canonical] = os.environ[alt]
                    break


_load_env()
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


class Settings:
    # --- LLM (Groq) ---
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    # Main model: investigation (tool calling), drafting, escalation.
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    # Fast model: triage, evidence check (judge), memory extraction.
    groq_fast_model: str = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")

    # --- Vector DB (Qdrant) ---
    # If QDRANT_URL is set we use a Qdrant server (Docker / Cloud); otherwise embedded local mode.
    qdrant_url: str | None = os.getenv("QDRANT_URL") or None
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_path: str = os.getenv("QDRANT_PATH", str(DATA_DIR / "qdrant"))
    collection: str = os.getenv("QDRANT_COLLECTION", "support_cases")
    # Client-side request rate towards a Qdrant server (free cloud cluster: small CPU/RAM). 0 = unlimited.
    qdrant_max_rps: float = float(os.getenv("QDRANT_MAX_RPS", "5"))
    qdrant_upload_batch: int = _int("QDRANT_UPLOAD_BATCH", 128)
    dense_model: str = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    sparse_model: str = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

    # --- Local structured storage ---
    app_db: Path = DATA_DIR / "app.db"            # historical tickets + live cases
    checkpoint_db: Path = DATA_DIR / "checkpoints.db"  # LangGraph short-term memory
    memory_db: Path = DATA_DIR / "memory.db"      # LangGraph store: long-term customer memory
    eval_holdout: Path = DATA_DIR / "eval_holdout.jsonl"

    # --- Agent budgets ---
    retrieval_k: int = _int("RETRIEVAL_K", 6)
    max_tool_rounds: int = _int("MAX_TOOL_ROUNDS", 2)    # loop guard for the ReAct loop (raise for deeper digging)
    max_eval_retries: int = _int("MAX_EVAL_RETRIES", 1)  # evidence-check retries before forced escalation
    max_memory_facts: int = _int("MAX_MEMORY_FACTS", 10)


settings = Settings()
