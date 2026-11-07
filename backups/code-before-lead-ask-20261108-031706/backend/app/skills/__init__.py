"""Reusable skills = instruction modules stored as markdown files with a small front-matter header.

Only a catalog (name + description) is shown to the model when choosing; full instructions of a
skill are loaded into a prompt only after it has been selected for the current case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from ..config import SKILLS_DIR


@dataclass
class Skill:
    name: str
    description: str
    stage: str
    applies_to: list[str] = field(default_factory=list)
    body: str = ""


def _parse(text: str) -> tuple[dict, str]:
    meta: dict = {}
    if text.startswith("---"):
        _, header, body = text.split("---", 2)
        for line in header.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return meta, body.strip()
    return meta, text.strip()


@lru_cache
def all_skills() -> dict[str, Skill]:
    out = {}
    for p in sorted(SKILLS_DIR.glob("*.md")):
        meta, body = _parse(p.read_text(encoding="utf-8"))
        name = meta.get("name", p.stem)
        out[name] = Skill(
            name=name,
            description=meta.get("description", ""),
            stage=meta.get("stage", "investigate"),
            applies_to=[x.strip() for x in meta.get("applies_to", "").split(",") if x.strip()],
            body=body,
        )
    return out


def catalog(stage: str) -> str:
    return "\n".join(f"- {s.name}: {s.description}" for s in all_skills().values() if s.stage == stage)


def names(stage: str) -> list[str]:
    return [s.name for s in all_skills().values() if s.stage == stage]


BRIEF_MARKER = "<!-- brief-end -->"


def render(skill_names: list[str], brief: bool = False) -> str:
    """Full skill instructions. brief=True keeps only the part above BRIEF_MARKER (e.g. the escalation
    criteria while investigating; the hand-off format is only needed when actually escalating)."""
    skills = all_skills()
    parts = []
    for n in skill_names:
        if n in skills:
            body = skills[n].body
            if brief and BRIEF_MARKER in body:
                body = body.split(BRIEF_MARKER, 1)[0].rstrip()
            parts.append(f"### SKILL: {n}\n{body.replace(BRIEF_MARKER, '').strip()}")
    return "\n\n".join(parts)


def default_for(ticket_type: str | None, queue: str | None) -> list[str]:
    """Fallback selection when the model does not pick a valid skill."""
    picks = []
    for s in all_skills().values():
        if s.stage != "investigate":
            continue
        if (ticket_type and ticket_type in s.applies_to) or (queue and queue in s.applies_to):
            picks.append(s.name)
    return picks[:2] or ["incident_investigation"]
