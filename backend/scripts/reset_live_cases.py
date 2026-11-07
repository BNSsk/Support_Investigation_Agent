"""Remove the trusted "live" cases (accepted resolutions indexed by earlier runs) from the vector index, so the
demos start again from the historical dataset only. Run it after rehearsals, before a demo: each accepted
demo answer is indexed and changes how the next run of the same scenario behaves (e.g. a vague ticket can
be answered from an earlier clarified copy instead of triggering a clarification).

    python scripts/reset_live_cases.py          # list what would be removed (no changes)
    python scripts/reset_live_cases.py --yes    # remove them
    python scripts/reset_live_cases.py --case-id CASE-261104-ABC123 --yes   # remove only these case(s)
    python scripts/reset_live_cases.py --all --yes   # FRESH START: also delete every case, its event log,
                                                     # all checkpoints and long-term customer memory
    python scripts/reset_live_cases.py --restore ../backups/reset-20261107-191909   # undo an --all reset

Default: case rows stay in app.db (marked not indexed) so the case list and audit trail are kept, and
historical tickets, long-term customer memory and checkpoints are not touched.
--all: a full backup (the three SQLite files + the live Qdrant points with their vectors) is written to
<project root>/backups/reset-<timestamp>/ first. Historical tickets and the LLM usage log are always kept.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import models  # noqa: E402

from app.config import ROOT_DIR, settings  # noqa: E402
from app.vectorstore import get_index  # noqa: E402

LIVE = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value="live"))])


def live_filter(case_ids: list[str] | None) -> models.Filter:
    """Only live points (never historical); optionally only the given case ids."""
    if not case_ids:
        return LIVE
    return models.Filter(must=[LIVE.must[0], models.FieldCondition(key="case_id", match=models.MatchAny(any=case_ids))])


def live_points(idx, flt: models.Filter = LIVE, full: bool = False) -> list:
    out, offset = [], None
    while True:
        pts, offset = idx._q(idx.client.scroll, settings.collection, scroll_filter=flt, limit=200, offset=offset,
                             with_payload=True if full else ["case_id", "customer_id", "subject"], with_vectors=full)
        out += pts
        if offset is None:
            return out


def _vec(v):
    return v.model_dump() if hasattr(v, "model_dump") else v


def backup(idx) -> Path:
    """Consistent copies of the SQLite stores + every live Qdrant point (payload and vectors) as JSON."""
    dest = ROOT_DIR / "backups" / f"reset-{time.strftime('%Y%m%d-%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=True)
    for path in (settings.app_db, settings.checkpoint_db, settings.memory_db):
        if path.exists():
            with sqlite3.connect(path) as src, sqlite3.connect(dest / path.name) as dst:
                src.backup(dst)
    pts = live_points(idx, LIVE, full=True)
    dump = [{"id": str(p.id), "payload": p.payload,
             "vector": {k: _vec(v) for k, v in (p.vector or {}).items()} if isinstance(p.vector, dict) else p.vector}
            for p in pts]
    (dest / f"qdrant_live_points_{settings.collection}.json").write_text(json.dumps(dump), encoding="utf-8")
    return dest


def wipe_local_stores() -> dict:
    """Delete every case + event log, all checkpoints and all long-term memory (historical tickets stay)."""
    out = {}
    with sqlite3.connect(settings.app_db) as c:
        out["cases"] = c.execute("DELETE FROM cases").rowcount
        out["events"] = c.execute("DELETE FROM events").rowcount
    with sqlite3.connect(settings.checkpoint_db) as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for t in ("checkpoints", "writes"):
            if t in tables:
                out[f"checkpoint_{t}"] = c.execute(f"DELETE FROM {t}").rowcount
    with sqlite3.connect(settings.memory_db) as c:
        out["memory_items"] = c.execute("DELETE FROM store").rowcount
    for path in (settings.app_db, settings.checkpoint_db, settings.memory_db):
        c = sqlite3.connect(path, isolation_level=None)
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        c.execute("VACUUM")  # drop the deleted rows from disk too
        c.close()
    return out


def restore(src: Path) -> None:
    """Put back what an --all reset removed: rows are merged (INSERT OR IGNORE), nothing current is overwritten.
    Historical tickets and the LLM usage log are not touched."""
    for live, name, tables in ((settings.app_db, "app.db", ("cases", "events")),
                               (settings.checkpoint_db, "checkpoints.db", ("checkpoints", "writes")),
                               (settings.memory_db, "memory.db", ("store",))):
        bak = src / name
        if not bak.exists():
            print(f"  {name}: not in backup, skipped")
            continue
        c = sqlite3.connect(live)
        c.execute("ATTACH DATABASE ? AS bak", (str(bak),))
        for t in tables:
            cols = [r[1] for r in c.execute(f"PRAGMA bak.table_info({t})")]
            if not cols:
                continue
            before = c.execute(f"SELECT COUNT(*) FROM main.{t}").fetchone()[0]
            col_list = ", ".join(f'"{x}"' for x in cols)
            c.execute(f"INSERT OR IGNORE INTO main.{t} ({col_list}) SELECT {col_list} FROM bak.{t}")
            after = c.execute(f"SELECT COUNT(*) FROM main.{t}").fetchone()[0]
            print(f"  {name}/{t}: +{after - before} row(s) (now {after})")
        c.commit()
        c.execute("DETACH DATABASE bak")
        c.close()
    dump = src / f"qdrant_live_points_{settings.collection}.json"
    if not dump.exists():
        print(f"  {dump.name}: not in backup, no index points restored")
        return
    pts = json.loads(dump.read_text(encoding="utf-8"))
    points = [models.PointStruct(
        id=p["id"], payload=p["payload"],
        vector={k: (models.SparseVector(**v) if isinstance(v, dict) else v) for k, v in p["vector"].items()})
        for p in pts]
    idx = get_index()
    if points:
        idx.upsert_points(points)
    print(f"  Qdrant {settings.collection}: {len(points)} live point(s) upserted; live cases now: {idx.count('live', max_age=0)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually remove (default: list only)")
    ap.add_argument("--case-id", action="append", help="remove only this live case (repeatable)")
    ap.add_argument("--all", action="store_true",
                    help="fresh start: also delete all cases, event logs, checkpoints and customer memory (backed up)")
    ap.add_argument("--restore", metavar="BACKUP_DIR",
                    help="put back the cases, logs, checkpoints, memory and live index points from an --all backup")
    args = ap.parse_args()
    if args.restore:
        src = Path(args.restore).resolve()
        if not (src / "app.db").exists():
            ap.error(f"{src} is not a backup folder (no app.db in it)")
        print(f"Restoring from {src}")
        restore(src)
        return
    if args.all and args.case_id:
        ap.error("--all cannot be combined with --case-id")

    idx = get_index()
    flt = live_filter(args.case_id)
    pts = live_points(idx, flt)
    ids = sorted({p.payload.get("case_id") for p in pts if p.payload.get("case_id")})
    print(f"collection {settings.collection}: {len(pts)} live case(s)")
    for p in sorted(pts, key=lambda p: p.payload.get("case_id") or ""):
        print(f"  {p.payload.get('case_id')} | {p.payload.get('customer_id')} | {(p.payload.get('subject') or '')[:70]}")
    if args.all:
        with sqlite3.connect(f"file:{settings.app_db}?mode=ro", uri=True) as c:
            n_cases = c.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        with sqlite3.connect(f"file:{settings.memory_db}?mode=ro", uri=True) as c:
            n_mem = c.execute("SELECT COUNT(*) FROM store").fetchone()[0]
        print(f"--all: {n_cases} case row(s) + their event logs, all checkpoints, {n_mem} memory item(s)")
    if not pts and not args.all:
        return
    if not args.yes:
        print("\nNothing removed. Re-run with --yes to remove them.")
        return

    if args.all:
        print(f"\nBackup written to {backup(idx)}")
    if pts:
        idx._q(idx.client.delete, settings.collection, points_selector=models.FilterSelector(filter=flt), wait=True)
        idx._count_cache.clear()
    if args.all:
        print(f"Deleted from local stores: {wipe_local_stores()}")
        print(f"Removed {len(pts)} point(s) from Qdrant. Live cases left: {idx.count('live', max_age=0)}")
        return
    with sqlite3.connect(settings.app_db) as c:
        n = c.execute(f"UPDATE cases SET indexed=0 WHERE case_id IN ({','.join('?' * len(ids))})", ids).rowcount
    left = idx.count("live", max_age=0)
    print(f"\nRemoved {len(pts)} point(s) from Qdrant; {n} case row(s) marked not indexed. Live cases left: {left}")


if __name__ == "__main__":
    main()
