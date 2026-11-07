"""Copy the embedded (local) Qdrant collection to the Qdrant server in QDRANT_URL (e.g. Qdrant Cloud)
without re-computing embeddings. Rate-limited (QDRANT_MAX_RPS, QDRANT_UPLOAD_BATCH) and idempotent,
so it can simply be re-run if interrupted.

    python scripts/sync_qdrant.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import models  # noqa: E402

from app.config import settings  # noqa: E402
from app.vectorstore import CaseIndex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-collection", help="local collection name (auto-detected if omitted)")
    args = ap.parse_args()
    if not settings.qdrant_url:
        sys.exit("QDRANT_URL is not set - nothing to sync to.")
    src = CaseIndex(url=None)                 # embedded store in backend/data/qdrant
    dst = CaseIndex()                         # server from QDRANT_URL
    local = [c.name for c in src.client.get_collections().collections]
    source = args.source_collection or (settings.collection if settings.collection in local else
                                        (local[0] if len(local) == 1 else None))
    if not source:
        sys.exit(f"Cannot pick the local collection automatically from {local}; pass --source-collection.")
    total = src.client.count(source, exact=True).count
    print(f"source (embedded): {source} = {total:,} points | target: Qdrant server / {settings.collection}")
    print(f"throttle: {settings.qdrant_max_rps} req/s, {settings.qdrant_upload_batch} points/request")
    dst.ensure_collection(dim=384)            # creates collection + strict-mode payload indexes

    t0, sent, offset = time.time(), 0, None
    while True:
        recs, offset = src.client.scroll(source, limit=512, offset=offset,
                                         with_payload=True, with_vectors=True)
        if not recs:
            break
        dst.upsert_points([models.PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in recs])
        sent += len(recs)
        el = time.time() - t0
        print(f"  {sent:,}/{total:,}  {sent / el:,.0f} pts/s  retries={dst.throttle.retries}", flush=True)
        if offset is None:
            break

    n = dst.count(max_age=0)
    print(f"\ntarget now has {n:,} points (source {total:,}) in {time.time() - t0:,.0f}s, retries={dst.throttle.retries}")
    hits = dst.search("VPN disconnects after firmware update", k=3, queue="Technical Support")
    print("filtered hybrid query on target:", [(h["case_id"], h["similarity"]) for h in hits])
    src.client.close()


if __name__ == "__main__":
    main()
