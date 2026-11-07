"""Qdrant case index with hybrid retrieval.

* Dense vectors  : BAAI/bge-small-en-v1.5 (384-d, local ONNX via fastembed)
* Sparse vectors : BM25 (catches exact error codes, product names, ids)
* Fusion         : Reciprocal Rank Fusion inside Qdrant (Query API prefetch + RRF)
* Payload filters: queue / type / priority / tags / source  (metadata-aware retrieval)

The same collection holds the historical dataset (source="historical") and newly resolved,
human-accepted live cases (source="live") -> the knowledge base keeps growing.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from .config import settings
from .ratelimit import QdrantThrottle

_NS = uuid.UUID("6f1c2a4e-1d1b-4c55-9d0e-6a2b8f0c7e11")
# Strict mode on Qdrant Cloud rejects filters on non-indexed payload fields -> index every filtered field.
INDEXED_FIELDS = ("queue", "type", "priority", "tags", "source", "case_id")


def point_id(case_id: str) -> str:
    return str(uuid.uuid5(_NS, case_id))


def case_text(subject: str | None, body: str | None) -> str:
    return f"{(subject or '').strip()}\n{(body or '').strip()}".strip()


class CaseIndex:
    _lock = threading.Lock()

    def __init__(self, url: str | None = "env", path: str | None = None) -> None:
        url = settings.qdrant_url if url == "env" else url
        if url:
            self.client = QdrantClient(url=url, api_key=settings.qdrant_api_key, timeout=60)
            self.local = False
            self.throttle = QdrantThrottle(settings.qdrant_max_rps)
        else:
            self.client = QdrantClient(path=path or settings.qdrant_path)
            self.local = True
            self.throttle = QdrantThrottle(0)
        self._dense = None
        self._sparse = None
        self._count_cache: dict[str | None, tuple[float, int]] = {}

    def _q(self, fn, *args, **kwargs):
        """Every Qdrant call goes through the client-side rate limiter + retry/backoff."""
        return self.throttle.call(fn, *args, **kwargs)

    # ------------------------------------------------------------ embedders
    @property
    def dense(self):
        if self._dense is None:
            from fastembed import TextEmbedding
            self._dense = TextEmbedding(settings.dense_model)
        return self._dense

    @property
    def sparse(self):
        if self._sparse is None:
            from fastembed import SparseTextEmbedding
            self._sparse = SparseTextEmbedding(settings.sparse_model)
        return self._sparse

    def embed_dense(self, texts: list[str], batch_size: int = 128) -> list[np.ndarray]:
        return list(self.dense.embed(texts, batch_size=batch_size))

    def embed_sparse(self, texts: list[str], batch_size: int = 256) -> list[models.SparseVector]:
        return [models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
                for e in self.sparse.embed(texts, batch_size=batch_size)]

    # ------------------------------------------------------------ collection
    def exists(self) -> bool:
        if getattr(self, "_exists", False):  # cached once true: saves one request per search
            return True
        self._exists = bool(self._q(self.client.collection_exists, settings.collection))
        return self._exists

    def ensure_collection(self, recreate: bool = False, dim: int | None = None) -> None:
        if recreate and self.exists():
            self._q(self.client.delete_collection, settings.collection)
            self._exists = False
        if not self.exists():
            dim = dim or len(self.embed_dense(["dimension probe"])[0])
            self._q(self.client.create_collection,
                    settings.collection,
                    vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
                    sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
        self.ensure_payload_indexes()

    def ensure_payload_indexes(self) -> None:
        if self.local:  # payload indexes are a server feature
            return
        existing = set((self._q(self.client.get_collection, settings.collection).payload_schema or {}).keys())
        for field in INDEXED_FIELDS:
            if field not in existing:
                self._q(self.client.create_payload_index, settings.collection, field,
                        models.PayloadSchemaType.KEYWORD, wait=True)

    def count(self, source: str | None = None, max_age: float = 30.0) -> int:
        hit = self._count_cache.get(source)
        if hit and time.time() - hit[0] < max_age:  # the UI polls /health; don't spend RPS on it
            return hit[1]
        if not self.exists():
            return 0
        flt = None
        if source:
            flt = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
        n = self._q(self.client.count, settings.collection, count_filter=flt, exact=True).count
        self._count_cache[source] = (time.time(), n)
        return n

    # ------------------------------------------------------------ write
    def upsert_points(self, points: list[models.PointStruct]) -> None:
        """Write pre-built points in rate-limited batches (QDRANT_UPLOAD_BATCH per request)."""
        step = settings.qdrant_upload_batch if not self.local else 512
        for i in range(0, len(points), step):
            with self._lock:
                self._q(self.client.upsert, settings.collection, points=points[i:i + step], wait=True)
        self._count_cache.clear()

    def upsert_cases(self, records: list[dict], batch_size: int = 256, progress=None) -> None:
        """records: dicts with case_id, subject, body, answer, queue, type, priority, tags, source."""
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            texts = [case_text(r.get("subject"), r.get("body")) for r in batch]
            # Dense: first ~1200 chars carry the issue; small batches limit padding cost on CPU.
            dense = self.embed_dense([t[:1200] for t in texts], batch_size=32)
            sparse = self.embed_sparse(texts)  # BM25 sees the full text
            points = [
                models.PointStruct(
                    id=point_id(r["case_id"]),
                    vector={"dense": d.tolist(), "bm25": s},
                    payload={k: r.get(k) for k in
                             ("case_id", "subject", "body", "answer", "queue", "type", "priority", "tags",
                              "source", "resolved_at", "customer_id",
                              "lead_verified", "root_cause", "lead_solution")},
                )
                for r, d, s in zip(batch, dense, sparse)
            ]
            self.upsert_points(points)
            if progress:
                progress(min(start + batch_size, len(records)), len(records))

    # ------------------------------------------------------------ read
    @staticmethod
    def build_filter(queue=None, ticket_type=None, priority=None, tag=None, source=None,
                     exclude_ids: list[str] | None = None) -> models.Filter | None:
        must, must_not = [], []
        for key, val in (("queue", queue), ("type", ticket_type), ("priority", priority), ("source", source)):
            if val:
                must.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))
        if tag:
            must.append(models.FieldCondition(key="tags", match=models.MatchAny(any=[tag])))
        if exclude_ids:
            must_not.append(models.FieldCondition(key="case_id", match=models.MatchAny(any=exclude_ids)))
        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)

    def search(self, query: str, k: int = 6, *, queue=None, ticket_type=None, priority=None, tag=None,
               source=None, exclude_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Hybrid (dense + BM25, RRF-fused) search. Returns de-duplicated hits with cosine similarity."""
        if not query.strip() or not self.exists():
            return []
        qd = self.embed_dense([query])[0]
        qs = self.embed_sparse([query])[0]
        flt = self.build_filter(queue, ticket_type, priority, tag, source, exclude_ids)
        pre = max(k * 4, 20)
        with self._lock:
            res = self._q(
                self.client.query_points,
                settings.collection,
                prefetch=[
                    models.Prefetch(query=qd.tolist(), using="dense", limit=pre, filter=flt),
                    models.Prefetch(query=qs, using="bm25", limit=pre, filter=flt),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k * 3,
                with_payload=True,
                with_vectors=["dense"],
            )
        hits, seen = [], set()
        for rank, p in enumerate(res.points):
            pl = p.payload or {}
            sig = (pl.get("subject") or "")[:60].lower() + "|" + (pl.get("body") or "")[:80].lower()
            if sig in seen:  # dataset contains near-duplicate tickets
                continue
            seen.add(sig)
            vec = p.vector.get("dense") if isinstance(p.vector, dict) else None
            sim = float(np.dot(qd, np.asarray(vec))) if vec is not None else None
            hits.append({**pl, "similarity": round(sim, 4) if sim is not None else None,
                         "rrf_score": round(float(p.score), 4), "rank": rank + 1})
            if len(hits) >= k:
                break
        return hits


_index: CaseIndex | None = None
_index_lock = threading.Lock()


def get_index() -> CaseIndex:
    global _index
    with _index_lock:
        if _index is None:
            _index = CaseIndex()
        return _index
