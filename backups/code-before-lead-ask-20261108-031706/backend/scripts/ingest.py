"""Ingest + index the historical ticket dataset.

    python scripts/ingest.py                # full English subset (~28K)
    python scripts/ingest.py --limit 3000   # quick dev run

Steps: download CSVs (if missing) -> keep English -> clean/dedupe -> hold out an eval split ->
store in SQLite (historical_tickets) -> embed (dense + BM25) -> upsert into Qdrant.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.config import RAW_DIR, settings  # noqa: E402
from app.vectorstore import get_index  # noqa: E402

REPO = "Tobi-Bueck/customer-support-tickets"
FILES = ["aa_dataset-tickets-multi-lang-5-2-50-version.csv", "dataset-tickets-multi-lang-4-20k.csv"]
TAG_COLS = [f"tag_{i}" for i in range(1, 9)]


def download() -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in FILES:
        p = RAW_DIR / f
        if not p.exists():
            from huggingface_hub import hf_hub_download
            print(f"Downloading {f} ...")
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=RAW_DIR)
        paths.append(p)
    return paths


def clean(s) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\\n", "\n").strip()


def load_english(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["language"] == "en"].copy()
    for col in ("subject", "body", "answer"):
        df[col] = df[col].map(clean)
    df = df[(df["body"].str.len() > 20) & (df["answer"].str.len() > 20)]
    df = df.drop_duplicates(subset=["subject", "body"]).reset_index(drop=True)
    df["tags"] = df[TAG_COLS].apply(
        lambda r: sorted({str(t).strip() for t in r if isinstance(t, str) and t.strip()}), axis=1)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="index only the first N tickets (0 = all)")
    ap.add_argument("--holdout", type=int, default=40, help="tickets held out for evaluation")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    t0 = time.time()
    df = load_english(download())
    print(f"English tickets after cleaning/dedupe: {len(df):,}")

    holdout = df.sample(n=args.holdout, random_state=42)
    df = df.drop(holdout.index).reset_index(drop=True)
    with open(settings.eval_holdout, "w", encoding="utf-8") as fh:
        for i, r in enumerate(holdout.itertuples()):
            fh.write(json.dumps({"id": f"EVAL-{i:03d}", "subject": r.subject, "body": r.body, "answer": r.answer,
                                 "queue": r.queue, "type": r.type, "priority": r.priority,
                                 "tags": list(r.tags)}) + "\n")
    print(f"Held out {len(holdout)} tickets for evaluation -> {settings.eval_holdout}")

    if args.limit:
        df = df.head(args.limit)

    records = [{
        "id": f"HT-{i:05d}", "case_id": f"HT-{i:05d}", "subject": r.subject, "body": r.body, "answer": r.answer,
        "type": r.type, "queue": r.queue, "priority": r.priority, "tags": list(r.tags), "source": "historical",
    } for i, r in enumerate(df.itertuples())]

    db.init_db()
    db.reset_historical()
    db.insert_historical(records)
    print(f"SQLite: stored {len(records):,} historical tickets in {settings.app_db}")

    index = get_index()
    print(f"Qdrant: {'server ' + settings.qdrant_url if settings.qdrant_url else 'embedded at ' + settings.qdrant_path}")
    index.ensure_collection(recreate=True)

    def progress(done, total):
        el = time.time() - t0
        print(f"  indexed {done:,}/{total:,}  ({el:,.0f}s elapsed)", flush=True)

    # Sorting by length keeps ONNX batches uniformly sized (much less padding work on CPU).
    records.sort(key=lambda r: len(r["subject"]) + len(r["body"]))
    index.upsert_cases(records, batch_size=args.batch, progress=progress)
    print(f"Done. Collection '{settings.collection}' has {index.count():,} points. Total {time.time() - t0:,.0f}s")
    index.client.close()


if __name__ == "__main__":
    main()
