import os
import sqlite3
import time
import re
import argparse
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

import config_full as config

DB_PATH = config.SQLITE_DB_FULL
CHROMA_DIR = config.CHROMA_DIR_FULL

COLLECTION_NAME = getattr(config, "CHROMA_COLLECTION_FULL", "papers_all")
EMBED_MODEL = getattr(config, "SENTENCE_TFORMER", None) or getattr(
    config, "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

PAPERS_PER_BATCH = int(getattr(config, "PAPERS_PER_BATCH", 200))
CHROMA_MAX_BATCH = int(getattr(config, "CHROMA_MAX_BATCH", 5400))
CHUNK_MAX_CHARS = int(getattr(config, "CHUNK_MAX_CHARS", 12000))

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def safe_meta(val: Any, default: Any = "N/A") -> Any:
    if val is None:
        return default
    if isinstance(val, (int, float, bool)):
        return val
    s = str(val).strip()
    return s if s else default


def _pick_year(pub_date: Any) -> str:
    s = "" if pub_date is None else str(pub_date)
    m = _YEAR_RE.search(s)
    return m.group(0) if m else safe_meta(pub_date)


def _split_text(text: str, max_chars: int = 12000) -> List[str]:
    if not text:
        return []
    t = text.strip()
    if len(t) <= max_chars:
        return [t]

    parts: List[str] = []
    i = 0
    n = len(t)
    while i < n:
        j = min(i + max_chars, n)
        parts.append(t[i:j])
        i = j
    return parts


def detect_works_fulltext_column(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(works);")
    cols = [r[1] for r in cur.fetchall()]
    if "full_text" in cols:
        return "full_text"
    if "fultext" in cols:
        return "fultext"
    if "fulltext" in cols:
        return "fulltext"
    raise RuntimeError("Could not find works fulltext column. Expected one of: full_text, fultext, fulltext")


def build_paper_text(title: Any, summary: Any, fulltext: Any) -> str:
    t = (title or "").strip()
    s = (summary or "").strip()
    f = (fulltext or "").strip()

    parts: List[str] = []
    if t:
        parts.append(t)
    if s:
        parts.append(s)
    if f:
        parts.append(f)

    return "\n\n".join(parts).strip()


def iter_rows(conn: sqlite3.Connection, works_fulltext_col: str, fetch_size: int = 8000):
    cur = conn.cursor()
    sql = f"""
        SELECT
            r.paper_id,
            r.id,
            r.researcher_name,
            r.work_title,
            r.authors,
            r.info,
            r.doi,
            r.publication_date,
            r.primary_topic,
            w.summary,
            w.{works_fulltext_col}
        FROM research_info r
        LEFT JOIN works w
          ON r.paper_id = w.paper_id
    """
    cur.execute(sql)
    while True:
        rows = cur.fetchmany(fetch_size)
        if not rows:
            break
        for row in rows:
            yield row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="Delete and rebuild collection")
    args = ap.parse_args()

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    if args.rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Old collection removed: {COLLECTION_NAME}")
        except Exception:
            pass

    col = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder)

    conn = sqlite3.connect(DB_PATH)
    works_fulltext_col = detect_works_fulltext_column(conn)

    by_paper: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    for row in iter_rows(conn, works_fulltext_col, fetch_size=8000):
        (
            paper_id,
            r_id,
            researcher_name,
            work_title,
            authors,
            info,
            doi,
            publication_date,
            primary_topic,
            summary,
            fulltext,
        ) = row

        pid = str(paper_id).strip() if paper_id is not None else ""
        if not pid:
            continue

        doc_text = build_paper_text(work_title, summary, fulltext)
        if not doc_text:
            continue

        meta = {
            "paper_id": safe_meta(pid),
            "research_info_id": safe_meta(r_id),
            "researcher": safe_meta(researcher_name, "Unknown"),
            "title": safe_meta((work_title or "").strip(), "Untitled"),
            "authors": safe_meta(authors),
            "doi": safe_meta(doi),
            "year": _pick_year(publication_date),
            "publication_date": safe_meta(publication_date),
            "primary_topic": safe_meta(primary_topic),
        }

        prev = by_paper.get(pid)
        if prev is None or len(doc_text) > len(prev[0]):
            by_paper[pid] = (doc_text, meta)

    conn.close()

    paper_ids = list(by_paper.keys())
    print(f"Papers prepared: {len(paper_ids)}")

    total_start = time.time()

    for batch_idx in tqdm(range(0, len(paper_ids), PAPERS_PER_BATCH), desc="Ingesting papers", unit="batch"):
        ids_batch = paper_ids[batch_idx : batch_idx + PAPERS_PER_BATCH]

        docs: List[str] = []
        ids: List[str] = []
        metas: List[Dict[str, Any]] = []

        for pid in ids_batch:
            doc_text, meta = by_paper[pid]
            chunks = _split_text(doc_text, max_chars=CHUNK_MAX_CHARS)
            if not chunks:
                continue

            for part_i, chunk in enumerate(chunks, start=1):
                docs.append(chunk)
                ids.append(f"{pid}_part{part_i}")
                m = dict(meta)
                m["chunk"] = part_i
                m["chunks_total"] = len(chunks)
                metas.append(m)

        if not docs:
            continue

        batch_start = time.time()
        for i in range(0, len(docs), CHROMA_MAX_BATCH):
            col.upsert(
                documents=docs[i : i + CHROMA_MAX_BATCH],
                ids=ids[i : i + CHROMA_MAX_BATCH],
                metadatas=metas[i : i + CHROMA_MAX_BATCH],
            )

        speed = len(docs) / (time.time() - batch_start + 1e-6)
        bnum = (batch_idx // PAPERS_PER_BATCH) + 1
        print(f"Batch {bnum}: {len(docs)} chunks at {speed:.2f} chunks/sec")

    print(f"Ingestion complete into {COLLECTION_NAME}")
    print(f"Total time: {time.time() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()
