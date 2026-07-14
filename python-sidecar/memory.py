"""RAG memory module — Mechanism A of the self-improvement loop.

Ingest documents/chat -> chunk -> embed -> store (SQLite metadata + vector
store) -> retrieve top-k at query time. Facts live here; this updates the
instant you add data and never corrupts the model.

The vector store is pluggable:
  * ``InMemoryVectorStore`` — pure-Python cosine search. Zero deps; used for
    tests, offline dev, and small corpora.
  * ``LanceVectorStore``    — embedded LanceDB (ANN, on-disk, scales). Used in
    production; imported lazily so the module works without lancedb installed.

Embeddings come from the engine (``engine.embed``), so the same memory code runs
with MLX embeddings on Mac, llama.cpp elsewhere, or the fake hashed embedder in
tests.
"""
from __future__ import annotations

import math
import os
import re
import uuid
from dataclasses import dataclass
from typing import Optional, Protocol

from store import Store, now_ms


# --------------------------------------------------------------------------
# Auto-fact extraction — zero-cost heuristics, not a general NLU extractor.
# Deliberately conservative (high-precision phrasing only) so it never fires
# on ordinary conversation and never costs an extra model call. Anything
# outside these patterns still goes through Settings -> Memory, or an
# explicit "remember ..." instruction, which the catch-all pattern below
# handles regardless of topic.
# --------------------------------------------------------------------------
_REMEMBER_RE = re.compile(
    r"^\s*(?:please\s+)?(?:remember|note|save)(?:\s+this)?(?:\s+that)?\s*[:\-]?\s*(.{4,300})$",
    re.I,
)
# Every capture stops at punctuation, a clause-joining conjunction, or end of
# string — without this, "my name is X and I live in Y" would swallow the
# whole rest of the sentence into the name.
_STOP = r"(?=[.!,;]|\s+(?:and|but|so|because|who|which|today|right now)\b|$)"
_AUTO_FACT_PATTERNS = [
    (re.compile(r"\bmy name is ([A-Za-z][\w' -]{0,40}?)" + _STOP, re.I),
     "The user's name is {0}."),
    (re.compile(r"\bcall me ([A-Za-z][\w' -]{0,40}?)" + _STOP, re.I),
     "The user prefers to be called {0}."),
    (re.compile(r"\bi live in ([A-Za-z][\w' ,-]{0,60}?)" + _STOP, re.I),
     "The user lives in {0}."),
    (re.compile(r"\bi'?m from ([A-Za-z][\w' ,-]{0,60}?)" + _STOP, re.I),
     "The user is from {0}."),
    (re.compile(r"\bi work as (?:an?|the)?\s*([A-Za-z][\w' -]{0,60}?)" + _STOP, re.I),
     "The user works as {0}."),
    (re.compile(r"\bmy job is ([A-Za-z][\w' -]{0,60}?)" + _STOP, re.I),
     "The user's job is {0}."),
    (re.compile(r"\bmy favorite ([\w -]{1,20}?) is ([A-Za-z0-9][\w' -]{0,60}?)" + _STOP, re.I),
     "The user's favorite {0} is {1}."),
]


def extract_auto_facts(text: str) -> list[str]:
    """Pull durable personal facts out of a single chat message, if any.

    An explicit "remember/save/note ..." instruction always wins and is
    stored verbatim (that's the user directly asking); otherwise we scan for
    a small set of high-precision self-disclosure patterns (name, location,
    job, favorites). Returns a list of complete sentences ready to store.
    """
    text = text.strip()
    if not text:
        return []
    m = _REMEMBER_RE.match(text)
    if m:
        fact = m.group(1).strip().rstrip(".")
        return [fact + "."] if fact else []
    facts = []
    for pattern, template in _AUTO_FACT_PATTERNS:
        m = pattern.search(text)
        if m:
            facts.append(template.format(*[g.strip() for g in m.groups()]))
    return facts


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into overlapping word-windows.

    Word-based (not char-based) so chunks align roughly to token budgets.
    ``chunk_size``/``overlap`` are in words.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks, start = [], 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += step
    return chunks


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~0.75 words/token heuristic)."""
    return int(len(text.split()) / 0.75) + 1


# --------------------------------------------------------------------------
# Vector store interface + implementations
# --------------------------------------------------------------------------
class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: list[list[float]], texts: list[str]) -> None: ...
    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Exact cosine search in Python. Fine for tests + small corpora."""

    def __init__(self):
        self._vecs: dict[str, list[float]] = {}
        self._texts: dict[str, str] = {}

    def add(self, ids, vectors, texts):
        for i, v, t in zip(ids, vectors, texts):
            self._vecs[i] = v
            self._texts[i] = t

    def search(self, vector, k):
        scored = [(i, _cosine(vector, v)) for i, v in self._vecs.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def delete(self, ids):
        for i in ids:
            self._vecs.pop(i, None)
            self._texts.pop(i, None)

    def count(self):
        return len(self._vecs)


class LanceVectorStore:
    """Embedded LanceDB vector store (production). Lazy import."""

    def __init__(self, path: str, dim: int, table: str = "memory"):
        try:
            import lancedb
            import pyarrow as pa  # noqa: F401
        except ImportError as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "lancedb not installed. Run: pip install lancedb pyarrow\n"
                "Or use InMemoryVectorStore for small corpora / tests."
            ) from e
        self._lancedb = lancedb
        self.dim = dim
        self.db = lancedb.connect(path)
        self.table_name = table
        self._table = None

    def _ensure_table(self, sample_vec):  # pragma: no cover - optional dep
        if self._table is not None:
            return
        import pyarrow as pa
        if self.table_name in self.db.table_names():
            self._table = self.db.open_table(self.table_name)
        else:
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.dim)),
                pa.field("text", pa.string()),
            ])
            self._table = self.db.create_table(self.table_name, schema=schema)

    def add(self, ids, vectors, texts):  # pragma: no cover - optional dep
        self._ensure_table(vectors[0] if vectors else None)
        rows = [{"id": i, "vector": v, "text": t}
                for i, v, t in zip(ids, vectors, texts)]
        self._table.add(rows)

    def search(self, vector, k):  # pragma: no cover - optional dep
        self._ensure_table(vector)
        res = self._table.search(vector).limit(k).to_list()
        # LanceDB returns _distance (L2); convert to a similarity-ish score.
        return [(r["id"], 1.0 / (1.0 + r.get("_distance", 0.0))) for r in res]

    def delete(self, ids):  # pragma: no cover - optional dep
        self._ensure_table(None)
        id_list = ",".join(f"'{i}'" for i in ids)
        self._table.delete(f"id IN ({id_list})")

    def count(self):  # pragma: no cover - optional dep
        if self._table is None and self.table_name in self.db.table_names():
            self._table = self.db.open_table(self.table_name)
        return self._table.count_rows() if self._table else 0


# --------------------------------------------------------------------------
# Embedder protocol (satisfied by any EngineDriver)
# --------------------------------------------------------------------------
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    source: str


# --------------------------------------------------------------------------
# Memory manager
# --------------------------------------------------------------------------
class Memory:
    """Orchestrates chunk -> embed -> store -> retrieve."""

    def __init__(self, store: Store, embedder: Embedder,
                 vector_store: VectorStore,
                 embedding_model: str = "fake",
                 top_k: int = 5):
        self.store = store
        self.embedder = embedder
        self.vs = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k

    # ---- ingest ----------------------------------------------------------
    def ingest_text(self, text: str, source: str, source_type: str = "note",
                    metadata: Optional[dict] = None) -> list[str]:
        """Chunk, embed, and store a piece of text. Returns new chunk ids."""
        import json
        chunks = chunk_text(text)
        if not chunks:
            return []
        vectors = self.embedder.embed(chunks)
        ids, ts = [], now_ms()
        rows = []
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            cid = str(uuid.uuid4())
            ids.append(cid)
            rows.append((cid, source, source_type, chunk, approx_tokens(chunk),
                         idx, self.embedding_model, ts, 0, None,
                         json.dumps(metadata or {})))
        self.store.conn.executemany(
            "INSERT INTO memory_chunks (id,source,source_type,text,token_count,"
            "chunk_index,embedding_model,created_at,retrieval_count,"
            "last_retrieved,metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.store.conn.commit()
        self.vs.add(ids, vectors, chunks)
        return ids

    def ingest_fact_if_new(self, text: str, source: str = "auto") -> Optional[str]:
        """Store a short fact unless an identical one is already remembered —
        stops the same self-disclosure ("my name is X") from re-saving a
        duplicate chunk every time it's repeated across a conversation."""
        existing = self.store.query(
            "SELECT id FROM memory_chunks WHERE lower(text)=lower(?) LIMIT 1",
            (text.strip(),),
        )
        if existing:
            return None
        ids = self.ingest_text(text, source=source)
        return text if ids else None

    def ingest_file(self, path: str, metadata: Optional[dict] = None) -> list[str]:
        with open(path, "r", errors="replace") as f:
            text = f.read()
        return self.ingest_text(text, source=path, source_type="file",
                                metadata=metadata)

    # ---- retrieve --------------------------------------------------------
    def retrieve(self, query: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        """Return the top-k most relevant chunks and bump their usage stats."""
        # Embedding runs on the same serialized GPU worker as generation, so
        # an empty store would still cost an embed round-trip in front of
        # every chat reply. Skip it outright when there's nothing to search.
        if self.vs.count() == 0:
            return []
        k = k or self.top_k
        qvec = self.embedder.embed([query])[0]
        hits = self.vs.search(qvec, k)
        if not hits:
            return []
        id_to_score = dict(hits)
        placeholders = ",".join("?" for _ in id_to_score)
        rows = self.store.query(
            f"SELECT id,text,source FROM memory_chunks WHERE id IN ({placeholders})",
            tuple(id_to_score.keys()),
        )
        by_id = {r["id"]: r for r in rows}
        results = []
        ts = now_ms()
        for cid, score in hits:
            r = by_id.get(cid)
            if r is None:
                continue
            results.append(RetrievedChunk(cid, r["text"], score, r["source"]))
            self.store.conn.execute(
                "UPDATE memory_chunks SET retrieval_count=retrieval_count+1, "
                "last_retrieved=? WHERE id=?", (ts, cid),
            )
        self.store.conn.commit()
        return results

    def build_context(self, query: str, k: Optional[int] = None,
                      max_chars: int = 4000) -> str:
        """Retrieve and format chunks into a context block for the prompt."""
        chunks = self.retrieve(query, k)
        parts, total = [], 0
        for c in chunks:
            block = f"[source: {os.path.basename(c.source)}]\n{c.text}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)

    # ---- management ------------------------------------------------------
    def delete_chunk(self, chunk_id: str) -> None:
        self.store.conn.execute("DELETE FROM memory_chunks WHERE id=?", (chunk_id,))
        self.store.conn.commit()
        self.vs.delete([chunk_id])

    def list_chunks(self, limit: int = 200, offset: int = 0) -> list[dict]:
        rows = self.store.query(
            "SELECT id,source,source_type,substr(text,1,200) AS preview,"
            "token_count,retrieval_count,last_retrieved,created_at "
            "FROM memory_chunks ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        row = self.store.query(
            "SELECT COUNT(*) AS n, COALESCE(SUM(token_count),0) AS toks, "
            "COALESCE(SUM(retrieval_count),0) AS retrievals FROM memory_chunks"
        )[0]
        return {"chunks": row["n"], "tokens": row["toks"],
                "retrievals": row["retrievals"], "vectors": self.vs.count()}
