"""Tests for the RAG memory module — ingest -> embed -> store -> retrieve."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from store import Store
from engine import make_engine
from memory import (
    Memory, InMemoryVectorStore, chunk_text, approx_tokens, _cosine,
    _keyword_overlap,
)


def make_memory(tmp="/tmp/aria_mem_test"):
    os.makedirs(tmp, exist_ok=True)
    store = Store(":memory:")
    eng = make_engine("fake", tmp + "/models", tmp + "/adapters", dim=256)
    vs = InMemoryVectorStore()
    return Memory(store, eng, vs, embedding_model="fake", top_k=3), store


def test_chunking_basic():
    assert chunk_text("") == []
    assert chunk_text("a b c") == ["a b c"]
    words = " ".join(f"w{i}" for i in range(2000))
    chunks = chunk_text(words, chunk_size=800, overlap=120)
    assert len(chunks) >= 3
    # overlap: consecutive chunks share tail/head words
    c0 = chunks[0].split()
    c1 = chunks[1].split()
    assert c0[-1] != c1[0]  # they advance
    assert set(c0[-120:]) & set(c1[:120])  # but overlap


def test_approx_tokens():
    assert approx_tokens("one two three") >= 3


def test_ingest_and_retrieve():
    mem, store = make_memory()
    ids = mem.ingest_text(
        "The capital of France is Paris. The Eiffel Tower is in Paris.",
        source="geo.txt", source_type="file")
    assert len(ids) == 1
    mem.ingest_text(
        "Python is a programming language. It is used for data science.",
        source="prog.txt", source_type="file")
    mem.ingest_text(
        "The mitochondria is the powerhouse of the cell.",
        source="bio.txt", source_type="file")

    # a France query should surface the geo chunk first
    hits = mem.retrieve("What is the capital of France?", k=3)
    assert hits, "expected retrieval hits"
    assert "Paris" in hits[0].text
    assert hits[0].source == "geo.txt"


def test_retrieval_bumps_usage():
    mem, store = make_memory()
    ids = mem.ingest_text("alpha beta gamma delta", source="s.txt")
    cid = ids[0]
    before = store.query("SELECT retrieval_count FROM memory_chunks WHERE id=?", (cid,))[0]["retrieval_count"]
    assert before == 0
    mem.retrieve("alpha beta", k=1)
    after = store.query("SELECT retrieval_count, last_retrieved FROM memory_chunks WHERE id=?", (cid,))[0]
    assert after["retrieval_count"] == 1
    assert after["last_retrieved"] is not None


def test_build_context_and_stats():
    mem, store = make_memory()
    mem.ingest_text("Neptune is the eighth planet from the Sun.", source="space.txt")
    ctx = mem.build_context("Which planet is eighth from the Sun?", k=2)
    assert "Neptune" in ctx
    assert "source:" in ctx
    s = mem.stats()
    assert s["chunks"] == 1 and s["vectors"] == 1


def test_delete_chunk():
    mem, store = make_memory()
    ids = mem.ingest_text("delete me please", source="d.txt")
    assert mem.stats()["vectors"] == 1
    mem.delete_chunk(ids[0])
    assert mem.stats()["vectors"] == 0
    assert mem.stats()["chunks"] == 0


def test_cosine_identity():
    v = [0.1, 0.2, 0.3]
    assert abs(_cosine(v, v) - 1.0) < 1e-9


def test_keyword_overlap():
    assert _keyword_overlap("favorite color", "My favorite color is teal.") == 1.0
    assert _keyword_overlap("random unrelated query", "My favorite color is teal.") == 0.0
    assert _keyword_overlap("", "anything") == 0.0


class _StubEmbedder:
    """Embeds nothing meaningful — used with _FixedRankVectorStore so the
    test controls raw vector-search ranking directly instead of depending
    on any particular embedding model's quality."""
    def embed(self, texts):
        return [[0.0] for _ in texts]


class _FixedRankVectorStore:
    """Ignores the actual query vector; always ranks by insertion order
    (earliest-added chunk = highest raw vector score). Used to simulate a
    real observed live failure: pure vector search burying a chunk that
    literally contains the answer beneath chunks that merely score higher
    on embedding similarity (e.g. a "favorite color" query not surfacing a
    chunk reading "My favorite color is teal.")."""

    def __init__(self, scores):
        self._order = []
        self._texts = {}
        self._scores = scores  # raw vector score per insertion index

    def add(self, ids, vectors, texts):
        for i, t in zip(ids, texts):
            self._order.append(i)
            self._texts[i] = t

    def search(self, vector, k):
        hits = [(cid, self._scores[idx]) for idx, cid in enumerate(self._order)]
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:k]

    def delete(self, ids):
        for i in ids:
            self._texts.pop(i, None)

    def count(self):
        return len(self._texts)


def test_retrieve_reranks_verbatim_keyword_match_above_weak_vector_score():
    store = Store(":memory:")
    # The verbatim match is inserted last and given the *lowest* raw vector
    # score on purpose — reproducing the bug where pure vector search ranked
    # it below three keyword-irrelevant chunks.
    vs = _FixedRankVectorStore(scores=[0.95, 0.90, 0.85, 0.80])
    mem = Memory(store, _StubEmbedder(), vs, embedding_model="fake", top_k=3)
    mem.ingest_text("The user enjoys hiking on weekends.", source="a.txt")
    mem.ingest_text("The user's dog is named Comet.", source="b.txt")
    mem.ingest_text("The user works as a software engineer.", source="c.txt")
    mem.ingest_text("My favorite color is teal.", source="d.txt")

    hits = mem.retrieve("What is my favorite color?", k=3)
    assert any("teal" in h.text for h in hits), (
        "verbatim match should surface in top-k even though its raw vector "
        "score was the lowest of all four candidates"
    )
    assert hits[0].source == "d.txt"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
