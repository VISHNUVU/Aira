# Data Schemas

All persistent state lives in one SQLite database (`~/.aria/aria.db`) plus a
vector store (`~/.aria/vectors/`). SQLite holds metadata and relationships; the
vector store holds embedding vectors keyed by `chunk_id`. This split keeps
metadata queryable with SQL while letting the vector index do ANN search.

Schema version is tracked in the `meta` table so migrations are explicit.

---

## 1. `memory_chunks` — the RAG memory store

One row per indexed chunk of user data. The embedding vector itself lives in the
vector store under the same `id`; this table is the queryable metadata + source
of truth for what the model can recall.

```sql
CREATE TABLE memory_chunks (
    id              TEXT PRIMARY KEY,       -- uuid4
    source          TEXT NOT NULL,          -- file path / "chat" / URL
    source_type     TEXT NOT NULL,          -- 'file' | 'chat' | 'note' | 'web'
    text            TEXT NOT NULL,          -- the chunk content
    token_count     INTEGER,                -- approx tokens (for budgeting)
    chunk_index     INTEGER,                -- position within source doc
    embedding_model TEXT NOT NULL,          -- e.g. 'nomic-embed-text-v1.5'
    created_at      INTEGER NOT NULL,       -- epoch ms
    retrieval_count INTEGER NOT NULL DEFAULT 0,   -- times returned in top-k
    last_retrieved  INTEGER,                -- epoch ms, nullable
    metadata        TEXT                    -- JSON blob: page, heading, tags...
);
CREATE INDEX idx_memory_source ON memory_chunks(source);
CREATE INDEX idx_memory_created ON memory_chunks(created_at);
```

Vector store record (LanceDB/Chroma): `{ id, vector: float[dim], text }` — `id`
matches `memory_chunks.id`. `retrieval_count`/`last_retrieved` are bumped on every
retrieval so the Memory browser can show what's actually being used.

---

## 2. `examples` — the training-example store

One row per validated feedback event, normalized to an instruction→preferred-
response pair. This is the fuel for QLoRA.

```sql
CREATE TABLE examples (
    id               TEXT PRIMARY KEY,      -- uuid4
    instruction      TEXT NOT NULL,         -- the user prompt / task
    context          TEXT,                  -- optional retrieved context used
    preferred_output TEXT NOT NULL,         -- the good answer (edited/approved)
    rejected_output  TEXT,                  -- optional: the original, if edited
    feedback_type    TEXT NOT NULL,         -- 'thumbs_up' | 'edit' | 'correction'
    dedup_hash       TEXT NOT NULL,         -- sha256(instruction+preferred) — unique
    used_in_run      TEXT,                  -- training_run.id once consumed, else NULL
    split            TEXT,                  -- 'train' | 'held_out' | NULL (unassigned)
    created_at       INTEGER NOT NULL,      -- epoch ms
    source_turn_id   TEXT                   -- optional link to the chat turn
);
CREATE UNIQUE INDEX idx_examples_dedup ON examples(dedup_hash);
CREATE INDEX idx_examples_unused ON examples(used_in_run) WHERE used_in_run IS NULL;
```

**Trigger rule:** when `COUNT(*) WHERE used_in_run IS NULL >= TRAIN_THRESHOLD`
(default 200), the trainer is eligible to run. Exported to JSONL for the engine in
the Gemma 4 chat format:
```json
{"messages":[{"role":"user","content":"<instruction>"},
             {"role":"assistant","content":"<preferred_output>"}]}
```

---

## 3. `adapters` — the adapter registry

One row per LoRA adapter ever produced. Exactly one may be `is_active = 1`.

```sql
CREATE TABLE adapters (
    id             TEXT PRIMARY KEY,        -- 'v0001', 'v0002', ...
    base_model     TEXT NOT NULL,           -- 'gemma-4-e4b'
    path           TEXT NOT NULL,           -- ~/.aria/adapters/v0001
    training_run   TEXT NOT NULL,           -- FK training_runs.id
    eval_score     REAL,                    -- primary metric on held-out set
    baseline_score REAL,                    -- active adapter's score at eval time
    promoted       INTEGER NOT NULL,        -- 1 if it beat baseline
    is_active      INTEGER NOT NULL DEFAULT 0,
    n_examples     INTEGER,                 -- examples used to train it
    created_at     INTEGER NOT NULL,
    notes          TEXT
);
CREATE UNIQUE INDEX idx_adapters_active ON adapters(is_active) WHERE is_active = 1;
```

The partial unique index enforces the "at most one active adapter" invariant at
the DB level. Rollback = set `is_active=0` everywhere, `is_active=1` on the chosen
version, and tell the engine to `set_adapter(path)`.

---

## 4. `training_runs` — training run log

One row per fine-tune attempt, whether or not it promoted.

```sql
CREATE TABLE training_runs (
    id             TEXT PRIMARY KEY,        -- uuid4
    status         TEXT NOT NULL,           -- 'queued'|'running'|'evaluating'|
                                            -- 'promoted'|'rejected'|'failed'
    base_model     TEXT NOT NULL,
    n_train        INTEGER,
    n_held_out     INTEGER,
    config         TEXT,                    -- JSON: lr, rank, iters, batch...
    train_loss     TEXT,                    -- JSON array of loss points
    eval_score     REAL,
    baseline_score REAL,
    adapter_id     TEXT,                    -- FK adapters.id if produced
    started_at     INTEGER,
    finished_at    INTEGER,
    error          TEXT                     -- populated on 'failed'
);
CREATE INDEX idx_runs_status ON training_runs(status);
```

---

## 5. `tool_calls` — the tool invocation log

Every function call the model makes, for the Tools panel audit log.

```sql
CREATE TABLE tool_calls (
    id           TEXT PRIMARY KEY,          -- uuid4
    tool_name    TEXT NOT NULL,
    arguments    TEXT NOT NULL,             -- JSON of args the model supplied
    result       TEXT,                      -- JSON/text result (truncated for log)
    status       TEXT NOT NULL,             -- 'ok' | 'error' | 'denied'
    duration_ms  INTEGER,
    turn_id      TEXT,                      -- chat turn that triggered it
    created_at   INTEGER NOT NULL
);
CREATE INDEX idx_toolcalls_created ON tool_calls(created_at);
CREATE INDEX idx_toolcalls_name ON tool_calls(tool_name);
```

---

## 6. `meta` — key/value app state

```sql
CREATE TABLE meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
-- rows: schema_version, active_model, train_threshold, embedding_model,
--       last_train_at, tools_enabled (JSON list) ...
```

---

## 7. Config file (`~/.aria/config.json`)

Non-secret runtime config the sidecar reads on boot.

```json
{
  "active_model": "gemma-4-e4b",
  "models_dir": "~/.aria/models",
  "adapters_dir": "~/.aria/adapters",
  "embedding_model": "nomic-embed-text-v1.5",
  "vector_dim": 768,
  "retrieval_top_k": 5,
  "train": {
    "threshold": 200,
    "held_out_frac": 0.15,
    "lora_rank": 16,
    "learning_rate": 1e-4,
    "iters": 600,
    "batch_size": 1,
    "max_seq_len": 4096,
    "min_improvement": 0.0
  },
  "tools_enabled": ["file_search", "current_time"],
  "engine": "mlx"
}
```

---

## 8. Entity relationships

```
examples.used_in_run ──────────► training_runs.id
training_runs.adapter_id ──────► adapters.id
adapters.training_run ─────────► training_runs.id
memory_chunks.id ══════════════► vector_store record (same id)
tool_calls.turn_id ────────────► (chat turn, in-memory / app-level)
```

The `examples ↔ training_runs ↔ adapters` triangle is the audit trail of the
self-improvement loop: which examples produced which run produced which adapter,
and whether it was promoted.
